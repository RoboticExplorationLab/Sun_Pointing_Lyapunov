using LinearAlgebra
using Plots
include("magnetic_field.jl")
using SatelliteDynamics
using DelimitedFiles

#Quaternion functions
function hat(v)
    return [0 -v[3] v[2];
            v[3] 0 -v[1];
            -v[2] v[1] 0]
end

function L(q)
    s = q[1]
    v = q[2:4]
    L = [s    -v';
         v  s*I+hat(v)]
    return L
end

T = Diagonal([1; -ones(3)])
H = [zeros(1,3); I]

function qtoQ(q)
    return H'*T*L(q)*T*L(q)*H
end

function G(q)
    G = L(q)*H
end

function B_igrf(x,t)
    Bmag = IGRF13(x,t)
    return Bmag
end

############################ Defining Constants. Change here to your own spacecraft specs #############################
#Inertia (from CAD)
J = [.0043 -.0003 0.0;
          -.0003 .0049 0.0;
            0.0   0.0 .0035]

umax = 50*.1*.1*0.1
major_axis = [-0.382683; 0.92388; 0.0]
Δt = 0.05 # This will run all simulations at 20Hz
#Random inertia error
V = eigvecs(J)
V_true = V*exp(hat(0.1*randn(3)))
D = eigvals(J)
D_true = D+0.03*J[2,2]*randn(3)
J_true = V_true*Diagonal(D_true)*V_true'
ω_bias = (pi/180)*randn(3)
h_target = 0.2*0.005
##################################Setting epoch and orbit. Change here for different orbits and epochs##################
epc0 = Epoch(2020, 11, 8, 12, 0, 0, 0.0)
#Test of some of the SatelliteDynamics functions

#Orbit we want (around ISS orbit conditions)
sso1 = [6981e3, 0.00064, 95, 1.0804, 27.7899, 190];

# Convert osculating elements to Cartesean state
# returns position and velocity (m, m/s). This is the intial position
eci0_1 = sOSCtoCART(sso1, use_degrees=true)
########################################################################################################################

function ground_truth_sat_dynamics(x, epc)
    
    r = x[1:3] #satellite position in inertial frame
    v = x[4:6] #satellite velocity in inertial frame
        
    #look up this term. seems to give a rotation matrix
    PN = bias_precession_nutation(epc)
    
    #Compute the sun and moon positions in ECI frame
    r_sun = sun_position(epc)
    r_moon = moon_position(epc)
    
    #define the acceleration variable
    a = zeros(eltype(x), 3)
    
    #compute acceleration caused by Earth gravity (includes J2)
    #modeled by a spherical harmonic gravity field
    #look up this term. seems to give a rotation matrix
    #this is the gravity code that is working
    ###########################################################################################################
    #compute the gravitational acceleration based off the series expansion up to J2
    μ = 3.986004418e14 #m3/s2
    J2 = 1.08264e-3 
        
    a_2bp = (-μ*r)/(norm(r))^3
    
    Iz = [0,0,1]
    
    a_J2 = ((3*μ*J2*R_EARTH^2)/(2*norm(r)^5))*((((5*dot(r, Iz)^2)/norm(r)^2)-1)*r - 2*dot(r,Iz)*Iz)     

    a_grav = a_2bp + a_J2
    
    a += a_grav
    ############################################################################################################
    
    #atmospheric drag
    #compute the atmospheric density from density harris priester model
    ρ = density_harris_priester(epc,r)

    
    #computes acceleration due to drag in inertial directions
    cd = 2.0 #drag coefficient
    area_drag = 0.1 #in m2 #area normal to the velocity direction
    m = 1.0
    
    a += accel_drag(x, ρ, m, area_drag, cd, Array{Real,2}(PN))
    
    
    #Solar Radiation Pressure
    area_srp = 1.0
    coef_srp = 1.8
    a += accel_srp(x, r_sun, m, area_srp, coef_srp)
    
    #acceleration due to external bodies
    a+= accel_thirdbody_sun(x, r_sun)
    
    #COMMENTED FOR TESTING
    a+= accel_thirdbody_moon(x, r_moon)
    
            
    xdot = x[4:6]
    vdot = a
    
    x_dot = [xdot; vdot]
    
    return x_dot
    
end

function dynamics(t,x)
    q = x[1:4]
    q = q/norm(q)
    s = x[5:7]
    s = s/norm(s)
    ω = x[8:10]
    pos = x[11:13]
    vel = x[14:16]
    lin_state = [pos; vel]
    q̇ = 0.5*G(q)*ω
    ṡ = hat(s)*ω

    Q = qtoQ(q)
    # b = Q'*B(t)
    b = Q'*B_igrf(pos,epc0+t)

    #u = flat_spin_controller(s,b,ω)
    u = sun_pointing_controller(s,b,ω)
    τ = hat(u)*b
    a = ground_truth_sat_dynamics(lin_state, epc0+t)
    
    ω̇ = J_true\(τ - hat(ω)*J_true*ω)

    return [q̇; ṡ; ω̇  ;a]
end

function sun_pointing_controller(s,b,ω, spin_axis_target = [0; 0; 1], h_target = 0.2*0.005, umax = 0.1)
    h = J*(ω + ω_bias + 0.01*norm(ω)*randn(3))
    
    # h_target = 0.2*0.005
    # spin_axis_target = major_axis
    b = b + 0.01*norm(b)*randn(3)
    s = s + 0.01*norm(s)*randn(3)

    #These error thresholds correspond to about 15 and just under 10 degrees
    if norm(spin_axis_target-h/h_target) > 0.26
        u = hat(b)*(spin_axis_target-h/h_target)
    elseif norm(s-h/norm(h)) > 0.15
        u = hat(b)*(s-h/norm(h))
    else
        #done
        return u = [0; 0; 0]
    end
    
    u = umax*u/norm(u)
end

function rkstep(t,x)
    f1 = dynamics(t,x)
    f2 = dynamics(t+0.5*Δt, x + 0.5*Δt*f1)
    f3 = dynamics(t+0.5*Δt, x + 0.5*Δt*f2)
    f4 = dynamics(t+Δt, x + Δt*f3)
    xn = x + (Δt/6)*(f1 + 2*f2 + 2*f3 + f4)
    xn[1:4] .= xn[1:4]/norm(xn[1:4])
    xn[5:7] .= xn[5:7]/norm(xn[5:7])
    return xn
end

function run_simulation(x0, t0, tf, Δt)
    t = t0
    thist = t0:Δt:tf
    xhist = zeros(length(x0), length(thist))
    xhist[:,1] .= x0
    for k = 1:(length(thist)-1)
        xhist[:,k+1] .= rkstep(t, xhist[:,k])
    end
    return xhist, thist
end

function calculate_errors(x,t, h_target = 0.2*0.005)
    sun_error = zeros(length(t))
    momentum_error = zeros(length(t))
    vel_error = zeros(length(t))
    for k = 1:length(t)
        spin_axis = x[8:10,k]/norm(x[8:10,k]);
        sun_error[k] = (180/pi)*acos(spin_axis'*x[5:7,k])
        vel_error[k] = (180/pi)*acos(spin_axis'*x[14:16,k]/norm(x[14:16,k]))
        momentum_error[k] = norm(major_axis .- J_true*x[8:10,k]/h_target)

    end
    return sun_error, momentum_error, vel_error
end

function plot_sun_error(x, thist)
    sun_error, momentum_error, vel_error = calculate_errors(x, thist)
    plot(thist,sun_error,legend=false, xlabel="Time (s)", ylabel="Sun Pointing Error (degrees)", fontfamily = "Computer Modern")
end

function plot_momentum_error(x, thist)
    sun_error, momentum_error, vel_error = calculate_errors(x, thist)
    plot(thist,momentum_error,legend=false, xlabel="Time (s)", ylabel="Momentum Error", fontfamily = "Computer Modern")
end

function plot_vel_error(x, thist)
    sun_error, momentum_error, vel_error = calculate_errors(x, thist)
    plot(thist,vel_error,legend=false, xlabel="Time (s)", ylabel="Velocity Error (degrees)", fontfamily = "Computer Modern")
end

function plot_trajectory(x)
    plot3d(x[8,:],x[9,:],x[10,:],legend = false, fontfamily = "Computer Modern")
end

function plot_lyapunov(x, thist)
    lyapunov = zeros(length(thist))
    h_hist = J_true*x[8:10,:]
    s_hist = x[5:7,:]
    h_target = 0.2*0.005
    spin_axis_target = major_axis
    for k = 1:length(thist)
        lyapunov[k] = max(max(norm(s_hist[:,k]-h_hist[:,k]/norm(h_hist[:,k])),0.15), max(norm(spin_axis_target-h_hist[:,k]/h_target),0.26))
    end
    idx = 1:100:length(thist)
    h_hist = h_hist*1000
    plot(h_hist[1,idx],h_hist[2,idx],h_hist[3,idx],line_z = lyapunov[idx], colorbar=false, linewidth=2, legend=false, fontfamily = "Computer Modern")
    #plot(thist,lyapunov) 
end

function plot_lyapunov_sun_pointing(x, thist)
    lyapunov = zeros(length(thist))
    h_hist = J_true*x[8:10,:]
    s_hist = x[5:7,:]
    for k = 1:length(thist)
        lyapunov[k] = max(norm(s_hist[:,k]-h_hist[:,k]/norm(h_hist[:,k])),0.15)
    end
    idx = 1:100:length(thist)
    h_hist = h_hist*1000
    plot(h_hist[1,idx],h_hist[2,idx],h_hist[3,idx],line_z = lyapunov[idx], colorbar=true, linewidth=2, legend=false, xlabel="hx (gm2/s)",ylabel="hy (gm2/s)",zlabel="hz (gm2/s)",title="Lyapunov Function")
    #plot(thist,lyapunov) 
end

function plot_lyapunov_flat_spin(x, thist)
    lyapunov = zeros(length(thist))
    h_hist = J_true*x[8:10,:]
    s_hist = x[5:7,:]
    h_target = 0.2*0.005
    spin_axis_target = major_axis
    for k = 1:length(thist)
        lyapunov[k] = max(norm(spin_axis_target-h_hist[:,k]/h_target),0.26)
    end
    idx = 1:100:length(thist)
    h_hist = h_hist*1000
    plot(h_hist[1,idx],h_hist[2,idx],h_hist[3,idx],line_z = lyapunov[idx], colorbar=true, linewidth=2, legend=false, xlabel="hx (gm2/s)",ylabel="hy (gm2/s)",zlabel="hz (gm2/s)",title="Lyapunov Function")
    #plot(thist,lyapunov) 
end

function plot_which_lyapunov(x, thist)
    lyapunov = zeros(length(thist))
    h_hist = J_true*x[8:10,:]
    s_hist = x[5:7,:]
    h_target = 0.2*0.005
    spin_axis_target = major_axis
    for k = 1:length(thist)
        if max(norm(s_hist[:,k]-h_hist[:,k]/norm(h_hist[:,k])),0.15) > max(norm(spin_axis_target-h_hist[:,k]/h_target),0.26)
            lyapunov[k] = 1
        else
            lyapunov[k] = 2
        end
    end
    idx = 1:1:length(thist)
    h_hist = h_hist*1000
    plot(h_hist[1,idx],h_hist[2,idx],h_hist[3,idx],line_z = lyapunov[idx], c =:blues, legend=false,fontfamily = "Computer Modern", size = (600,400) )
    #plot(thist,lyapunov) 
end
function baseline_sun_pointing_controller(s,b,ω)
    h = J*(ω + ω_bias + 0.02*norm(ω)*randn(3))
    α = 0.5
    # h_target = 0.2*0.005
    # spin_axis_target = major_axis
    spin_axis_target = norm(h)*[0.0;1.0;0.0]
    b = b + 0.01*norm(b)*randn(3)
    s = s + 0.01*norm(s)*randn(3)
    #These error thresholds correspond to about 15 and just under 10 degrees
    u = hat(b)*((1-α)*(spin_axis_target-h) + α*(s*norm(h)-h))

    u = umax*u/norm(u)
end

function dynamics_baseline(t,x)
    q = x[1:4]
    q = q/norm(q)
    s = x[5:7]
    s = s/norm(s)
    ω = x[8:10]
    pos = x[11:13]
    vel = x[14:16]
    lin_state = [pos; vel]
    q̇ = 0.5*G(q)*ω
    ṡ = hat(s)*ω

    Q = qtoQ(q)
    # b = Q'*B(t)
    b = Q'*B_igrf(pos,epc0+t)

    #u = flat_spin_controller(s,b,ω)
    u = baseline_sun_pointing_controller(s,b,ω)
    τ = hat(u)*b
    a = ground_truth_sat_dynamics(lin_state, epc0+t)
    
    ω̇ = J_true\(τ - hat(ω)*J_true*ω)

    return [q̇; ṡ; ω̇  ;a]
end

function rkstep_baseline(t,x,Δt)
    f1 = dynamics_baseline(t,x)
    f2 = dynamics_baseline(t+0.5*Δt, x + 0.5*Δt*f1)
    f3 = dynamics_baseline(t+0.5*Δt, x + 0.5*Δt*f2)
    f4 = dynamics_baseline(t+Δt, x + Δt*f3)
    xn = x + (Δt/6)*(f1 + 2*f2 + 2*f3 + f4)
    xn[1:4] .= xn[1:4]/norm(xn[1:4])
    xn[5:7] .= xn[5:7]/norm(xn[5:7])
    return xn
end

function save_csv(x, filename)
    writedlm(filename, x)
end

function run_monte_carlo_sim(N, sso1, Δt, Tfinal, h_target = 0.2*0.005)
    eci0_1 = sOSCtoCART(sso1, use_degrees=true)
    thist = 0:Δt:Tfinal
    sun_error_mc = zeros(length(thist),N)
    moment_error_mc = zeros(length(thist),N)
    xhist_mc = zeros(16,length(thist),N)
    for k = 1:N
        s0 = [1; 0; 0] + 0.1*randn(3)
        s0 = s0/norm(s0)
        ω0 = (10*pi/180)*randn(3)
        q0 = randn(4)
        q0 = q0/norm(q0)
        x0 = [q0; s0; ω0;eci0_1[1:3]; eci0_1[4:6]]
        xhist_mc[:,1,k] .= x0

        for j = 1:(length(thist)-1)
            xhist_mc[:,j+1,k] .= rkstep(thist[j],xhist_mc[:,j,k])
        end
        for j = 1:length(thist)
            spin_axis = xhist_mc[8:10,j,k]/norm(xhist_mc[8:10,j,k])
            sun_error_mc[j,k] = (180/pi)*acos(spin_axis'*xhist_mc[5:7,j,k])
            moment_error_mc[j,k] = norm(major_axis .- J_true*xhist_mc[8:10,j,k]/h_target)
        end
    end
    return sun_error_mc, moment_error_mc,xhist_mc
end

function plot_mc_sun(x_mc,N,thist,titletext)
    idx = 1:10:length(thist)
    plot(thist[idx]/3600,x_mc[idx,:],legend = false, color = :blue, alpha = 0.1, xlabel = "Time (hours)", ylabel = "Sun error (deg)", title = titletext, fontfamily = "Palatino", xticks  = 0:0.5:4.5)
    average_sun_error = zeros(length(idx))
    for k = 1:length(idx)
        average_sun_error[k] = sum(x_mc[idx[k],:])/N
    end
    # display("average Sun tracking error:",average_sun_error[end])
    plot!(thist[idx]/3600,average_sun_error, color = :red, linewidth = 2, label = "Average Sun error")
end

function run_combined_mc(N, sso1, Δt, Tfinal, h_target = 0.2*0.005)
    eci0_1 = sOSCtoCART(sso1, use_degrees=true)
    thist_baseline = 0:Δt:Tfinal
    thist = 0:Δt:Tfinal
    sun_error_mc_baseline = zeros(length(thist_baseline),N)
    momentum_error_mc_baseline = zeros(length(thist_baseline),N)
    xhist_mc_baseline = zeros(16,length(thist_baseline),N)
    sun_error_mc = zeros(length(thist),N)
    momentum_error_mc = zeros(length(thist),N)
    xhist_mc = zeros(16,length(thist),N)
    for k = 1:N
        s0 = [1; 0; 0] + 0.1*randn(3)
        s0 = s0/norm(s0)
        ω0 = (10*pi/180)*randn(3)
        q0 = randn(4)
        q0 = q0/norm(q0)
        x0 = [q0; s0; ω0;eci0_1[1:3]; eci0_1[4:6]]
        xhist_mc_baseline[:,1,k] .= x0
        xhist_mc[:,1,k] .= x0

        for j = 1:(length(thist_baseline)-1)
            xhist_mc_baseline[:,j+1,k] .= rkstep_baseline(thist_baseline[j],xhist_mc_baseline[:,j,k], Δt)
            xhist_mc[:,j+1,k] .= rkstep(thist[j],xhist_mc[:,j,k])
        end
        for j = 1:length(thist_baseline)
            spin_axis_baseline = xhist_mc_baseline[8:10,j,k]/norm(xhist_mc_baseline[8:10,j,k])
            sun_error_mc_baseline[j,k] = (180/pi)*acos(spin_axis_baseline'*xhist_mc_baseline[5:7,j,k])
            momentum_error_mc_baseline[j,k] = norm(major_axis .- J_true*xhist_mc_baseline[8:10,j,k]/norm(J_true*xhist_mc_baseline[8:10,j,k]))
            spin_axis = xhist_mc[8:10,j,k]/norm(xhist_mc[8:10,j,k])
            sun_error_mc[j,k] = (180/pi)*acos(spin_axis'*xhist_mc[5:7,j,k])
            momentum_error_mc[j,k] = norm(major_axis .- J_true*xhist_mc[8:10,j,k]/h_target)
        end
    end
    return sun_error_mc, xhist_mc, momentum_error_mc, sun_error_mc_baseline, xhist_mc_baseline, momentum_error_mc_baseline
end