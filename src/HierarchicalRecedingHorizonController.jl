module HierarchicalRecedingHorizonController
using AutomotiveDrivingModels
using NearestNeighbors

import AutomotiveDrivingModels: get_actions!, observe!, action_context, get_name
import Base.rand
import PyPlot

export
    HRHC,
    accelHRHC,
    mapHRHC,
    GenerateMotionMap,
    screenCollision,
    MotionPrimitives,
    MotionPrimitivesAcceleration,
    QuadraticMask,
    CurveDist,
    wrap_to_π,
    getLegalMoves!,
    getSuccessorStates!,
    getSuccessorStates,
    loopProjectionKD,
    kdProject,
    computeTrajectory,
    screenTrajectory,
    checkCollision,
    generateObstacleMap,
    updateObstacleMap!,
    getObstacleCoords,
    calculateObjective,
    plot_stϕ,
    plotHRHCInfo,
    plotObjectiveHorizon,
    plotSplineRoadway

type HRHC <: DriverModel{NextState, IntegratedContinuous}
    action_context::IntegratedContinuous
    v_cmds # possible velocity commands
    δ_cmds # possible δ_cmds
    ΔXYθ # state changes associated with cmd = (v_command, δ_command)
    legal_ΔXYθ # reachable from current v, δ
    legal_v # reachable from current v, δ
    legal_δ # reachable from current v, δ
    v_idx_low::Int # index lowest reachable v_command
    v_idx_high::Int # index highest reachable v_command
    δ_idx_low::Int # index lowest reachable δ_command
    δ_idx_high::Int # index highest reachable δ_command
    successor_states # reachable next_states

    #car parameters with bicycle geometry model
    car_length::Float64 # wheel base
    car_width::Float64
    car_ID::Int

    # current v, current δ
    v::Float64
    δ::Float64
    curve_ind::Int
    Δs::Float64

    # planning horizon
    h::Int
    Δt::Float64

    # logit level
    k::Int

    # reachable range of V and δ within a single time step
    ΔV₊::Float64
    ΔV₋::Float64
    Δδ::Float64

    V_MIN::Float64
    V_MAX::Float64
    V_STEPS::Int
    δ_MAX::Float64
    δ_MIN::Float64
    δ_STEPS::Int

    # maximum deviation from center of track (if |t| > T_MAX, car is out of bounds)
    T_MAX::Float64

    # Action = Next State
    action::NextState

    function HRHC(car_ID::Int,roadway,context;
        car_length::Float64=4.8,
        car_width::Float64=2.5,
        v::Float64=0.0,
        δ::Float64=0.0,
        h::Int=10,
        Δt::Float64=1.0/24,
        ΔV₊::Float64=1.55,
        ΔV₋::Float64=3.05,
        Δδ::Float64=Float64(π)/12,
        V_MIN::Float64=0.0,
        V_MAX::Float64=100.0,
        V_STEPS::Int=101,
        δ_MAX::Float64=Float64(π)/8,
        δ_MIN::Float64=-Float64(π)/8,
        δ_STEPS::Int=16,
        k::Int=1
        )

        hrhc = new()

        hrhc.V_MIN=V_MIN
        hrhc.V_MAX=V_MAX
        hrhc.V_STEPS=V_STEPS
        hrhc.δ_MAX=δ_MAX
        hrhc.δ_MIN=δ_MIN
        hrhc.δ_STEPS=δ_STEPS
        hrhc.T_MAX=(roadway.segments[1].lanes[1].width - car_width)/2.0

        hrhc.car_ID = car_ID
        hrhc.car_length=car_length
        hrhc.car_width=car_width
        hrhc.h=h
        hrhc.Δt=Δt
        hrhc.ΔV₊=ΔV₊
        hrhc.ΔV₋=ΔV₋
        hrhc.Δδ=Δδ

        hrhc.k=k

        hrhc.v_cmds, hrhc.δ_cmds, hrhc.ΔXYθ = MotionPrimitives(car_length,car_width,h,Δt,V_MIN,V_MAX,V_STEPS,δ_MAX,δ_MIN,δ_STEPS)
        QuadraticMask!(hrhc.ΔXYθ)

        hrhc.v=v
        hrhc.δ=δ
        hrhc.curve_ind=1
        hrhc.Δs=roadway.segments[1].lanes[1].curve[2].s-roadway.segments[1].lanes[1].curve[1].s
        hrhc.action_context=context
        hrhc.action = NextState(VehicleState(VecSE2(0,0,0),0.0))

        hrhc
    end
end
# utility functions
function MotionPrimitives(car_length,car_width,h,Δt,v_min,v_max,v_steps,δ_max,δ_min,δ_steps)
    """
    Return a library of motion primitives (arcs of constant radius) representing short paths that the car can follow.
    """
    # 3D array to store motion_primitives
    motion_primitives = zeros(v_steps,2*δ_steps+1,2) # v: 0,...,99; δ: -45:45, (arc length, +/- radius)

    v = linspace(v_min,v_max,v_steps)*ones(1,2*δ_steps+1)
    δ = (linspace(-δ_max,δ_max,δ_steps*2+1)*ones(1,v_steps))' # steering angle

    motion_primitives[:,:,1] = v*Δt*h # arc length = velocity * time
    motion_primitives[:,:,2] = car_length./sin(δ) # radius of curvature (+ or -)
    motion_primitives[:,1+δ_steps,2] = Inf; # radius of curvature is infinite if steering angle = 0

    destination_primitives = zeros(v_steps,2*δ_steps+1,h,3) # lookup table defining car's location at each of next h time steps

    for i = 1:h
        # angle = 2π * arc_length / r
        dθ = v*Δt*i ./ motion_primitives[:,:,2]

        # dX = abs(radius) * sin(angle)
        destination_primitives[:,:,i,1] = abs(motion_primitives[:,:,2]) .* sin(abs(dθ))
        destination_primitives[:,1+δ_steps,i,1] = v[:,1+δ_steps]*Δt*i # centerline

        # dY = radius * (1 - cos(angle))
        destination_primitives[:,:,i,2] = motion_primitives[:,:,2].*(1 - cos(dθ))
        destination_primitives[:,1+δ_steps,i,2] = 0 # centerline

        destination_primitives[:,:,i,3] = dθ
    end

    # motion_primitives[v, δ, i,j] = s (arc_length), r (radius of curvature)
    # destination_primitives[v, δ, h, 1=Δx,2=Δy,3=Δθ]= changes in x, y and θ after h time steps
    return v, δ, destination_primitives
end
function QuadraticMask!(library)
  """ quadratic logical mask (simplified dynamic model) """
    A = size(library)[1]
    B = size(library)[2]
    X = linspace(0,100,A)*ones(1,B)
    Y = ones(A,1)*linspace(-20,20,B)'
    f = X.^2 + 24*(Y.^2 - 10^2)
    mask = 1.0*(f.<10000)

    #p = PyPlot.scatter(X,Y.*mask)
    #PyPlot.xlabel("velocity")
    #PyPlot.ylabel("steering angle")
    #PyPlot.show()

    if length(size(library)) == 3
        for i in 1 : size(library)[3]
            library[:,:,i] = library[:,:,i] .* mask
        end
    end
    if length(size(library)) == 4
        for i in 1 : size(library)[3]
            for j in 1 : size(library)[4]
                library[:,:,i,j] = library[:,:,i,j] .* mask
            end
        end
    end
end
function CurveDist(pt1::CurvePt, pt2::CurvePt)
    d = sqrt((pt1.pos.x - pt2.pos.x)^2 + (pt1.pos.y - pt2.pos.y)^2)
end
function wrap_to_π(θ)
    θ = θ - div(θ,2*Float64(π))*(2*Float64(π))
    θ = θ + (θ .< -π).*(2*Float64(π)) - (θ .> π).*(2*Float64(π))
end

# HRHC functions
function getLegalMoves!(hrhc::HRHC, scene; h=hrhc.h)
    v_norm = scene.vehicles[hrhc.car_ID].state.v / hrhc.V_MAX

    # Restrict search space to reachable states
    hrhc.v_idx_low = max(1, round(Int,(v_norm - hrhc.ΔV₋/hrhc.V_MAX)*hrhc.V_STEPS)) # index of lowest reachable v in the next time step
    hrhc.v_idx_high = min(hrhc.V_STEPS, 1+round(Int, (v_norm + hrhc.ΔV₊/hrhc.V_MAX)*hrhc.V_STEPS)) # highest reachable v in the next time step

    # Restrict search space to reachable states
    hrhc.δ_idx_low = max(1, (hrhc.δ_STEPS+1) + round(Int,((hrhc.δ - hrhc.Δδ)/(hrhc.δ_MAX - hrhc.δ_MIN))*(2*hrhc.δ_STEPS+1)))
    hrhc.δ_idx_high = min((2*hrhc.δ_STEPS+1), (hrhc.δ_STEPS+1) + round(Int,((hrhc.δ + hrhc.Δδ)/(hrhc.δ_MAX - hrhc.δ_MIN))*(2*hrhc.δ_STEPS+1)))

    # legal_moves = motion_primitives[v_idx_low:v_idx_high,δ_idx_low:δ_idx_high,:]
    hrhc.legal_ΔXYθ = hrhc.ΔXYθ[hrhc.v_idx_low:hrhc.v_idx_high,hrhc.δ_idx_low:hrhc.δ_idx_high,h,:] # ΔX, ΔY, Δθ
    hrhc.legal_v = hrhc.v_cmds[hrhc.v_idx_low:hrhc.v_idx_high,hrhc.δ_idx_low:hrhc.δ_idx_high]
    hrhc.legal_δ = hrhc.δ_cmds[hrhc.v_idx_low:hrhc.v_idx_high,hrhc.δ_idx_low:hrhc.δ_idx_high]

    return
end
function getSuccessorStates!(hrhc::HRHC, scene::Scene)
    """ gets legal successor_states from motion primitives library """
    pos = scene.vehicles[hrhc.car_ID].state.posG # global x,y,z of car

    ΔX = hrhc.legal_ΔXYθ[:,:,1] * cos(pos.θ) + hrhc.legal_ΔXYθ[:,:,2] * -sin(pos.θ)
    ΔY = hrhc.legal_ΔXYθ[:,:,1] * sin(pos.θ) + hrhc.legal_ΔXYθ[:,:,2] * cos(pos.θ)
    Δθ = hrhc.legal_ΔXYθ[:,:,3]

    hrhc.successor_states = zeros(size(hrhc.legal_ΔXYθ))
    hrhc.successor_states[:,:,1] = ΔX + pos.x
    hrhc.successor_states[:,:,2] = ΔY + pos.y
    hrhc.successor_states[:,:,3] = Δθ + pos.θ

    return
end
function loopProjectionKD(hrhc,scene,roadway,tree)
    """
    projects all points in hrhc.successor_states to the kdtree representing
    the spline points along the centerline of roadway
    """
    curve = roadway.segments[1].lanes[1].curve

    s_grid = zeros(size(hrhc.successor_states,1),size(hrhc.successor_states,2))
    t_grid = zeros(size(s_grid))
    ϕ_grid = zeros(size(s_grid))
    idx_grid = zeros(Int,size(s_grid))

    pts = [reshape(hrhc.successor_states[:,:,1],size(hrhc.successor_states[:,:,1],1)*size(hrhc.successor_states[:,:,1],2),1)';
    reshape(hrhc.successor_states[:,:,2],size(hrhc.successor_states[:,:,2],1)*size(hrhc.successor_states[:,:,2],2),1)']
    idxs_list, _ = knn(tree,pts,1)
    idxs=reshape(idxs_list,size(hrhc.successor_states[:,:,2],1),size(hrhc.successor_states[:,:,2],2))


    for i in 1:size(s_grid,1)
        for j in 1:size(s_grid,2)
            idxA = idxs[i,j][1]-1
            idxB = idxs[i,j][1]+1
            if idxs[i,j][1] == length(curve)
                idxB = 1 # wrap to the beginning of the curve
            end
            if idxs[i,j][1] == 1
                idxA = length(curve) # wrap to the end of the curve
            end
            x = hrhc.successor_states[i,j,1]
            y = hrhc.successor_states[i,j,2]
            dA = sqrt(sum(([curve[idxA].pos.x, curve[idxA].pos.y]-[x,y]).^2))
            dB = sqrt(sum(([curve[idxB].pos.x, curve[idxB].pos.y]-[x,y]).^2))
            if dA < dB
                idxB = idxs[i,j][1]
            else
                idxA = idxs[i,j][1]
            end

            # project
            vec1 = [curve[idxB].pos.x - curve[idxA].pos.x, curve[idxB].pos.y - curve[idxA].pos.y, 0]
            vec2 = [x - curve[idxA].pos.x, y - curve[idxA].pos.y, 0]
            idx_t = dot(vec2, vec1)/norm(vec1)^2

            s_θ = curve[idxA].pos.θ + idx_t*(curve[idxB].pos.θ - curve[idxA].pos.θ)

            s_grid[i,j] = curve[idxA].s + idx_t*hrhc.Δs
            t_grid[i,j] = norm(vec2 - idx_t*vec1)*sign(sum(cross(vec1, vec2)))
            ϕ_grid[i,j] = wrap_to_π(hrhc.successor_states[i,j,3] - s_θ)
            idx_grid[i,j] = idxA
        end
    end
    # account for wrap-around
    s_grid[s_grid .< scene.vehicles[hrhc.car_ID].state.posF.s] += curve[end].s + hrhc.Δs

    return s_grid, t_grid, ϕ_grid
end
function kdProject(x,y,θ,tree,roadway,hrhc)
    """
    project single (x,y,Θ) point to roadway spline using kdtree to find the nearest spline point
    """
    curve = roadway.segments[1].lanes[1].curve
    # Δs = roadway.segments[1].lanes[1].curve[2].s - roadway.segments[1].lanes[1].curve[1].s
    idx_list,dist = knn(tree,[x;y],1)
    idx = idx_list[1]
    idxA = idx-1
    idxB = idx+1

    if idx == length(curve)
        idxB = 1 # back to the beginning of the curve
    end
    if idx == 1
        idxA = length(curve)
    end
    dA = sqrt(sum(([curve[idxA].pos.x, curve[idxA].pos.y]-[x,y]).^2))
    dB = sqrt(sum(([curve[idxB].pos.x, curve[idxB].pos.y]-[x,y]).^2))
    if dA < dB
        idxB = idx
    else
        idxA = idx
    end

    # project
    vec1 = [curve[idxB].pos.x - curve[idxA].pos.x, curve[idxB].pos.y - curve[idxA].pos.y, 0]
    vec2 = [x - curve[idxA].pos.x, y - curve[idxA].pos.y, 0]
    idx_t = dot(vec2, vec1)/norm(vec1)^2

    pθ = curve[idxA].pos.θ + idx_t*(curve[idxB].pos.θ - curve[idxA].pos.θ)

    s = curve[idxA].s + idx_t*hrhc.Δs
    t = -norm(vec2 - idx_t*vec1)
    ϕ = wrap_to_π(θ - pθ)

    s,t,ϕ,idxA
end
function computeTrajectory(hrhc::HRHC, scene, cmd_index; h=hrhc.h)
    pos = scene.vehicles[hrhc.car_ID].state.posG

    traj_ΔXYθ = hrhc.ΔXYθ[cmd_index[1],cmd_index[2],1:h,:]

    ΔX = traj_ΔXYθ[:,1] * cos(pos.θ) + traj_ΔXYθ[:,2] * -sin(pos.θ)
    ΔY = traj_ΔXYθ[:,1] * sin(pos.θ) + traj_ΔXYθ[:,2] * cos(pos.θ)
    Δθ = traj_ΔXYθ[:,3]

    trajectory = zeros(size(traj_ΔXYθ,1),size(traj_ΔXYθ,2)+2)
    trajectory[:,1] = ΔX + pos.x
    trajectory[:,2] = ΔY + pos.y
    trajectory[:,3] = Δθ + pos.θ
    trajectory[:,4] = hrhc.v_cmds[cmd_index[1],cmd_index[2]]
    trajectory[:,5] = hrhc.δ_cmds[cmd_index[1],cmd_index[2]]

    return trajectory
end
function screenTrajectory(trajectory, obstacleMap, scene, roadway, hrhc, tree, k_level)
    out_of_bounds = false
    collision_flag = false
    # check out of bounds
    for i in 1 : size(trajectory,1)
        x = trajectory[i,1]
        y = trajectory[i,2]
        θ = trajectory[i,3]
        s,t,ϕ = kdProject(x,y,θ,tree,roadway,hrhc)

        if abs(t) > hrhc.T_MAX
            out_of_bounds=true
            return out_of_bounds
        end
    end
    # check for collision
    # threshold_dist = 4.0*hrhc.car_length
    # if k_level >= 1
    #     for (id,car) in obstacleMap[k_level - 1]
    #         if id != hrhc.car_ID
    #             state = scene.vehicles[hrhc.car_ID].state
    #             state2 = scene.vehicles[id].state
    #             diff = state.posG - state2.posG
    #             s1,_,_ = kdProject(state.posG.x,state.posG.y,state.posG.θ,tree,roadway,hrhc)
    #             s2,_,_ = kdProject(state2.posG.x,state2.posG.y,state2.posG.θ,tree,roadway,hrhc)
    #             if (norm([diff.x, diff.y]) < threshold_dist) && (s1 <= s2) # check which car is ahead
    #                 for i in 1: size(trajectory,1)
    #                     egoCar = VecSE2(trajectory[i,1],trajectory[i,2],trajectory[i,3])
    #                     car2 = VecSE2(car[i,1],car[i,2],car[i,3])
    #                     collision_flag = checkCollision(scene,egoCar,car2,hrhc.car_ID,id)
    #                     if collision_flag
    #                         return collision_flag
    #                     end
    #                 end
    #             end
    #         end
    #     end
    # end

    return out_of_bounds
end
function checkCollision(scene,car1::VecSE2,car2::VecSE2,id1,id2)
    L1 = scene.vehicles[id1].def.length
    w1 = scene.vehicles[id1].def.width
    L2 = scene.vehicles[id2].def.length
    w2 = scene.vehicles[id2].def.width
    diff = car1 - car2
    R = [cos(diff.θ) -sin(diff.θ); sin(diff.θ) cos(diff.θ)]
    car2_pts = R*[L2 L2 -L2 -L2; -w2 w2 w2 -w2]/2.0 + [diff.x 0;0 diff.y]*ones(2,4)
    collision_flag = (sum((abs(car2_pts[1,:]) .>= L1/2.0).*(abs(car2_pts[2,:]) .>= w1/2.0)) > 0)
    return collision_flag
end
function generateObstacleMap(scene, models)
    k = maximum([driver.k for (id, driver) in models])
    n = length(scene)
    h = maximum([driver.h for (id, driver) in models]) # h should be the same for all vehicles on the track
    obstacleDict = Dict()
    for level in 0:k
        obstacleDict[level] = Dict()
        for (id, driver) in models
            obstacleDict[level][id] = zeros(h,5) # x,y,θ,v,δ
        end
    end

    return obstacleDict
end
function updateObstacleMap!(obstacleMap, level, car_ID, trajectory)
    obstacleMap[level][car_ID][1:size(trajectory,1),1:size(trajectory,2)] = trajectory
end
function getObstacleCoords(obstacleMap, level, car_ID, h)
    return obstacleMap[level][car_ID][h,:]
end
function calculateObjective(hrhc, scene, roadway, tree, s, t, ϕ, obstacleMap, k_level, h; f_ϕ=0.0, f_t=0.1, f_tϕ=3.0)
    """
    Calculates the value of the optimization objective function for every state
      in hrhc.successor_states
    """
    state = scene.vehicles[hrhc.car_ID].state
    dS = s - state.posF.s
    dS = dS / maximum(dS) # normalize
    ϕMAX = Float64(π)/2

    # penalize large t (proximity to edge of track)
    cost_t = (exp(((10-h+f_t)*abs(t/hrhc.T_MAX).^2)) )/exp(f_t) + Inf*(t.>hrhc.T_MAX)
    # penalize large ϕ (steering away from forward direction on the track)
    cost_ϕ = (exp(((10-h+f_ϕ)*abs(ϕ/ϕMAX).^2)) )/exp(f_ϕ)
    # penalize when t and ϕ have the same sign
    A = [1 1; 1 1]
    cost_x = (((ϕ/ϕMAX)*A[1,1] + (t/hrhc.T_MAX)*A[2,1]).*(ϕ/ϕMAX) + ((ϕ/ϕMAX)*A[1,2] + (t/hrhc.T_MAX)*A[2,2]).*(t/hrhc.T_MAX))/2
    cost_tϕ = (exp(f_tϕ*cost_x) )/exp(1)
    eligibility_mask = ((hrhc.successor_states[:,:,1] .== state.posG.x).*(hrhc.successor_states[:,:,2] .== state.posG.y))

    # obstacles
    collisionCost = zeros(size(cost_t))
    threshold_dist = hrhc.car_length*4 # must be at least this close before we care to calculate collision cost
    if k_level >= 1
        for (id,car) in obstacleMap[k_level - 1]
            if id != hrhc.car_ID
                state = scene.vehicles[hrhc.car_ID].state
                state2 = scene.vehicles[id].state
                diff = state.posG - state2.posG
                s1,_,_ = kdProject(state.posG.x,state.posG.y,state.posG.θ,tree,roadway,hrhc)
                s2,_,_ = kdProject(state2.posG.x,state2.posG.y,state2.posG.θ,tree,roadway,hrhc)
                if (norm([diff.x, diff.y]) < threshold_dist) && (s1 <= s2) # don't care if opponent is behind us
                    pos = VecSE2(car[h,1:3]) # x,y,θ of opponent at time step h
                    ΔX = hrhc.successor_states[:,:,1] - pos.x # Δx, with opponent at origin
                    ΔY = hrhc.successor_states[:,:,2] - pos.y # Δy with opponent at origin
                    Δθ = hrhc.successor_states[:,:,3] - pos.θ # Δθ with opponent at origin
                    pts = [hrhc.car_length hrhc.car_length -hrhc.car_length -hrhc.car_length 0;
                        -hrhc.car_width hrhc.car_width hrhc.car_width -hrhc.car_width 0]/1.8
                    pX = zeros(size(pts,2),size(hrhc.successor_states,1),size(hrhc.successor_states,2))
                    pY = zeros(size(pX))
                    for i in 1:size(pts,2)
                        pX[i,:,:] = pts[1,i]*cos(Δθ) - pts[2,i]*sin(Δθ) + ΔX
                        pY[i,:,:] = pts[1,i]*sin(Δθ) + pts[2,i].*cos(Δθ) + ΔY
                    end

                    collisionFlag = (maximum((abs(pX) .< hrhc.car_length/1.0),1)[1,:,:]).*(maximum((abs(pY) .< hrhc.car_width/1.9),1)[1,:,:])
                    collisionCost = .001+(collisionFlag .>= 1)./(minimum(abs(pX),1)[1,:,:].*minimum(abs(pY),1)[1,:,:])
                    # collisionCost = Inf.*collisionFlag
                end
            end
        end
    end

    objective = cost_t + cost_ϕ + cost_tϕ + 1 - dS + collisionCost + Inf * eligibility_mask
    return objective
end
function AutomotiveDrivingModels.observe!(hrhc::HRHC, scene::Scene, roadway::Roadway, egoid::Int, tree::KDTree, obstacleMap, k_level)
    """
    Observe the current environment and select optimal action to apply at next
    time step
    """
    if k_level > hrhc.k
        return
    end
    state = scene.vehicles[hrhc.car_ID].state
    hrhc.curve_ind = state.posF.roadind.ind.i
    v = state.v # current v
    hrhc.v = v

    trajectory = zeros(hrhc.h,5)
    action_selected = false
    abs_cmd = (1,1)

    i = 0
    for i in 0:(hrhc.h-1)
        if action_selected
            break # out of for loop
        end

        # get legal (reachable from current v, δ) actions
        getLegalMoves!(hrhc, scene, h=hrhc.h-i)

        # calculate successor states
        getSuccessorStates!(hrhc, scene)

        # project successor states onto track
        s,t,ϕ = loopProjectionKD(hrhc, scene, roadway, tree)

        # optimization objective
        objective = calculateObjective(hrhc, scene, roadway, tree, s, t, ϕ, obstacleMap, k_level, hrhc.h-i,f_t=0.0)

        while (action_selected==false) && (minimum(objective) != Inf)
            index = indmin(objective) # find get a better method of optimizing this
            cmd = ind2sub(s, index)
            abs_cmd = (cmd[1]+hrhc.v_idx_low-1, cmd[2]+hrhc.δ_idx_low-1)

            # compute full trajectory up to horizon
            trajectory = computeTrajectory(hrhc, scene, abs_cmd, h=hrhc.h-i)

            # screen trajectory for collisions / validity
            out_of_bounds = screenTrajectory(trajectory, obstacleMap, scene, roadway, hrhc, tree, k_level)

            if out_of_bounds
                objective[index] = Inf
            else
                action_selected=true
                updateObstacleMap!(obstacleMap, k_level, hrhc.car_ID, trajectory)
            end
        end
    end

    hrhc.δ = hrhc.δ_cmds[abs_cmd[1], abs_cmd[2]]
    hrhc.v = hrhc.v_cmds[abs_cmd[1], abs_cmd[2]]

    next_state = VehicleState(VecSE2(trajectory[1,1:3]),roadway,hrhc.v)
    hrhc.action = NextState(next_state) # action
end
# AutomotiveDrivingModels.observe!(hrhc::HRHC, scene::Scene, roadway::Roadway, egoid::Int, tree::KDTree, obstacleMap, k_level) = observe!
AutomotiveDrivingModels.get_name(::HRHC) = "HRHC"
AutomotiveDrivingModels.action_context(driver::HRHC) = driver.action_context # AutomotiveDrivingModels.action_context
Base.rand(hrhc::HRHC) = hrhc.action


##################### Constant Acceleration-Based HRHC ########################
type accelHRHC <: DriverModel{NextState, IntegratedContinuous}
    action_context::IntegratedContinuous
    v_cmds # possible velocity commands
    δ_cmds # possible δ_cmds
    ΔXYθ # state changes associated with cmd = (v_command, δ_command) - this is the motion map
    eligibility_mask # logical array with zeros at locations of tire force saturation
    v_idx::Int # index of v in v_cmds
    δ_idx::Int # index of δ in δ_cmds
    successor_states # reachable next_states

    #car parameters with bicycle geometry model
    car_length::Float64 # wheel base
    car_width::Float64
    car_ID::Int

    # current v, current δ
    v::Float64
    δ::Float64
    curve_ind::Int
    Δs::Float64

    # planning horizon
    h::Int
    Δt::Float64

    # logit level
    k::Int

    V_MIN::Float64
    V_MAX::Float64

    a_max::Float64 # max acceleration (m/s)
    μ::Float64 # friction coefficient

    # maximum deviation from center of track (if |t| > T_MAX, car is out of bounds)
    T_MAX::Float64

    # Action = Next State
    action::NextState

    function accelHRHC(car_ID::Int,roadway,context;
        car_length::Float64=4.8,
        car_width::Float64=2.5,
        v::Float64=0.0,
        δ::Float64=0.0,
        h::Int=10,
        Δt::Float64=1.0/24,
        ΔV₊::Float64=1.55,
        ΔV₋::Float64=3.05,
        Δδ::Float64=Float64(π)/12,
        V_MIN::Float64=0.0,
        V_MAX::Float64=100.0,
        a_max::Float64=10.0,
        μ::Float64=20.0,
        g::Float64=9.81,
        δ_MAX::Float64=Float64(π)/8,
        δ_MIN::Float64=-Float64(π)/8,
        δ_STEP::Float64=Float64(π)/64,
        k::Int=1
        )

        hrhc = new()

        hrhc.V_MIN=V_MIN
        hrhc.V_MAX=V_MAX

        hrhc.T_MAX=(roadway.segments[1].lanes[1].width - car_width)/2.0

        hrhc.car_ID = car_ID
        hrhc.car_length=car_length
        hrhc.car_width=car_width
        hrhc.h=h
        hrhc.Δt=Δt
        hrhc.a_max=a_max
        hrhc.μ=μ

        hrhc.k=k
        v_step = a_max*Δt

        hrhc.v_cmds, hrhc.δ_cmds, _, hrhc.ΔXYθ, hrhc.eligibility_mask  = MotionPrimitivesAcceleration(car_length,h,Δt,V_MIN,V_MAX,a_max,v_step,δ_MIN,δ_MAX,δ_STEP,μ,g)

        hrhc.v=v
        hrhc.δ=δ
        hrhc.v_idx = 1
        hrhc.δ_idx = Int((size(hrhc.δ_cmds, 2) - 1)/2)
        hrhc.curve_ind=1
        hrhc.Δs=roadway.segments[1].lanes[1].curve[2].s-roadway.segments[1].lanes[1].curve[1].s
        hrhc.action_context=context
        hrhc.action = NextState(VehicleState(VecSE2(0,0,0),0.0))

        hrhc
    end
end
function MotionPrimitivesAcceleration(wheel_base,h,Δt,v_min,v_max,a_max,v_step,δ_min,δ_max,δ_step,μ,g)
    """
    Return a library of motion primitives (arcs of constant radius) representing short paths that the car can follow.
    """
    # 3D array to store motion_primitives
    # radius = zeros(v_steps,2*δ_steps+1) # v: 0,...,99; δ: -45:45, (arc length, +/- radius)

    v = linspace(v_min,v_max,round(v_max/v_step))*ones(1,2*Int(round((δ_max)/δ_step))+1) # velocity
    δ = (linspace(-δ_max,δ_max,2*round((δ_max)/δ_step)+1)*ones(1,Int(round(v_max/v_step))))' # steering angle

    radius = wheel_base./sin(δ) # radius of curvature (+ or -)
    radius[:,Int(round((δ_max)/δ_step))+1] = Inf; # radius of curvature is infinite if steering angle = 0

    n_actions = 3 # slow down, stay, speed up
    destination_primitives = zeros(size(v,1),size(v,2),n_actions,h,3)

    accels = [-a_max,0,a_max]
    for j in 1:3
        v₀ = copy(v)
        a = accels[j]
        for i = 1:h
            vᵢ = max(v₀+a*i*Δt, 0.0) # velocity at end of time step
            tire_force = sqrt((((vᵢ).^2)./abs(radius)).^2 + a^2) # tire force must be less than μ*g
            v_threshold = ones(size(v,1),1)*maximum(vᵢ.*(tire_force .< μ*g), 1)
            vᵢ = min(vᵢ,v_threshold)
            Δs = max((v₀+vᵢ)/2.0, 0.0)*i*Δt

            Δθ = Δs./radius # change in heading
            if j == 3 # accel = -1
                v₀[1:end-i,:] = v[i+1:end,:] # increase v₀ by one step for next round
            elseif j == 1 # accel = +1
                v₀[i+1:end,:] = v[1:end-i,:] # increase v₀ by one step for next round
            end
            # dX = abs(radius) * sin(angle)
            destination_primitives[:,:,j,i,1] = abs(radius) .* sin(abs(Δθ))
            destination_primitives[:,Int(round((δ_max)/δ_step))+1,j,i,1] = Δs[:,Int(round((δ_max)/δ_step))+1] # centerline

            # dY = radius * (1 - cos(angle))
            destination_primitives[:,:,j,i,2] = radius.*(1 - cos(Δθ))
            destination_primitives[:,Int(round((δ_max)/δ_step))+1,j,i,2] = 0 # centerline

            destination_primitives[:,:,j,i,3] = Δθ
        end
    end

    tire_force = sqrt((((v).^2)./abs(radius)).^2) # tire force must be less than μ*g
    mask = (tire_force .< μ*g)
    v_threshold = ones(size(v,1),1)*maximum(v.*mask, 1)
    v = min(v,v_threshold)
    motion_map = Dict()
    for i in 1:size(v,1) # motion_map[i] contains the reachable states from v[i]
        motion_map[i] = destination_primitives[i,:,:,:,:]
    end

    return v, δ, destination_primitives, motion_map, mask
end
function getSuccessorStates!(hrhc::accelHRHC, scene::Scene)
    """ gets legal successor_states from motion primitives library """
    pos = scene.vehicles[hrhc.car_ID].state.posG # global x,y,z of car

    ΔX = hrhc.ΔXYθ[hrhc.v_idx][:,:,hrhc.h,1] * cos(pos.θ) + hrhc.ΔXYθ[hrhc.v_idx][:,:,hrhc.h,2] * -sin(pos.θ)
    ΔY = hrhc.ΔXYθ[hrhc.v_idx][:,:,hrhc.h,1] * sin(pos.θ) + hrhc.ΔXYθ[hrhc.v_idx][:,:,hrhc.h,2] * cos(pos.θ)
    Δθ = hrhc.ΔXYθ[hrhc.v_idx][:,:,hrhc.h,3]

    hrhc.successor_states = zeros(size(hrhc.ΔXYθ[hrhc.v_idx][:,:,hrhc.h,:]))
    hrhc.successor_states[:,:,1] = ΔX + pos.x
    hrhc.successor_states[:,:,2] = ΔY + pos.y
    hrhc.successor_states[:,:,3] = Δθ + pos.θ

    return
end
function computeTrajectory(hrhc::accelHRHC, scene, v_cmd, δ_cmd; h=hrhc.h)
    pos = scene.vehicles[hrhc.car_ID].state.posG

    traj_ΔXYθ = hrhc.ΔXYθ[hrhc.v_idx][δ_cmd,v_cmd,1:h,:]

    ΔX = traj_ΔXYθ[:,1] * cos(pos.θ) + traj_ΔXYθ[:,2] * -sin(pos.θ)
    ΔY = traj_ΔXYθ[:,1] * sin(pos.θ) + traj_ΔXYθ[:,2] * cos(pos.θ)
    Δθ = traj_ΔXYθ[:,3]

    trajectory = zeros(size(traj_ΔXYθ,1),size(traj_ΔXYθ,2)+2)
    trajectory[:,1] = ΔX + pos.x
    trajectory[:,2] = ΔY + pos.y
    trajectory[:,3] = Δθ + pos.θ
    trajectory[:,4] = hrhc.v_cmds[v_cmd,δ_cmd]
    trajectory[:,5] = hrhc.δ_cmds[v_cmd,δ_cmd]

    return trajectory
end
function AutomotiveDrivingModels.observe!(hrhc::accelHRHC, scene::Scene, roadway::Roadway, egoid::Int, tree::KDTree, obstacleMap, k_level)
    """
    Observe the current environment and select optimal action to apply at next
    time step
    """
    if k_level > hrhc.k
        return
    end
    state = scene.vehicles[hrhc.car_ID].state
    hrhc.curve_ind = state.posF.roadind.ind.i
    v = state.v # current v
    hrhc.v = v

    trajectory = zeros(hrhc.h,5)
    action_selected = false
    abs_cmd = (hrhc.v_idx,hrhc.δ_idx)
    v_cmd = hrhc.v_idx
    δ_cmd = hrhc.δ_idx

    i = 0
    for i in 0:(hrhc.h-1)
        if action_selected
            break # out of for loop
        end

        # calculate successor states
        getSuccessorStates!(hrhc, scene)

        # project successor states onto track
        s,t,ϕ = loopProjectionKD(hrhc, scene, roadway, tree)

        # optimization objective
        objective = calculateObjective2(hrhc.car_ID, scene.vehicles[hrhc.car_ID].state.posF.s,
        s, t, ϕ, hrhc.T_MAX)

        while (action_selected==false) && (minimum(objective) != Inf)
            index = indmin(objective) # find get a better method of optimizing this
            δ_cmd, v_cmd = ind2sub(s, index)

            # compute full trajectory up to horizon
            trajectory = computeTrajectory(hrhc, scene, v_cmd,δ_cmd, h=hrhc.h-i)

            # screen trajectory for collisions / validity
            out_of_bounds = screenTrajectory(trajectory, obstacleMap, scene, roadway, hrhc, tree, k_level)

            if out_of_bounds
                objective[index] = Inf
            else
                action_selected=true
                updateObstacleMap!(obstacleMap, k_level, hrhc.car_ID, trajectory)
            end
        end
    end

    hrhc.δ_idx = δ_cmd
    hrhc.v_idx = min(max(1,hrhc.v_idx+v_cmd-2), size(hrhc.v,1))

    hrhc.δ = hrhc.δ_cmds[hrhc.v_idx, δ_cmd] # next δ
    hrhc.v = hrhc.v_cmds[hrhc.v_idx, δ_cmd] # next v

    next_state = VehicleState(VecSE2(trajectory[1,1:3]),roadway,hrhc.v)
    hrhc.action = NextState(next_state) # action
end
AutomotiveDrivingModels.get_name(::accelHRHC) = "accelHRHC"
AutomotiveDrivingModels.action_context(driver::accelHRHC) = driver.action_context # AutomotiveDrivingModels.action_context
Base.rand(hrhc::accelHRHC) = hrhc.action

type mapHRHC <: DriverModel{NextState, IntegratedContinuous}
    action_context::IntegratedContinuous
    car_ID::Int
    v_map
    δ_map
    motion_map # state changes associated with cmd = (v_command, δ_command) - this is the motion map
    v_cmd::Int # index of v in v_cmds
    δ_cmd::Int # index of δ in δ_cmds
    #car parameters with bicycle geometry model
    car_length::Float64 # wheel base
    car_width::Float64
    a_step::Float64 # max acceleration (m/s)
    μ::Float64 # friction coefficient
    v_range # possible velocityies
    δ_range # possible steering angles
    successor_states # array of next states
    # current v, current δ
    v::Float64
    δ::Float64
    curve_ind::Int
    Δs::Float64
    # planning horizon
    h::Int
    Δt::Float64
    # logit level
    k::Int
    # maximum deviation from center of track (if |t| > T_MAX, car is out of bounds)
    T_MAX::Float64
    # Action = Next State
    action::NextState
    function mapHRHC(car_ID::Int,roadway,context;
        car_length::Float64=4.8,
        wheel_base::Float64=4.4,
        car_width::Float64=2.5,
        v::Float64=0.0,
        δ::Float64=0.0,
        h::Int=10,
        Δt::Float64=1.0/24,
        ΔV₊::Float64=1.55,
        ΔV₋::Float64=3.05,
        Δδ::Float64=Float64(π)/12,
        v_min::Float64=0.0,
        v_max::Float64=100.0,
        a_step::Float64=10.0,
        a_range=[-1,0,1],
        μ::Float64=20.0,
        g::Float64=9.81,
        δ_max::Float64=Float64(π)/6,
        δ_step::Float64=Float64(π)/128,
        k::Int=1
        )

        hrhc = new()

        hrhc.T_MAX=(roadway.segments[1].lanes[1].width - car_width)/2.0

        hrhc.car_ID = car_ID
        hrhc.car_length=car_length
        hrhc.car_width=car_width
        hrhc.h=h
        hrhc.Δt=Δt
        hrhc.a_step=a_step
        hrhc.μ=μ

        hrhc.k=k

        hrhc.motion_map, hrhc.v_map, hrhc.δ_map, hrhc.v_range, hrhc.δ_range = GenerateMotionMap(v_min,
            v_max,a_step,δ_max,δ_step,wheel_base,h,μ,Δt,a_range=a_range)

        hrhc.successor_states = zeros(size(hrhc.motion_map[1],1),size(hrhc.motion_map[1],2),3)

        hrhc.v=v
        hrhc.δ=δ
        hrhc.v_cmd = 1
        hrhc.δ_cmd = Int((length(hrhc.δ_range) - 1)/2)
        hrhc.curve_ind=1
        hrhc.Δs=roadway.segments[1].lanes[1].curve[2].s-roadway.segments[1].lanes[1].curve[1].s
        hrhc.action_context=context
        hrhc.action = NextState(VehicleState(VecSE2(0,0,0),0.0))

        hrhc
    end
end
function GenerateMotionMap(v_min,v_max,a_step,δ_max,δ_step,wheel_base,h,μ,Δt;g=9.81,a_range=[-1,0,1])

    v_step = a_step*Δt
    v_range = linspace(v_min,v_max,round(v_max/v_step))
    δ_range = linspace(-δ_max,δ_max,2*round((δ_max)/δ_step)+1)
    δ_range = (1/maximum(δ_range))*δ_range.*abs(δ_range) # concentrate toward the middle
    R_range = wheel_base./tan(δ_range)
    # a_range = [-1,0,1]
    motion_map = Dict()
    v_map = Dict()
    δ_map = Dict()

    for v_idx in 1:length(v_range)
        v = v_range[v_idx]
        v_map[v_idx] = zeros(Int,length(a_range),length(δ_range))
        δ_map[v_idx] = zeros(Int,length(a_range),length(δ_range))
        for a_idx in 1:length(a_range)
            Δv_idx = a_range[a_idx]
            tire_force = sqrt(((v^2)./abs(R_range)).^2 + (Δv_idx*a_step)^2)
            if v_idx + Δv_idx >= 1 && v_idx + Δv_idx <= length(v_range)
                v_map[v_idx][a_idx,:] = Int(v_idx + Δv_idx)# v_range[v_idx+Δv_idx]
            else
                v_map[v_idx][a_idx,:] = Int(v_idx) #v_range[v_idx]
            end
            if v_idx > 1
                for j in 1:length(δ_range)
                    if tire_force[j] >= μ*g
                        v_map[v_idx][a_idx,j] = Int(v_idx) - 1# v_range[v_idx-1] # decelerate
                    end
                end
            end
            #δ_map[v_idx][a_idx,:] = δ_range.*(tire_force .< μ*g) +
             #   maximum(abs(δ_range[tire_force .< μ*g])).*sign(δ_range).*(tire_force .>= μ*g)
            δ_map[v_idx][a_idx,:] = [i for i in 1:length(δ_range)].*(tire_force .< μ*g) +
            indmax(abs(δ_range).*(tire_force .< μ*g)).*(tire_force .>= μ*g).*-sign(δ_range) +
            (length(δ_range)+1).*(sign(δ_range) .> 0).*(tire_force .>= μ*g)
        end
    end


    # fill motion map
    for v_idx in 1:length(v_range)
        motion_map[v_idx] = zeros(length(a_range),length(δ_range),h,3)
        level_v_idx = copy(v_map[v_idx])
        level_δ_idx = copy(δ_map[v_idx])
        Δθ = zeros(length(a_range),length(δ_range))
        for i in 1:h
            radius = wheel_base./tan(δ_range[level_δ_idx])
            Δs = ((v_range[v_idx] + v_range[level_v_idx])./2.0)*Δt
            Δθ = Δs./radius
            ΔX = abs(radius) .* sin(abs(Δθ))
            ΔX[:,Int((length(δ_range)-1)/2)+1] = Δs[:,Int((length(δ_range)-1)/2)+1]
            ΔY = radius.*(1 - cos(Δθ))
            ΔY[:,Int((length(δ_range)-1)/2)+1] = 0
            if i == 1
                motion_map[v_idx][:,:,i,1] = ΔX
                motion_map[v_idx][:,:,i,2] = ΔY
                motion_map[v_idx][:,:,i,3] = Δθ
            else
                motion_map[v_idx][:,:,i,1] = motion_map[v_idx][:,:,i-1,1] +
                    ΔX.*cos(motion_map[v_idx][:,:,i-1,3]) - ΔY.*sin(motion_map[v_idx][:,:,i-1,3])
                motion_map[v_idx][:,:,i,2] = motion_map[v_idx][:,:,i-1,2] +
                    ΔX.*sin(motion_map[v_idx][:,:,i-1,3]) + ΔY.*cos(motion_map[v_idx][:,:,i-1,3])
                motion_map[v_idx][:,:,i,3] = Δθ + motion_map[v_idx][:,:,i-1,3]  # + motion_map[v_idx][:,:,i-1,3]
            end
            for j in 1:length(level_v_idx) # update to next velocity
                level_v_idx[j] = v_map[level_v_idx[j]][j]
                level_δ_idx[j] = δ_map[level_v_idx[j]][j]
            end
        end
    end
    return motion_map, v_map, δ_map, v_range, δ_range
end
function getSuccessorStates(ΔXYθ, car_ID, h, scene::Scene)
    """ gets legal successor_states from motion primitives library """
    pos = scene.vehicles[car_ID].state.posG # global x,y,z of car

    ΔX = ΔXYθ[:,:,h,1] * cos(pos.θ) + ΔXYθ[:,:,h,2] * -sin(pos.θ)
    ΔY = ΔXYθ[:,:,h,1] * sin(pos.θ) + ΔXYθ[:,:,h,2] * cos(pos.θ)
    Δθ = ΔXYθ[:,:,h,3]

    successor_states = zeros(size(ΔXYθ[:,:,h,:]))
    successor_states[:,:,1] = ΔX + pos.x
    successor_states[:,:,2] = ΔY + pos.y
    successor_states[:,:,3] = Δθ + pos.θ

    return successor_states
end
function computeTrajectory(ΔXYθ, car_ID, scene, v_cmd, δ_cmd, h)
    pos = scene.vehicles[car_ID].state.posG

    trajectory = zeros(h,3)
    trajectory[:,1] = pos.x + ΔXYθ[v_cmd,δ_cmd,1:h,1]*cos(pos.θ) + ΔXYθ[v_cmd,δ_cmd,1:h,2]*-sin(pos.θ)
    trajectory[:,2] = pos.y + ΔXYθ[v_cmd,δ_cmd,1:h,1]*sin(pos.θ) + ΔXYθ[v_cmd,δ_cmd,1:h,2]*cos(pos.θ)
    trajectory[:,3] = pos.θ + ΔXYθ[v_cmd,δ_cmd,1:h,3]

    return trajectory
end
function screenCollision(hrhc, obstacleMap, tree, roadway, scene, k_level)
    L = hrhc.car_length + 2.5
    W = hrhc.car_width

    collisionFlag = zeros(size(hrhc.successor_states[:,:,1])) # stores locations collisions
    threshold_dist = hrhc.car_length*4 # must be at least this close before we care to calculate collision cost

    if k_level >= 1
        # for all other cars...
        for (id,trajectory) in obstacleMap[k_level - 1]
            if id != hrhc.car_ID
                state = scene.vehicles[hrhc.car_ID].state
                state2 = scene.vehicles[id].state
                diff = state.posG - state2.posG
                s1,_,_ = kdProject(state.posG.x,state.posG.y,state.posG.θ,tree,roadway,hrhc)
                s2,_,_ = kdProject(state2.posG.x,state2.posG.y,state2.posG.θ,tree,roadway,hrhc)
                if (norm([diff.x, diff.y]) < threshold_dist) && (s1 <= s2) # don't care if opponent is behind us
                    R_idx = zeros(Int, size(hrhc.successor_states,1)*size(hrhc.successor_states,2)) # to store the sorted indices of R
                    for i in 1:hrhc.h
                        pos = VecSE2(trajectory[i,1:3]) # x,y,θ of opponent at time step h
                        successor_states = getSuccessorStates(hrhc.motion_map[hrhc.v_cmd],hrhc.car_ID,i,scene)
                        ΔX = successor_states[:,:,1] - pos.x # Δx, with opponent at origin
                        ΔY = successor_states[:,:,2] - pos.y # Δy with opponent at origin
                        Δθ = successor_states[:,:,3] - pos.θ # Δθ with opponent at origin
                        R = sqrt(ΔX.^2 + ΔY.^2)
                        R_idx = sortperm(reshape(R,size(R,1)*size(R,2),1)[:])
                        for idx in R_idx
                            if collisionFlag[idx] == 1
                                continue
                            end
                            if R[idx] > hrhc.car_length # no collision, and all other R values are greater
                                break
                            end
                            # R is less than hrhc.car_length
                            ψ = atan2(ΔY[idx],ΔX[idx]) - Δθ[idx]
                            r = W*L/sqrt((L*sin(ψ))^2 + (W*cos(ψ))^2) # radius of ellipse at given angle
                            if R[idx]-r < W*L/8 # collision
                                collisionFlag[idx] = 1
                            end
                        end
                    end
                end
            end
        end
    end
    return collisionFlag
end
function TailgateAvoidance(hrhc, obstacleMap, tree, roadway, scene, k_level)
    L = 2*hrhc.car_length
    W = hrhc.car_width

    collisionFlag = zeros(size(hrhc.successor_states[:,:,1])) # stores locations collisions
    threshold_dist = hrhc.car_length*4 # must be at least this close before we care to calculate collision cost

    R = zeros(size(hrhc.successor_states[:,:,1]))
    ψ = zeros(size(hrhc.successor_states[:,:,1]))
    cost = zeros(size(hrhc.successor_states[:,:,1]))

    if k_level >= 1
        # for all other cars...
        for (id,trajectory) in obstacleMap[k_level - 1]
            if id != hrhc.car_ID && id == 3
                state = scene.vehicles[hrhc.car_ID].state
                state2 = scene.vehicles[id].state
                diff = state.posG - state2.posG
                s1,_,_ = kdProject(state.posG.x,state.posG.y,state.posG.θ,tree,roadway,hrhc)
                s2,_,_ = kdProject(state2.posG.x,state2.posG.y,state2.posG.θ,tree,roadway,hrhc)
                if (norm([diff.x, diff.y]) < threshold_dist) && (s1 <= s2) # don't care if opponent is behind us
                    pos = VecSE2(trajectory[hrhc.h,1:3]) # x,y,θ of opponent at time step h
                    successor_states = getSuccessorStates(hrhc.motion_map[hrhc.v_cmd],hrhc.car_ID,hrhc.h,scene)
                    ΔX = successor_states[:,:,1] - pos.x # Δx, with opponent at origin
                    ΔY = successor_states[:,:,2] - pos.y # Δy with opponent at origin
                    Δθ = successor_states[:,:,3] - pos.θ # Δθ with opponent at origin
                    R = sqrt(ΔX.^2 + ΔY.^2)
                    ψ = atan2(ΔY,ΔX) - Δθ

                    cost = (1./(W*R.*cos(ψ).^2 + 2*L*R.*sin(ψ).^2 + 1)).*(-cos(ψ).^3 + 1.1) + 1

                end
            end
        end
    end
    return cost
end
function calculateObjective(car_ID,s₀,s,t,ϕ,T_MAX;ϕ_MAX=Float64(π),s_factor=1.0)
    s_norm = (s-s₀)/maximum(s)
    t_norm = t/T_MAX
    ϕ_norm = ϕ/ϕ_MAX

    #costs
    t_cost = abs(t_norm).^6
    ϕ_cost = abs(ϕ_norm).^6
    s_factor = 1
    s_cost = s_factor*(1-s_norm)
    A = [1 .5; #  [ϕ t] [a1 a2] [ϕ]
        .5 0] #         [a2 a3] [t]
    tϕ_cost = A[1,1]*(ϕ_norm).^2 + (A[1,2]+A[2,1])*(ϕ_norm).*(t_norm) + A[2,2]*(t_norm).^2

    objective = 1+s_cost+t_cost+ϕ_cost+tϕ_cost
    return objective
end
# function calculateObjective(hrhc, scene, roadway, tree, s, t, ϕ, obstacleMap, k_level, h; f_ϕ=0.0, f_t=0.1, f_tϕ=3.0)
#     """
#     Calculates the value of the optimization objective function for every state
#       in hrhc.successor_states
#     """
#     state = scene.vehicles[hrhc.car_ID].state
#     dS = s - state.posF.s
#     dS = dS / maximum(dS) # normalize
#     ϕMAX = Float64(π)/2
#
#     # penalize large t (proximity to edge of track)
#     cost_t = (exp(((10-h+f_t)*abs(t/hrhc.T_MAX).^2)) - 1)/exp(f_t) + Inf*(t.>hrhc.T_MAX)
#     # penalize large ϕ (steering away from forward direction on the track)
#     cost_ϕ = (exp(((10-h+f_ϕ)*abs(ϕ/ϕMAX).^2)) - 1)/exp(f_ϕ)
#     # penalize when t and ϕ have the same sign
#     A = [1 1; 1 1]
#     cost_x = (((ϕ/ϕMAX)*A[1,1] + (t/hrhc.T_MAX)*A[2,1]).*(ϕ/ϕMAX) + ((ϕ/ϕMAX)*A[1,2] + (t/hrhc.T_MAX)*A[2,2]).*(t/hrhc.T_MAX))/2
#     cost_tϕ = (exp(f_tϕ*cost_x) - 1)/exp(1)
#     eligibility_mask = ((hrhc.successor_states[:,:,1] .== state.posG.x).*(hrhc.successor_states[:,:,2] .== state.posG.y))
#
#     # obstacles
#     collisionCost = zeros(size(cost_t))
#     threshold_dist = hrhc.car_length*4 # must be at least this close before we care to calculate collision cost
#     if k_level >= 1
#         for (id,car) in obstacleMap[k_level - 1]
#             if id != hrhc.car_ID
#                 state = scene.vehicles[hrhc.car_ID].state
#                 state2 = scene.vehicles[id].state
#                 diff = state.posG - state2.posG
#                 s1,_,_ = kdProject(state.posG.x,state.posG.y,state.posG.θ,tree,roadway,hrhc)
#                 s2,_,_ = kdProject(state2.posG.x,state2.posG.y,state2.posG.θ,tree,roadway,hrhc)
#                 if (norm([diff.x, diff.y]) < threshold_dist) && (s1 <= s2) # don't care if opponent is behind us
#                     pos = VecSE2(car[h,1:3]) # x,y,θ of opponent at time step h
#                     ΔX = hrhc.successor_states[:,:,1] - pos.x # Δx, with opponent at origin
#                     ΔY = hrhc.successor_states[:,:,2] - pos.y # Δy with opponent at origin
#                     Δθ = hrhc.successor_states[:,:,3] - pos.θ # Δθ with opponent at origin
#                     pts = [hrhc.car_length hrhc.car_length -hrhc.car_length -hrhc.car_length 0;
#                         -hrhc.car_width hrhc.car_width hrhc.car_width -hrhc.car_width 0]/1.8
#                     pX = zeros(size(pts,2),size(hrhc.successor_states,1),size(hrhc.successor_states,2))
#                     pY = zeros(size(pX))
#                     for i in 1:size(pts,2)
#                         pX[i,:,:] = pts[1,i]*cos(Δθ) - pts[2,i]*sin(Δθ) + ΔX
#                         pY[i,:,:] = pts[1,i]*sin(Δθ) + pts[2,i].*cos(Δθ) + ΔY
#                     end
#
#                     collisionFlag = (maximum((abs(pX) .< hrhc.car_length/1.0),1)[1,:,:]).*(maximum((abs(pY) .< hrhc.car_width/1.9),1)[1,:,:])
#                     collisionCost = .001+(collisionFlag .>= 1)./(minimum(abs(pX),1)[1,:,:].*minimum(abs(pY),1)[1,:,:])
#                     # collisionCost = Inf.*collisionFlag
#                 end
#             end
#         end
#     end
#
#     objective = cost_t + cost_ϕ + cost_tϕ + 1 - dS + collisionCost + Inf * eligibility_mask
#     return objective
# end
function AutomotiveDrivingModels.observe!(hrhc::mapHRHC, scene::Scene, roadway::Roadway, egoid::Int, tree::KDTree, obstacleMap, k_level)
    """
    Observe the current environment and select optimal action to apply at next
    time step
    """

    state = scene.vehicles[hrhc.car_ID].state
    hrhc.curve_ind = state.posF.roadind.ind.i
    v = state.v # current v
    hrhc.v = v
    trajectory = zeros(hrhc.h,3)
    action_selected = false
    δ_cmd = 1
    a_cmd = 1

    i = 0
    for i in 0:(hrhc.h-1)
        if action_selected
            break # out of for loop
        end
        # calculate successor states
        hrhc.successor_states = getSuccessorStates(hrhc.motion_map[hrhc.v_cmd], hrhc.car_ID, hrhc.h-i, scene)
        # project successor states onto track
        s,t,ϕ = loopProjectionKD(hrhc, scene, roadway, tree)
        # optimization objective
        objective = calculateObjective(hrhc.car_ID, scene.vehicles[hrhc.car_ID].state.posF.s,
        s, t, ϕ, hrhc.T_MAX)
        tailgateCost = TailgateAvoidance(hrhc, obstacleMap, tree, roadway, scene, k_level)
        objective = objective + tailgateCost
        collisionFlag = screenCollision(hrhc, obstacleMap, tree, roadway, scene, k_level)
        objective[collisionFlag .> 0] = Inf

        while (action_selected==false) && (minimum(objective) != Inf)
            index = indmin(objective) # find get a better method of optimizing this
            a_cmd, δ_cmd = ind2sub(s, index)
            hrhc.δ_cmd = δ_cmd
            # compute full trajectory up to horizon
            trajectory = computeTrajectory(hrhc.motion_map[hrhc.v_cmd], hrhc.car_ID, scene, a_cmd, δ_cmd, hrhc.h-i)
            # screen trajectory for collisions / validity
            out_of_bounds = screenTrajectory(trajectory, obstacleMap, scene, roadway, hrhc, tree, k_level)
            if out_of_bounds
                objective[index] = Inf
            else
                action_selected=true
                updateObstacleMap!(obstacleMap, k_level, hrhc.car_ID, trajectory)
                if k_level != hrhc.k # only assign commands for the given k_level
                    return # do not update the action
                end
            end
        end
    end

    hrhc.δ_cmd = δ_cmd
    # hrhc.v_cmd = max(hrhc.v_cmd + a_cmd - 2,1) # assumes 3 options for acceleration
    hrhc.v_cmd = hrhc.v_map[hrhc.v_cmd][a_cmd,δ_cmd]

    hrhc.δ = hrhc.δ_range[hrhc.δ_cmd] # next δ
    hrhc.v = hrhc.v_range[hrhc.v_cmd] # next v

    next_state = VehicleState(VecSE2(trajectory[1,1:3]),roadway,hrhc.v)
    hrhc.action = NextState(next_state) # action
end
AutomotiveDrivingModels.get_name(::mapHRHC) = "mapHRHC"
AutomotiveDrivingModels.action_context(driver::mapHRHC) = driver.action_context # AutomotiveDrivingModels.action_context
Base.rand(hrhc::mapHRHC) = hrhc.action

function plotSplineRoadway(x,y,θ,lane_width)
    perp_lines1 = zeros(2,length(x))
    perp_lines2 = zeros(2,length(x))

    perp_lines1[1,:] = x + (lane_width/2.0)*sin(θ)
    perp_lines1[2,:] = y - (lane_width/2.0)*cos(θ)
    perp_lines2[1,:] = x - (lane_width/2.0)*sin(θ)
    perp_lines2[2,:] = y + (lane_width/2.0)*cos(θ)

    # PyPlot.figure()
    # PyPlot.scatter(x,y)
    PyPlot.plot(x,y)
    PyPlot.plot(perp_lines1[1,:],perp_lines1[2,:],color="green")
    PyPlot.plot(perp_lines2[1,:],perp_lines2[2,:],color="green")
    PyPlot.axis("equal")
    # PyPlot.show()
end
function plotObjectiveHorizon(hrhc,scene,roadway,tree,trajectory,obstacleMap,xR,yR,θR)
    lo=hrhc.curve_ind
    if :V_MAX in fieldnames(hrhc)
        hi = hrhc.curve_ind + Int(1+div(hrhc.V_MAX*hrhc.Δt*hrhc.h,hrhc.Δs))
    else
        hi = hrhc.curve_ind + Int(1+div(hrhc.v_range[end]*hrhc.Δt*hrhc.h,hrhc.Δs))
    end
    lane_width = roadway.segments[1].lanes[1].width

    x = zeros(hrhc.h,size(hrhc.successor_states,1),size(hrhc.successor_states,2))
    y = zeros(size(x))
    Θ = zeros(size(x))
    s = zeros(size(x))
    t = zeros(size(x))
    ϕ = zeros(size(x))
    objective = zeros(size(x))

    for i in 1:hrhc.h
        getLegalMoves!(hrhc, scene, h=i)
        getSuccessorStates!(hrhc, scene)
        x[i,:,:] = copy(hrhc.successor_states[:,:,1])
        y[i,:,:] = copy(hrhc.successor_states[:,:,2])
        Θ[i,:,:] = copy(hrhc.successor_states[:,:,3])
        s[i,:,:], t[i,:,:], ϕ[i,:,:] = loopProjectionKD(hrhc,scene,roadway,tree)
        objective[i,:,:] = calculateObjective(hrhc,scene, roadway, tree,s[i,:,:],t[i,:,:],ϕ[i,:,:],obstacleMap,hrhc.k,hrhc.h)
    end

    PyPlot.figure(figsize=[12,4])

    PyPlot.subplot(141) # ϕ
    plotSplineRoadway(xR[lo:hi],yR[lo:hi],θR[lo:hi],lane_width)
    PyPlot.scatter(x,y,c=ϕ,edgecolor="none")
    PyPlot.plot(trajectory[:,1],trajectory[:,2],color="red")
    PyPlot.axis("off")
    PyPlot.title("|phi|")

    PyPlot.subplot(142) # s
    plotSplineRoadway(xR[lo:hi],yR[lo:hi],θR[lo:hi],lane_width)
    PyPlot.scatter(x,y,c=s,edgecolor="none")
    PyPlot.plot(trajectory[:,1],trajectory[:,2],color="red")
    PyPlot.axis("off")
    PyPlot.title("s")

    PyPlot.subplot(143) # t
    plotSplineRoadway(xR[lo:hi],yR[lo:hi],θR[lo:hi],lane_width)
    PyPlot.scatter(x,y,c=t,edgecolor="none")
    PyPlot.plot(trajectory[:,1],trajectory[:,2],color="red")
    PyPlot.axis("off")
    PyPlot.title("t")

    PyPlot.subplot(144) # objective
    plotSplineRoadway(xR[lo:hi],yR[lo:hi],θR[lo:hi],lane_width)
    PyPlot.scatter(x,y,c=log(objective),edgecolor="none")
    PyPlot.plot(trajectory[:,1],trajectory[:,2],color="red")
    PyPlot.axis("off")
    PyPlot.title("log objective")
end
function plot_stϕ(hrhc,roadway,scene,x,y,θ,trajectory,s,t,ϕ,objective)
    lo=hrhc.curve_ind
    hi=hrhc.curve_ind + Int(1+2*div(hrhc.V_MAX*hrhc.Δt*hrhc.h,hrhc.Δs))
    if hi > length(roadway.segments[1].lanes[1].curve)
        lo = length(roadway.segments[1].lanes[1].curve)
        hi=hrhc.curve_ind + Int(1+2*div(hrhc.V_MAX*hrhc.Δt*hrhc.h,hrhc.Δs))
    end
    lane_width = roadway.segments[1].lanes[1].width

    PyPlot.figure(figsize=[12,4])
    PyPlot.subplot(141)
    plotSplineRoadway(x[lo:hi],y[lo:hi],θ[lo:hi],lane_width)
    # PyPlot.scatter(Pts[1,:],Pts[2,:],color="red")
    PyPlot.scatter(hrhc.successor_states[:,:,1],hrhc.successor_states[:,:,2],c=abs(ϕ),edgecolor="none")
    PyPlot.plot(trajectory[:,1],trajectory[:,2],color="red")
    PyPlot.scatter(scene.vehicles[hrhc.car_ID].state.posG.x, scene.vehicles[hrhc.car_ID].state.posG.y, c="k", edgecolors="none",s=40)
    PyPlot.axis("off")
    PyPlot.title("|phi|")

    PyPlot.subplot(142)
    plotSplineRoadway(x[lo:hi],y[lo:hi],θ[lo:hi],lane_width)
    # PyPlot.scatter(Pts[1,:],Pts[2,:],color="red")
    PyPlot.scatter(hrhc.successor_states[:,:,1],hrhc.successor_states[:,:,2],c=abs(t),edgecolor="none")
    PyPlot.plot(trajectory[:,1],trajectory[:,2],color="red")
    PyPlot.scatter(scene.vehicles[hrhc.car_ID].state.posG.x, scene.vehicles[hrhc.car_ID].state.posG.y, c="k", edgecolors="none",s=40)
    PyPlot.axis("off")
    PyPlot.title("|t|")

    PyPlot.subplot(143)
    plotSplineRoadway(x[lo:hi],y[lo:hi],θ[lo:hi],lane_width)
    # PyPlot.scatter(Pts[1,:],Pts[2,:],color="red")
    PyPlot.scatter(hrhc.successor_states[:,:,1],hrhc.successor_states[:,:,2],c=s,edgecolor="none")
    PyPlot.scatter(scene.vehicles[hrhc.car_ID].state.posG.x, scene.vehicles[hrhc.car_ID].state.posG.y, c="k", edgecolors="none",s=40)
    PyPlot.plot(trajectory[:,1],trajectory[:,2],color="red")
    PyPlot.axis("off")
    PyPlot.title("s")


    PyPlot.subplot(144)
    plotSplineRoadway(x[lo:hi],y[lo:hi],θ[lo:hi],lane_width)
    # PyPlot.scatter(Pts[1,:],Pts[2,:],color="red")
    PyPlot.scatter(hrhc.successor_states[:,:,1],hrhc.successor_states[:,:,2],c=log(objective),edgecolor="none")
    PyPlot.plot(trajectory[:,1],trajectory[:,2],color="red")
    PyPlot.scatter(scene.vehicles[hrhc.car_ID].state.posG.x, scene.vehicles[hrhc.car_ID].state.posG.y, c="k", edgecolors="none",s=40)
    PyPlot.axis("off")
    PyPlot.title("objective")
end
function plotHRHCInfo(hrhc,models,scene,roadway,trajectory,cmd,x,y,Θ,s,t,ϕ,objective)
    lo = hrhc.curve_ind
    hi = hrhc.curve_ind + Int(1+2*div(hrhc.V_MAX*hrhc.Δt*hrhc.h,hrhc.Δs))
    lane_width = roadway.segments[1].lanes[1].width
    if hi > length(roadway.segments[1].lanes[1].curve)
        lo = length(roadway.segments[1].lanes[1].curve)
        hi=hrhc.curve_ind + Int(1+2*div(hrhc.V_MAX*hrhc.Δt*hrhc.h,hrhc.Δs))
    end
    PyPlot.figure(figsize=[12,10])
    # Plot Raceway
    PyPlot.subplot(221)
    # plotSplineRoadway(x[lo:hi],y[lo:hi],θ[lo:hi],lane_width)
    plotSplineRoadway(x,y,Θ,lane_width)
    PyPlot.scatter(hrhc.successor_states[:,:,1],hrhc.successor_states[:,:,2],color="red")
    PyPlot.plot(trajectory[:,1],trajectory[:,2],color="red")
    PyPlot.scatter(roadway.segments[1].lanes[1].curve[hrhc.curve_ind].pos.x, roadway.segments[1].lanes[1].curve[hrhc.curve_ind].pos.y, c="white", s=40)
    for (id,car) in models
        if id == hrhc.car_ID
            PyPlot.scatter(scene.vehicles[id].state.posG.x,scene.vehicles[id].state.posG.y,c="red",s=20)
        else
            PyPlot.scatter(scene.vehicles[id].state.posG.x,scene.vehicles[id].state.posG.y,c="blue",s=20)
        end
    end
    PyPlot.axis("off")
    PyPlot.title("Raceway with Motion Primitives")

    PyPlot.subplot(222)
    plotSplineRoadway(x[lo:hi],y[lo:hi],Θ[lo:hi],lane_width)
    PyPlot.scatter(scene.vehicles[hrhc.car_ID].state.posG.x, scene.vehicles[hrhc.car_ID].state.posG.y, c="red", edgecolors="none",s=40)
    PyPlot.scatter(hrhc.successor_states[:,:,1], hrhc.successor_states[:,:,2],c=log(objective),edgecolors="none")
    PyPlot.scatter(hrhc.successor_states[cmd[1],cmd[2],1], hrhc.successor_states[cmd[1],cmd[2],2],c="white",s=40)
    PyPlot.plot(trajectory[:,1],trajectory[:,2],color="red")
    PyPlot.axis("off")
    PyPlot.title("Log Objective Function")
end

end # module
