function plotHRHCInfo2(hrhc,scene,roadway,s,t,ϕ,objective)
    lo=hrhc.curve_ind
    hi = hrhc.curve_ind + Int(1+2*div(hrhc.V_MAX*hrhc.Δt*hrhc.h,hrhc.Δs))
    lo, hi
    PyPlot.figure(figsize=[12,10])
    # Plot Raceway
    PyPlot.subplot(221)
    # plotSplineRoadway(xP[lo:hi],yP[lo:hi],θP[lo:hi],lane_width)
    plotSplineRoadway(xP,yP,θP,lane_width)
    PyPlot.scatter(hrhc.successor_states[:,:,1],hrhc.successor_states[:,:,2],color="red")
    PyPlot.plot(trajectory[:,1],trajectory[:,2],color="red")
    PyPlot.scatter(roadway.segments[1].lanes[1].curve[hrhc.curve_ind].pos.x, roadway.segments[1].lanes[1].curve[hrhc.curve_ind].pos.y, c="white", s=40)
    PyPlot.axis("off")
    PyPlot.title("Raceway with Motion Primitives")

    PyPlot.subplot(222)
    plotSplineRoadway(xP[lo:hi],yP[lo:hi],θP[lo:hi],lane_width)
    # PyPlot.scatter(Pts[1,:],Pts[2,:],color="red")
    # PyPlot.scatter(hrhc.successor_states[:,:,1],hrhc.successor_states[:,:,2],color="red")
    PyPlot.scatter(hrhc.successor_states[:,:,1], hrhc.successor_states[:,:,2],c=log(objective),edgecolors="none")
    PyPlot.scatter(hrhc.successor_states[cmd[1],cmd[2],1], hrhc.successor_states[cmd[1],cmd[2],2],c="white",s=40)
    PyPlot.plot(trajectory[:,1],trajectory[:,2],color="red")
    PyPlot.axis("off")
    PyPlot.title("Log Objective Function")
end

