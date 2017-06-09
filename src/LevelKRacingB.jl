module LevelKRacing

using AutomotiveDrivingModels
using AutoViz
using NearestNeighbors

import AutomotiveDrivingModels: get_actions!, observe!, action_context, get_name
import Base.rand
import PyPlots

export
    # HierarchicalRecedingHorizonController
    HRHC,
    curveDist,
    wrap_to_π,
    kdProject,
    generateObstacleMap,
    updateObstacleMap!,
    generateMotionMap,
    screenCollision,
    tailgateAvoidance,
    getSuccessorStates,
    loopProjectionKD,
    computeTrajectory,
    screenTrajectory,
    checkCollision,
    calculateObjective,
    plot_stϕ,
    plotHRHCInfo,
    plotObjectiveHorizon,
    plotSplineRoadway,

    # LevelKVisualizations
    DemoState,
    DemoMotionPrimitives,
    DemoIncreasedHorizonBehavior,
    DemoBuildingBlocks,
    DemoObjective,
    DemoTailgateAvoidance,
    DemoObserveActObjective,
    DemoObstacleMap,

    # SplineRaceWay
    Raceway,

    # SplineUtils
    ClosedB_Spline,
    B_SplineDerivative,
    ResampleSplineEven,
    GenSplineRoadway,
    PlotSplineRoadway

include("HierarchicalRecedingHorizonController.jl")
include("LevelKVisualizations.jl")
include("SplineRaceWay.jl")
include("SplineUtils.jl")

end
