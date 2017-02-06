using AutomotiveDrivingModels
using AutoViz

car_len = 4.8
car_width = 1.8

roadway = gen_stadium_roadway(2)
scene = Scene()
push!(scene,Vehicle(VehicleState(VecSE2(10.0,-DEFAULT_LANE_WIDTH,0.0), roadway, 29.0+randn()), 
                    VehicleDef(1, AgentClass.CAR, car_len, car_width)))
push!(scene,Vehicle(VehicleState(VecSE2(40.0,0.0,0.0), roadway, 29.0+randn()), 
                    VehicleDef(2, AgentClass.CAR, car_len, car_width)))
push!(scene,Vehicle(VehicleState(VecSE2(70.0,-DEFAULT_LANE_WIDTH,0.0), roadway, 29.0+randn()), 
                    VehicleDef(3, AgentClass.CAR, car_len, car_width)))

render(scene, roadway, cam=FitToContentCamera())

render(gen_straight_roadway(3, 100.0), canvas_height=120)