using Agents

@agent Zombie OSMAgent begin
	infected::Bool
end

function initialize(; map_path=OSM.TEST_MAP)
	model = ABM(Zombie, OpenStreetMapSpace(map_path))

	for id in 1:100
		start = random_position(model) # at an intersection
		finish = OSM.random_road_position(model) # somewhere on a road
		route = OSM.plan_route(start, finish, model)
		human = Zombie(id, start, route, finish, false)
		add_agent_pos!(human, model)
	end

	# add a patient zero at a specific location
	# start = OSM.road((79.52320181536525, -119.78917553184259), model)
    # finish = OSM.intersection((39.510773, -119.75916700000002), model)
	start = random_position(model) # at an intersection
	finish = OSM.random_road_position(model) # somewhere on a road
	route = OSM.plan_route(start, finish, model)
	zombie = add_agent!(start, model, route, finish, true)

	return model
end

function agent_step!(agent, model)
	# each agent will move 25 meters
	move_along_route!(agent, model, 25)

	if is_stationary(agent, model) && rand(model.rng) < 0.1
		# when stationary maybe go somewhere else
		OSM.random_route!(agent, model)
		move_along_route!(agent, model, 25)
	end

	if agent.infected
		# infect if within 50 meters of a zombie
		map(i -> model[i].infected=true, nearby_ids(agent, model, 50))
	end
end

using OpenStreetMapXPlot, Plots
gr()

ac(agent) = agent.infected ? :green : :black
as(agent) = agent.infected ? 6 : 5

function plotagents(model)
	ids = model.scheduler(model)
	colors = [ac(model[i]) for i in ids]
	sizes = [as(model[i]) for i in ids]
	markers = :circle
	pos = [OSM.map_coordinates(model[i], model) for i in ids]

	scatter!(
		pos;
		markercolor = colors,
		markersize = sizes,
		markershapes = markers,
		label = "",
		markerstrokewidth = 0.5,
		markerstrokecolor = :black,
		markeralpha = 0.7,
	)
end

model = initialize(; map_path="limamap.osm")

@time step!(model, agent_step!, 10)
@time plotmap(model.space.m)
@time plotagents(model)

frames = @animate for i in 0:20
	print(i)
	i > 0 && step!(model, agent_step!, 1)
	plotmap(model.space.m)
	plotagents(model)
end

gif(frames, "zombie_outbreak.gif", fps=15)
