using Agents, InteractiveDynamics, GLMakie

mutable struct Fighter <: AbstractAgent
	id::Int
	pos::Dims{3}
	has_prisoner::Bool
	capture_time::Int
	shape::Symbol # for plotting
end

function battle(; fighters=50)
	model = ABM(
		Fighter,
		GridSpace((100,100,10); periodic=false);
		scheduler=Schedulers.randomly,
	)
	n = 0
	while n != fighters
		pos = (rand(model.rng, 1:100, 2)..., 1) # start at lvl 1
		if isempty(pos, model)
			add_agent!(pos, model, false, 0, :diamond)
			n += 1
		end
	end

	return model
end

model = battle()

space(agent) = agent.pos[1:2]
level(agent) = agent.pos[3]

function closest_target(agent::Fighter, ids::Vector{Int}, model::ABM)
	if length(ids) == 1
		closest = ids[1]
	else
		close_id = argmin(map(id -> edistance(space(agent), space(model[id]), model), ids))
		closest = ids[close_id]
	end
	return model[closest]
end

function battle!(one::Fighter, two::Fighter, model)
	if level(one) == level(two)
		one_winner = rand(model.rng) < 0.5
	elseif level(one) > level(two)
		one_winner = 2*rand(model.rng) > rand(model.rng)
	else
		one_winner = rand(model.rng) >  2*rand(model.rng)
	end

	one_winner ? (up = one; down = two) : (up = two; down = one)
	new_lvl_up = min(level(up) + 1, 10)
	new_pos_up = clamp.(rand(model.rng, -1:1, 2) .+ space(up), [1,1], size(model.space)[1:2])
	move_agent!(up, (new_lvl_up..., new_lvl_up), model)

	new_lvl_down = level(down) - 1
	if new_lvl_down == 0
		kill_agent!(down, model)
	else
		move_agent!(down, (space(down)..., new_lvl_down), model)
	end
end

function captor_behavior!(agent, model)
	close_ids = collect(nearby_ids(agent, model, (0,0,10)))
	if length(close_ids) == 1
		# Taunt prisoner or kill it
		prisoner = model[close_ids[1]]
		if prisoner.capture_time > 10
			agent.shape = :rect
			gain = ceil(Int, level(prisoner) / 2)
			new_lvl = min(level(agent) + gain, 10)
			kill_agent!(prisoner, model)
			agent.has_prisoner = false
			move_agent!(agent, (space(agent)..., new_lvl), model)
		end
	else
		# Someone is here to kill the captor. Could be more than 1 opponent
		prisoner = [model[id] for id in close_ids if model[id].capture_time > 0][1]
		exploiter = rand(
			model.rng,
			[
				model[id] for id in close_ids if
				model[id].capture_time == 0 && model[id].has_prisoner == false
			],
		)
		exploiter.shape = :rect
		gain = ceil(Int, level(agent) / 2)
		new_lvl = min(level(agent) + rand(model.rng, 1:gain), 10)
		kill_agent!(agent, model)
		move_agent!(exploiter, (space(exploiter)..., new_lvl), model)
		# Prisoner runs away in the commotion
		prisoner.shape = :utriangle
		prisoner.capture_time = 0
		walk!(prisoner, (rand(model.rng, -1:1, 2)...), model)
	end
end
