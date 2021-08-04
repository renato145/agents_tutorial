# The envirnonment contains sheep, wolves and grass
# wolves eat sheep
# sheep eat grass
using Agents, Random, InteractiveDynamics, GLMakie

mutable struct SheepWolf <: AbstractAgent
	id::Int
	pos::Dims{2}
	type::Symbol # :sheep or :wolf
	energy::Float64
	Δenergy::Float64
end

Sheep(id, pos, energy, Δe) = SheepWolf(id, pos, :sheep, energy, Δe)
Wolf(id, pos, energy, Δe) = SheepWolf(id, pos, :wolf, energy, Δe)

function initialize_model(;
	n_sheep = 100,
	n_wolves = 50,
	dims = (20,20),
	regrowth_time = 30,
	Δenergy_sheep = 4,
	Δenergy_wolf = 20,
	sheep_reproduce = 0.04,
	wolf_reproduce = 0.05,
	seed = 23182,
)
	rng = MersenneTwister(seed)
	space = GridSpace(dims, periodic = false)

	# This is a named tuple to manage grass regrowth
	properties = Dict(
		:fully_grown => falses(dims),
		:countdown => zeros(Int, dims),
		:sheep_reproduce => sheep_reproduce,
		:wolf_reproduce => wolf_reproduce,
		:regrowth_time => regrowth_time,
	)

	model = ABM(SheepWolf, space; properties, rng, scheduler=Schedulers.randomly)

	id = 0
	for _ in 1:n_sheep
		id += 1
		energy = rand(1:(Δenergy_sheep*2)) - 1
		sheep = Sheep(id, (0,0), energy, Δenergy_sheep)
		add_agent!(sheep, model)
	end
	for _ in 1:n_wolves
		id += 1
		energy = rand(1:(Δenergy_wolf*2)) - 1
		wolf = Wolf(id, (0,0), energy, Δenergy_wolf)
		add_agent!(wolf, model)
	end

	for p in positions(model) # Grass initialization
		fully_grown = rand(model.rng, Bool)
		countdown = fully_grown ? regrowth_time : rand(model.rng, 1:regrowth_time) - 1
		model.countdown[p...] = countdown
		model.fully_grown[p...] = fully_grown
	end

	return model
end

function sheepwolf_step!(agent::SheepWolf, model)
	if agent.type == :sheep
		sheep_step!(agent, model)
	else
		wolf_step!(agent, model)
	end
end

function sheep_step!(sheep, model)
	walk!(sheep, rand, model)
	sheep.energy -= 1
	sheep_eat!(sheep, model)
	if sheep.energy < 0
		kill_agent!(sheep, model)
		return
	end
	if rand(model.rng) <= model.sheep_reproduce
		reproduce!(sheep, model)
	end
end

function wolf_step!(wolf, model)
	walk!(wolf, rand, model)
	wolf.energy -= 1
	agents = collect(agents_in_position(wolf.pos, model))
	dinner = filter!(x -> x.type == :sheep, agents)
	wolf_eat!(wolf, dinner, model)
	if wolf.energy < 0
		kill_agent!(wolf, model)
		return
	end
	if rand(model.rng) <= model.wolf_reproduce
		reproduce!(wolf, model)
	end
end

function sheep_eat!(sheep, model)
	if model.fully_grown[sheep.pos...]
		sheep.energy += sheep.Δenergy
		model.fully_grown[sheep.pos...] = false
	end
end

function wolf_eat!(wolf, dinner, model)
	if !isempty(dinner)
		dinner = rand(model.rng, dinner)
		kill_agent!(dinner, model)
		wolf.energy += wolf.Δenergy
	end
end

function reproduce!(agent, model)
	agent.energy /= 2
	id = nextid(model)
	offspring = SheepWolf(
		id, agent.pos,
		agent.type,
		agent.energy,
		agent.Δenergy,
	)
	add_agent_pos!(offspring, model)
	return
end

function grass_step!(model)
	# @inbounds skips checking for performance
	@inbounds for p in positions(model)
		if !(model.fully_grown[p...])
			if model.countdown[p...] ≤ 0
				model.fully_grown[p...] = true
				model.countdown[p...] = model.regrowth_time
			else
				model.countdown[p...] -= 1
			end
		end
	end
end

offset(a) = a.type == :sheep ? (-0.7,-0.5) : (-0.3,-0.5)
ashape(a) = a.type == :sheep ? :circle : :utriangle
acolor(a) = a.type == :sheep ? RGBAf0(1.0, 1.0, 1.0, 0.8) : RGBAf0(0.2, 0.2, 0.2, 0.8)

# Let `abm_plot` treat grass as a heatmap
grasscolor(model) = model.countdown ./ model.regrowth_time
heatkwargs = (colormap=[:brown, :green], colorrange=(0,1))
plotkwargs = (ac = acolor, as = 15, am = ashape, offset = offset, heatarray = grasscolor, heatkwargs = heatkwargs)

# fig,_ = abm_plot(model; plotkwargs...)
# fig

# Define data collection
sheep(a) = a.type == :sheep
wolves(a) = a.type == :wolf
count_grass(model) = count(model.fully_grown)
adata = [(sheep,count), (wolves,count)]
mdata = [count_grass]

# Interactive window
params = Dict(
	:sheep_reproduce => 0:0.01:1.0,
	:wolf_reproduce => 0:0.01:1.0,
	:regrowth_time => 0:1:100
)

# equilibrium example
model = initialize_model(
	n_wolves = 20,
	dims = (25,25),
	Δenergy_sheep = 5,
	sheep_reproduce = 0.2,
	wolf_reproduce = 0.08,
	seed = 7758,
)

fig,adf,mdf = abm_data_exploration(model, sheepwolf_step!, grass_step!, params; adata, mdata, plotkwargs...)
