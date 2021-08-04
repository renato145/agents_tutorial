# The envirnonment contains sheep, wolves and grass
# wolves eat sheep
# sheep eat grass

using Agents, Random

mutable struct SheepWolf <: AbstractAgent
	id::Int
	pos::Dims{2}
	type::Symbol # :sheep or :wolf
	energy::Float64
	reproduction_prob::Float64
	Δenergy::Float64
end

Sheep(id, pos, energy, repr, Δe) = SheepWolf(id, pos, :sheep, energy, repr, Δe)
Wolf(id, pos, energy, repr, Δe) = SheepWolf(id, pos, :wolf, energy, repr, Δe)

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
	properties = (
		fully_grown = falses(dims),
		countdown = zeros(Int, dims),
		regrowth_time = regrowth_time,
	)

	model = ABM(SheepWolf, space; properties, rng, scheduler=Schedulers.randomly)

	id = 0
	for _ in 1:n_sheep
		id += 1
		energy = rand(1:(Δenergy_sheep*2)) - 1
		sheep = Sheep(id, (0,0), energy, sheep_reproduce, Δenergy_sheep)
		add_agent!(sheep, model)
	end
	for _ in 1:n_wolves
		id += 1
		energy = rand(1:(Δenergy_wolf*2)) - 1
		wolf = Wolf(id, (0,0), energy, wolf_reproduce, Δenergy_wolf)
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
	if rand(model.rng) <= sheep.reproduction_prob
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
	if rand(model.rng) <= wolf.reproduction_prob
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
		agent.reproduction_prob,
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

model = initialize_model()

using InteractiveDynamics, CairoMakie

offset(a) = a.type == :sheep ? (-0.7,-0.5) : (-0.3,-0.5)
ashape(a) = a.type == :sheep ? :circle : :utriangle
acolor(a) = a.type == :sheep ? RGBAf0(1.0, 1.0, 1.0, 0.8) : RGBAf0(0.2, 0.2, 0.2, 0.8)

# Let `abm_plot` treat grass as a heatmap
grasscolor(model) = model.countdown ./ model.regrowth_time

heatkwargs = (colormap=[:brown, :green], colorrange=(0,1))

plotkwargs = (
	ac = acolor,
	as = 15,
	am = ashape,
	offset = offset,
	heatarray = grasscolor,
	heatkwargs = heatkwargs,
)

fig,_ = abm_plot(model; plotkwargs...)
fig

# Define data collection
sheep(a) = a.type == :sheep
wolves(a) = a.type == :wolf
count_grass(model) = count(model.fully_grown)

# Run simulation
model = initialize_model()
n = 500
adata = [(sheep,count), (wolves,count)]
mdata = [count_grass]
adf,mdf = run!(model, sheepwolf_step!, grass_step!, n; adata, mdata)

function plot_population_timeseries(adf, mdf)
	figure = Figure(resolution = (600,400))
	ax = figure[1,1] = Axis(figure; xlabel="Step", ylabel="Population")
	sheepl = lines!(ax, adf.step, adf.count_sheep, color=:blue)
	wolfl = lines!(ax, adf.step, adf.count_wolves, color=:orange)
	grassl = lines!(ax, adf.step, mdf.count_grass, color=:green)
	figure[1,2] = Legend(figure, [sheepl,wolfl,grassl], ["Sheep","Wolves","Grass"])
	figure
end

plot_population_timeseries(adf, mdf)

# equilibrium example
model = initialize_model(
	n_wolves = 20,
	dims = (25,25),
	Δenergy_sheep = 5,
	sheep_reproduce = 0.2,
	wolf_reproduce = 0.08,
	seed = 7758,
)

adf,mdf = run!(model, sheepwolf_step!, grass_step!, n; adata, mdata)
plot_population_timeseries(adf, mdf)

# Video
model = initialize_model(
	n_wolves = 20,
	dims = (25,25),
	Δenergy_sheep = 5,
	sheep_reproduce = 0.2,
	wolf_reproduce = 0.08,
	seed = 7758,
)

abm_video(
	"sheepwolf.mp4",
	model,
	sheepwolf_step!,
	grass_step!;
	frames=300,
	framerate=8,
	plotkwargs...,
)
