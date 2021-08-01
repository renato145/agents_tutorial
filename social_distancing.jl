using Agents, Random

mutable struct Agent <: AbstractAgent
	id::Int
	pos::NTuple{2,Float64}
	vel::NTuple{2,Float64}
	mass::Float64
end

function ball_model(; speed=0.002)
	space2d = ContinuousSpace((1,1), 0.02)
	model = ABM(Agent, space2d, properties = Dict(:dt => 1.0), rng = MersenneTwister(42))

	# Add Agents
	for ind in 1:500
		pos = Tuple(rand(model.rng, 2))
		vel = sincos(2π * rand(model.rng)) .* speed
		add_agent!(pos, model, vel, 1.0)
	end
	model
end

model = ball_model()

agent_step!(agent, model) = move_agent!(agent, model, model.dt)

using InteractiveDynamics, CairoMakie

abm_video(
	"socialdist1.mp4",
	model,
	agent_step!,
	title = "Ball Model",
	frames = 50,
	spf = 2,
	framerate = 25
)

# Model interactions
function model_step!(model)
	for (a1,a2) in interacting_pairs(model, 0.012, :nearest)
		elastic_collision!(a1, a2, :mass)
	end
end

model2 = ball_model()

abm_video(
	"socialdist2.mp4",
	model2,
	agent_step!,
	model_step!,
	title = "Billiard-like",
	frames = 50,
	spf = 2,
	framerate = 25
)

# Immovable agents
model3 = ball_model()

for id in 1:400
	agent = model3[id]
	agent.mass = Inf
	agent.vel = (0.0,0.0)
end

abm_video(
	"socialdist3.mp4",
	model3,
	agent_step!,
	model_step!,
	title = "Billiard-like with stationary agents",
	frames = 50,
	spf = 2,
	framerate = 25
)

# Adding virus spread (SIR)

mutable struct PoorSoul <: AbstractAgent
	id::Int
	pos::NTuple{2,Float64}
	vel::NTuple{2,Float64}
	mass::Float64
	days_infected::Int
	status::Symbol
	β::Float64 # transmission probability
end

const steps_per_day = 24

using DrWatson: @dict

function sir_initialization(;
	infection_period = 30 * steps_per_day,
	detection_time = 14 * steps_per_day,
	reinfection_probability = 0.05,
	isolated = 0.0, # in percentage
	interaction_radius = 0.012,
	dt = 1.0,
	speed = 0.002,
	death_rate = 0.044, # from website of WHO
	N = 1000,
	initial_infected = 5,
	seed = 42,
	βmin = 0.4,
	βmax = 0.8,
)
	properties = @dict(
		infection_period,
		reinfection_probability,
		detection_time,
		death_rate,
		interaction_radius,
		dt,
	)
	space = ContinuousSpace((1,1), 0.02)
	model = ABM(PoorSoul, space, properties = properties, rng = MersenneTwister(seed))

	# Add individuals
	for ind in 1:N
		pos = Tuple(rand(model.rng, 2))
		status = ind ≤ N - initial_infected ? :S : :I
		isisolated = ind ≤ isolated * N
		mass = isisolated ? Inf : 1.0
		vel = isisolated ? (0.0,0.0) : sincos(2π * rand(model.rng)) .* speed

		# very high transmission probability (close encounters)
		β = (βmax - βmin) * rand(model.rng) + βmin
		add_agent!(pos, model, vel, mass, 0, status, β)
	end

	model
end

sir_model = sir_initialization()
sir_colors(a) = a.status == :S ? "#2b2b33" : a.status == :I ? "#bf2642" : "#338c54"

fig,abmstepper = abm_plot(sir_model; ac = sir_colors)
fig

function transmit!(a1, a2, rp)
	# Check if 1 is infected
	count(a.status == :I for a in (a1,a2)) ≠ 1 && return
	infected,healthy = a1.status == :I ? (a1,a2) : (a2,a1)

	rand(model.rng) > infected.β && return
	if healthy.status == :R
		rand(model.rng) > rp && return
	end
	healthy.status = :I
end

function sir_model_step!(model)
	r = model.interaction_radius
	for (a1,a2) in interacting_pairs(model, r, :nearest)
		transmit!(a1, a2, model.reinfection_probability)
		elastic_collision!(a1, a2, :mass)
	end
end


function sir_agent_step!(agent, model)
	move_agent!(agent, model, model.dt)
	update!(agent)
	recover_or_die!(agent, model)
end

update!(agent) = agent.status == :I && (agent.days_infected += 1)

function recover_or_die!(agent, model)
	if agent.days_infected ≥ model.infection_period
		if rand(model.rng) ≤ model.death_rate
			kill_agent!(agent, model)
		else
			agent.status = :R
			agent.days_infected = 0
		end
	end
end

sir_model = sir_initialization()

abm_video(
	"socialdist4.mp4",
	sir_model,
	sir_agent_step!,
	sir_model_step!;
	title = "SIR Model",
	frames = 300,
	ac = sir_colors,
	as = 10,
	spf = 1, # steps per frame
	framerate = 20,
)

# Exponential spread

infected(x) = count(i == :I for i in x)
recovered(x) = count(i == :R for i in x)
adata = [(:status, infected), (:status, recovered)]

r1,r2 = 0.04,0.33
β1,β2 = 0.5,0.1
sir_model1 = sir_initialization(reinfection_probability=r1, βmin=β1)
sir_model2 = sir_initialization(reinfection_probability=r2, βmin=β1)
sir_model3 = sir_initialization(reinfection_probability=r1, βmin=β2)

data1,_ = run!(sir_model1, sir_agent_step!, sir_model_step!, 2000; adata)
data2,_ = run!(sir_model2, sir_agent_step!, sir_model_step!, 2000; adata)
data3,_ = run!(sir_model3, sir_agent_step!, sir_model_step!, 2000; adata)

data1[(end-10):end, :]

figure = Figure()
ax = figure[1,1] = Axis(figure; ylabel = "Infected")
l1 = lines!(ax, data1[:, dataname((:status, infected))], color=:orange)
l2 = lines!(ax, data2[:, dataname((:status, infected))], color=:blue)
l3 = lines!(ax, data3[:, dataname((:status, infected))], color=:green)
figure[1,2] = Legend(figure, [l1,l2,l3], ["r=$r1, beta=$β1", "r=$r2, beta=$β1", "r=$r1, beta=$β2"])
figure

# Social distancing
sir_model = sir_initialization(isolated = 0.8)

abm_video(
	"socialdist5.mp4",
	sir_model,
	sir_agent_step!,
	sir_model_step!;
	title = "Social Distancing",
	frames = 300,
	ac = sir_colors,
	as = 10,
	spf = 2, # steps per frame
	framerate = 20,
)

r4 = 0.04
sir_model4 = sir_initialization(reinfection_probability=r4, βmin=β1, isolated=0.8)

data4,_ = run!(sir_model4, sir_agent_step!, sir_model_step!, 2000; adata)

l4 = lines!(ax, data4[:, dataname((:status, infected))], color=:red)
figure[1,2] = Legend(
	figure,
	[l1,l2,l3,l4],
	["r=$r1, beta=$β1", "r=$r2, beta=$β1", "r=$r1, beta=$β2", "r=$r4, social distancing"],
)
figure

# Interactive Visualization
using InteractiveDynamics
using GLMakie

sir_model = sir_initialization()
fig,abmstepper = abm_plot(sir_model; ac = sir_colors)
fig

# Manual doing 10 steps
Agents.step!(abmstepper, sir_model, agent_step!, model_step!, 10)
fig

# Play
abm_play(sir_model, sir_agent_step!, sir_model_step!; ac=sir_colors, as=10)

# Interactive exploration
sir_model = sir_initialization(βmin=0.3, βmax=0.8, isolated=0.2)

adata = [(:status, infected), (:status, recovered)]

params = Dict(
	:isolated => 0:0.1:1,
	:βmin => 0:0.1:1,
	:βmax => 0:0.1:1,
)

fig,agent_df,model_df = abm_data_exploration(sir_model, sir_agent_step!, sir_model_step!, params; adata, ac=sir_colors, as=10)
