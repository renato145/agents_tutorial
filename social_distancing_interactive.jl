using Agents, Random, InteractiveDynamics, GLMakie
using Statistics: mean

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

function transmit!(a1, a2, model)
	# Check if 1 is infected
	count(a.status == :I for a in (a1,a2)) ≠ 1 && return
	infected,healthy = a1.status == :I ? (a1,a2) : (a2,a1)

	rand(model.rng) > infected.β && return
	if healthy.status == :R
		rand(model.rng) > model.reinfection_probability && return
	end
	healthy.status = :I
end

function sir_model_step!(model)
	r = model.interaction_radius
	for (a1,a2) in interacting_pairs(model, r, :nearest)
		transmit!(a1, a2, model)
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

# Interactive exploration
infected(x) = count(i == :I for i in x)
recovered(x) = count(i == :R for i in x)
adata = [(:status, length), (:status, infected), (:status, recovered), (:days_infected, mean)]

sir_colors(a) = a.status == :S ? "#2b2b33" : a.status == :I ? "#bf2642" : "#338c54"

params = Dict(
	:death_rate => 0:0.001:0.1,
	:reinfection_probability => 0:0.01:0.2
)

sir_model = sir_initialization(βmin=0.3, βmax=0.8, isolated=0.2)
fig,agent_df,model_df = abm_data_exploration(sir_model, sir_agent_step!, sir_model_step!, params; adata, ac=sir_colors, as=10)
