module DeepQLearning

using NNGraph

export DQN, forward, act, learn
include("dqn.jl")

end # module
