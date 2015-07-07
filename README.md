# DeepQLearning.jl
Julia implementation of DeepMind's Deep Q-Learning algorithm as described in [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602). This code only implements the base algorithm. It does not include the code for a convolutional network. However, this can be easily added using Mocha.jl. In lieu of this is uses a simpler single layer neural network. Information on the original [RecurrenJS DQN implementation can be found here](http://cs.stanford.edu/people/karpathy/reinforcejs/waterworld.html)

*note: This library has been tested on various learning tasks and seems to be functioning correctly, but is not yet ready for public consumption.*

## Example code
```julia
using DeepQLearning

... coming soon ... I hope :)
```

##Dependencies 
This library requires [NNGraph.jl](https://github.com/Andy-P/NNGraph.jl). 

##Credits
This library draws on the work of [Andrej Karpathy](https://github.com/karpathy/reinforcejs)

## License
MIT
