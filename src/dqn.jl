immutable Experience
    s0::NNMatrix
    a0::Int64
    r0::Float64
    s1::NNMatrix
end

type DQN
    matrices::Array{NNMatrix,1}
    numStates::Int64 # length of the state vector (input vector)
    numHidden::Int64 # number of hidden nodes
    numActions::Int64 # number of actions (output vector)
    w1::NNMatrix
    b1::NNMatrix
    w2::NNMatrix
    b2::NNMatrix
    gamma::Float64 # reward discount
    epsilon::Float64 # epsilon-greedy policy
    alpha::Float64 # learning rate
    errorClamp::Float64
    expSize::Int64
    expAddProb::Float64
    expLearn::Int64
    expCnt::Int64
    experiences::Vector{Experience}
    s0::NNMatrix
    a0::Int64
    r0::Float64
    s1::NNMatrix
    a1::Int64
    lastG::Graph
    solver::Solver
    function DQN(numStates::Int64, numHidden::Int64, numActions::Int64;
                 std=0.01, gamma=0.75, epsilon=0.5, alpha=0.01, errorClamp=2.0,
                 expSize=5000, expAddProb=0.2, expLearn=10)

        matrices = Array(NNMatrix, 0) # reference to matrices used by solver
        w1 = randNNMat(numHidden,  numStates, std);  push!(matrices, w1)
        b1 = randNNMat(numHidden,          1, std);  push!(matrices, b1)
        w2 = randNNMat(numActions, numHidden, std);  push!(matrices, w2)
        b2 = randNNMat(numActions,         1, std);  push!(matrices, b2)

        new(matrices, numStates, numHidden, numActions, w1, b1, w2, b2,
            gamma, epsilon, alpha, errorClamp, expSize, expAddProb, expLearn,0, Array(Experience,0),
            NNMatrix(numStates, 1), 0, typemin(Float64), NNMatrix(numStates, 1), 0, Graph(),Solver())
    end
end

function forward(m::DQN,s::NNMatrix, doBP::Bool=false)
    g = Graph(false)
    h0 =  add(g, mul(g, m.w1, s), m.b1)
    hd = NNGraph.tanh(g, h0) # hidden state
    a = add(g, mul(g, m.w2, h0), m.b2) # action vector
    m.lastG = g
    return a
end

function act(m::DQN,s::NNMatrix)
    a = rand()<=m.epsilon? rand(1:m.numActions):indmax(forward(m,s,false).w)
    m.s0 = m.s1; m.a0 = m.a1 # shift old state/action pair
    m.s1 = s;  m.a1 = a # record new state/action pair
    return a # return selected action
end

function learnFromTuple(m::DQN, s0::NNMatrix, a0::Int64, r0::Float64, s1::NNMatrix)
    tmat = forward(m,s1,false)
    qmax = r0 + m.gamma * tmat.w[indmax(tmat.w)]

    pred = forward(m, s0, true)
#     println((a0, indmax(tmat.w), size(pred.w)))
    tdError = pred.w[a0,1] - qmax
    tdError = minimum([maximum([tdError,-m.errorClamp]),m.errorClamp]) # huber loss to robustify
    pred.dw[a0] = tdError
    backprop(m.lastG)
    # solverstats = step(m.solver, m.matrices, m.alpha, 1e-06, m.errorClamp)

    for k = 1:length(m.matrices)
        @inbounds mat = m.matrices[k] # mat ref
        @inbounds for j = 1:mat.d, i = 1:mat.n
            mat.w[i,j] += - m.alpha * mat.dw[i]
            mat.dw[i,j] = 0
        end
    end
    return tdError
end

function learn(m::DQN, r1::Float64)

    if m.alpha > 0 && m.r0 != typemin(Float64) && m.a0 != 0
        tdError = learnFromTuple(m, m.s0, m.a0, m.r0, m.s1)

        if rand() <=  m.expAddProb
            m.expCnt = m.expCnt >= m.expSize? 1 : m.expCnt + 1
#             println( m.expCnt)
            if length(m.experiences) < m.expSize
                push!(m.experiences,Experience(m.s0, m.a0, m.r0, m.s1))
            else
                m.experiences[m.expCnt] = Experience(m.s0, m.a0, m.r0, m.s1)
            end
        end

        #　memory replay
        len = length(m.experiences)
        for i = 1:min(len,m.expLearn)
            r = rand(1:len)
            exp = m.experiences[r]
            err = learnFromTuple(m, exp.s0, exp.a0, exp.r0, exp.s1)
        end
    end

    m.r0 = r1
end
