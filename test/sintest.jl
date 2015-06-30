using Gadfly
using DeepQLearning, NNGraph
xs = linspace(0,360,100)
ys = round(sin(degrees2radians(xs)),3)

# plot(x=xs,y=ys)


m = DQN(2,5,3)
s0 = NNMatrix([0.,0.]')
a0 = 2
r0 = 0.
for epoch = 1:101
    err = 0
    for i = 2:length(xs)
        x, x2, y = xs[i],xs[i-1], ys[i]
        s = [x,x2]
        s1 = NNMatrix(s')
        println((size(s)),size(s1.w))

        a1 = act(m,s1)

        r1 = sign(a1-2) * y
        err += r1
#         println((i,a1,r1))
        learn(m,s0,a0,r0,s1)
        s0 = s1
        a0 = a1
        r0 = r1
    end
    err /= length(xs)-1
    if epoch % 100 == 0 println("$epoch err = $(round(err,3))") end
end
