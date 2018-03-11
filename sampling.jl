using HDF5, Knet

function train_test_split(x,y)
  n = size(x,2)
  r = randperm(n)
  xtrain = x[:,r[1:Int(floor(n*0.8))]];
  ytrain = y[:,r[1:Int(floor(n*0.8))]];
  xtest = x[:,r[Int(floor(n*0.8))+1:end]];
  ytest = y[:,r[Int(floor(n*0.8))+1:end]];
  return xtrain, ytrain, xtest, ytest
end

function train(w, data; lr=.1)
    for (x,y) in data
        dw = lossgradient(w, x, y)
        for i in 1:length(w)
            w[i] -= lr * dw[i]
        end
    end
    return w
end

predict(w,x) = w[1]*mat(x) .+ w[2]
loss(w,x,ygold) =  nll(predict(w,x), ygold)

lossgradient  = grad(loss)

# Testing functions with housing data
include(Knet.dir("data","mnist.jl"))
xtrn, ytrn, xtst, ytst = mnist()
batchsize = 100
w = Any[ 0.1*randn(Float32,10,784), zeros(Float32,10,1) ]

# xtrain, ytrain, xtest, ytest = train_test_split(x,y)
dtrn = minibatch(xtrn, ytrn, batchsize)
dtst = minibatch(xtst, ytst, batchsize)

println((:epoch, 0, :trn, accuracy(w,dtrn,predict), :tst, accuracy(w,dtst,predict)))
for epoch=1:10
  train(w, dtrn; lr=0.5)
  println((:epoch, epoch, :trn, accuracy(w,dtrn,predict), :tst, accuracy(w,dtst,predict)))
end
