using HDF5, Knet

ccount(a)=(ndims(a)==1 ? 1 : size(a,ndims(a)))
cget(a,i)=(ndims(a)==1 ? error() : getindex(a, ntuple(i->(:), ndims(a)-1)..., i))

function train_test_split(x,y)
  n = size(x,2)
  r = randperm(n)
  xtrain = x[:,r[1:Int(floor(n*0.8))]];
  ytrain = y[:,r[1:Int(floor(n*0.8))]];
  xtest = x[:,r[Int(floor(n*0.8))+1:end]];
  ytest = y[:,r[Int(floor(n*0.8))+1:end]];
  return xtrain, ytrain, xtest, ytest
end

function train(w, data; lr=.1, epochs=1)
  tloss = []
  for epoch = 1:epochs
    eloss = 0
    for (x,y) in data
      eloss += loss(w, x, y)
      g = lossgradient(w, x, y)
      for i in 1:length(w)
        w[i] -= lr * g[i]
      end
    end
    push!(tloss, eloss/length(data))
  end
  return w
end

function accuracy(model, data)
  ncorrect = ninstance = 0
  for (x, y) in data
      ypred = predict(w, x)
      ncorrect += sum(y .* (ypred .== maximum(ypred,1)))
      ninstance += size(y,2)
  end
  return ncorrect/ninstance
end

function minibatch(x, y, batchsize)
    data = Any[]
    for i=1:batchsize:ccount(x)
        j=min(i+batchsize-1,ccount(x))
        push!(data, (cget(x,i:j), cget(y,i:j)))
    end
    return data
end

predict(w,x) = w[1]*x .+ w[2]
loss(w,x,y) = (sumabs2(y-predict(w,x)) / size(x,2))

lossgradient  = grad(loss)

# TODO make minibatch compatible with vizio data

# Testing functions with housing data
# include("data/housing.jl")
# x,y = housing()
# w = Any[0.1*randn(1,13), 0.0]
#
# xtrain, ytrain, xtest, ytest = train_test_split(x,y)
# batchsize = 100
#
# dtrain = minibatch(xtrain, ytrain, batchsize)
# dtest = minibatch(xtest, ytest, batchsize)
#
# for i=1:10; train(w, [(xtrain,ytrain)]); println(loss(w,xtrain,ytrain)); end
#
# println(accuracy(w, dtest))
