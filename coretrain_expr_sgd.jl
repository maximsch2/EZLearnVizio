using ProgressMeter, JLD, StatsBase
using HDF5, PyCall, Knet
SEED = Int32(get(PARAMS, "global.seed", 123456789))

@pyimport numpy as np
np.random[:seed](SEED)
@pyimport tensorflow as tf
tf.set_random_seed(SEED)

# @pyimport keras
# @pyimport keras.callbacks as kC
# @pyimport keras.models as kM
# @pyimport keras.layers as kL
# @pyimport keras.regularizers as kR


function get_class_weight(labels_matrix, smooth=0.1)
  labels = Int64[]
  for i in 1:size(labels_matrix, 1)
    append!(labels, find(labels_matrix[i, :]))
  end

  dict = countmap(labels)

  maxval = maximum(values(dict))
  result = Dict{Int64, Float64}()
  for k in keys(dict)
    result[k-1] = maxval*(1+smooth)/(dict[k] + maxval*smooth)
  end
  return result
end

immutable VecDataProvider
  data
  sample_idx
end


immutable SGDExpressionClassifier <: ExprView
  id::String
  prov::VecDataProvider
  batch_size
  val_split
  L2Reg
  optimizer
  droplr
  use_cw
end

SGDExpressionClassifier(prov; batch_size=256, val_split=0.05, L2Reg=1e-4, optimizer="rmsprop", droplr=true,
                          use_cw=false) =
                          SGDExpressionClassifier("expr", prov, batch_size, val_split, L2Reg, optimizer, droplr, use_cw)


function load_vec_data(fn, samples_obj, data_obj)
    samples, data = h5open(fn, "r") do f
        read(f[samples_obj]), read(f[data_obj])
    end

    samples_idx = Dict([(samples[i], i) for i=1:length(samples)])

    return VecDataProvider(data, samples_idx)
end

get_all_samples(prov::VecDataProvider) = collect(keys(prov.sample_idx))

get_one_vec(prov::VecDataProvider, sample) = prov.data[:, prov.sample_idx[sample]]
function collect_vecs(prov::VecDataProvider, samples)
    all_X = []
    for sample in samples
        push!(all_X, get_one_vec(prov, sample))
    end
    return hcat(all_X...)'
end


function get_labels_from_dict(gsms, labels_dict::Dict{String, Vector{String}}, idx_dict)
    Nsamples = length(gsms)
    y = zeros(Float64, (length(idx_dict), Nsamples))
    for i in 1:Nsamples
        gsm = gsms[i]
        if haskey(labels_dict, gsm)
          cur_labels = labels_dict[gsm]
          for term_id in cur_labels
              term_idx = idx_dict[term_id]
              y[term_idx, i] = 1
          end
        end
    end
    y
end

train_and_predict(v::SGDExpressionClassifier, labels, task) = train_expr_sgd_file(v, task.all_samples, labels)

function custom_minibatch(dsize, batchsize)
    indi = Any[]
    for i=1:batchsize:dsize
        j=min(i+batchsize-1,dsize)
        push!(indi, (i, j))
    end
    return indi
end

function getAtIndicesDict(dict_labels,indices)
    filterDict = Dict{String,Array{String,1}}()
    counter = 0
    for (n, val) in enumerate(dict_labels)
        counter+=1
        if counter in indices
          filterDict[val[1]] = val[2]
        end
    end
    return filterDict
end

function custom_earlyStopping(loss_history, patience=10,min_delta=0.01)
    # println(loss_history)
    num_epochs = length(loss_history)
    if num_epochs>patience
        delta = maximum(loss_history[num_epochs-patience+1:num_epochs])-minimum(loss_history[num_epochs-patience+1:num_epochs])
        if delta<min_delta
            println("Stopping (Early) after ",num_epochs," epochs ...")
            return true
        end
    end
    return false
end

function predict(w,x)
    for i=1:2:length(w)
        x = w[i]*mat(x) .+ w[i+1]
        if i<length(w)-1
            x = relu.(x) # max(0,x)
        end
    end
    return x
end

function loss(w,x,ygold)
    ypred = predict(w,x)
    ynorm = ypred .- log.(sum(exp.(ypred),1))
        -sum(ygold .* ynorm) / size(ygold,2)
end

function confussionmatrix(predictions, labels, d)
   c = zeros(Int32,d,d)
   for i in 1:length(labels)
       c[labels[i] + 1 ,predictions[i] + 1] += 1
   end
   return c
end

function train_expr_sgd_file(cls::SGDExpressionClassifier, all_gsms, train_labels)
    ssize = length(all_gsms)
    # approx 5% for validation
    val_size = min(10000, Int(round(ssize*0.05)))
    val_ind = rand(1:ssize, val_size)
    train_ind = setdiff(1:ssize, val_ind)

    train_labels = getAtIndicesDict(train_labels,train_ind)
    val_labels = getAtIndicesDict(train_labels,val_ind)

    train_gsms = sort(collect(keys(train_labels)))
    val_gsms = sort(collect(keys(val_labels)))

    cross_val_labels = load("data/precomp_final.jld")["data"]
    cross_val_gsms = collect(keys(cross_val_labels))

    @show SEED
    srand(SEED)
    train_gsms = train_gsms[randperm(length(train_gsms))]

    possible_labels = unique(vcat(values(train_labels)...));
    possible_labels_idx = Dict([(possible_labels[i], i) for i=1:length(possible_labels)])
    predicted_terms = Int32[parse(Int64, bto[5:end]) for bto in possible_labels]

    Nlabels = length(possible_labels)

    minibatch_size = cls.batch_size
    epochs = 100
    mini_batches = custom_minibatch(length(train_gsms),minibatch_size)
    num_batches = length(mini_batches)
    # val_mini_batches = custom_minibatch(val_size,val_batch_size)
    # num_val_batches = length(val_mini_batches)
    train_X_temp = collect_vecs(cls.prov, train_gsms[1:3])

    lossgradient = grad(loss)
    expit(x) = exp(x)/(1+exp(x))
    w = Any[ 0.1f0*randn(Float64,64,size(train_X_temp,2)), zeros(Float64,64,1),
             0.1f0*randn(Float64,size(possible_labels,1),64),  zeros(Float64,size(possible_labels,1),1) ]
    oAdam = optimizers(w, Adam)
    lr = 0.05
    # activation = kL.Dense(Nlabels, activation="softmax", input_dim=minibatch_size,
    #     kernel_regularizer=kR.l2(cls.L2Reg), kernel_initializer="zeros")(input)
    #
    # callbacks = []
    # if cls.droplr
    #   lrdrop = kC.ReduceLROnPlateau(monitor="val_loss", patience=5)
    #   # early_stopping = kC.EarlyStopping(monitor="val_loss", patience=10, min_delta=0.01)
    #   callbacks = [lrdrop]
    # end
    #
    # model = kM.Model(inputs=input, outputs=activation)
    # model[:compile](optimizer=cls.optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    val_X = collect_vecs(cls.prov, val_gsms)
    cross_val_X = collect_vecs(cls.prov, cross_val_gsms)
    val_Y = get_labels_from_dict(val_gsms, val_labels, possible_labels_idx)'
    cross_val_Y = get_labels_from_dict(cross_val_gsms, cross_val_labels, possible_labels_idx)'
    # val_Y_int = []
    # for yind in 1:size(val_Y,1)
    #     append!(val_Y_int,indmax(val_Y[yind,:]))
    # end
    cross_val_Y_int = []
    for yind in 1:size(cross_val_Y,1)
        append!(cross_val_Y_int,indmax(cross_val_Y[yind,:]))
    end
    val_X = transpose(mat(val_X))
    val_Y = transpose(mat(val_Y))
    cross_val_X = transpose(mat(cross_val_X))
    cross_val_Y = transpose(mat(cross_val_Y))
    # val_Y_int = convert(Array{UInt8},val_Y_int)

    loss_history = []
    m_history = 0
    for e in 1:epochs
        println((:epoch,e,"/",epochs))
        batchn = 0
        batch_loss_history = []
        println((:batch,0,"/",num_batches))
        for mb in 1:num_batches
            batchn+=1
            train_gsms_mini = train_gsms[mini_batches[mb][1]:mini_batches[mb][2]]
            train_labels_mini = getAtIndicesDict(train_labels,mini_batches[mb][1]:mini_batches[mb][2])
            train_X_mini = collect_vecs(cls.prov, train_gsms_mini)
            train_Y_mini = get_labels_from_dict(train_gsms_mini, train_labels_mini, possible_labels_idx)'
            # train_Y_mini_int = []
            # for yind in 1:size(train_Y_mini,1)
            #     append!(train_Y_mini_int,indmax(train_Y_mini[yind,:]))
            # end
            train_X_mini = transpose(mat(train_X_mini))
            # train_Y_mini_int = convert(Array{UInt8},train_Y_mini_int)
            train_Y_mini = transpose(train_Y_mini)
            dw = lossgradient(w, train_X_mini, train_Y_mini)
            Knet.update!(w, dw, oAdam)
            val_loss = loss(w,val_X,val_Y)
            push!(batch_loss_history,val_loss)
            println((:batch, batchn,"/",num_batches, :trn, loss(w,train_X_mini,train_Y_mini), :tst, val_loss))
        end
        push!(loss_history,batch_loss_history[end])
        if custom_earlyStopping(loss_history)
            break
        end
    end

    # for e in 1:epochs
    #     println("Epoch No.: ",e,"/",epochs)
    #     batchn = 0
    #     batch_loss_history = []
    #     for mb in 1:num_batches
    #         batchn+=1
    #         println("Batch No.: ",batchn,"/",num_batches)
    #         train_gsms_mini = train_gsms[mini_batches[mb][1]:mini_batches[mb][2]]
    #         train_labels_mini = getAtIndicesDict(train_labels,mini_batches[mb][1]:mini_batches[mb][2])
    #         train_X_mini = collect_vecs(cls.prov, train_gsms_mini)
    #         train_Y_mini = get_labels_from_dict(train_gsms_mini, train_labels_mini, possible_labels_idx)'
    #         # @show size(train_X_mini), size(train_Y_mini)
    #         if cls.use_cw
    #           cw = get_class_weight(train_Y)
    #           m_history = model[:fit](train_X_mini, train_Y_mini, epochs=1, validation_data=(val_X,val_Y), callbacks=callbacks, class_weight=cw)
    #         else
    #           m_history = model[:fit](train_X_mini, train_Y_mini, epochs=1, validation_data=(val_X,val_Y), callbacks=callbacks)
    #         end
    #         push!(batch_loss_history, m_history[:history]["val_acc"][1])
    #     end
    #     push!(loss_history, batch_loss_history[end])
    #     if custom_earlyStopping(loss_history)
    #         break
    #     end
    # end

    # Prediction in batches
    mini_batches_all = custom_minibatch(ssize,Int(round(ssize*0.1)))
    num_batches_all = length(mini_batches_all)
    all_probs = 0

    for mb in 1:num_batches_all
        mini_all_X = collect_vecs(cls.prov, all_gsms[mini_batches_all[mb][1]:mini_batches_all[mb][2]])
        mini_all_X = transpose(mat(mini_all_X))
        preds_mini = predict(w,mini_all_X)
        probs_mini = expit.(preds_mini)
        if mb==1
            all_probs = probs_mini
        else
            all_probs = hcat(all_probs, probs_mini)
        end
    end

    cross_val_preds = predict(w,cross_val_X)
    cross_val_probs = expit.(cross_val_preds)

    predictions = []
    for row_n in 1:size(cross_val_probs,2)
        max_index = indmax(cross_val_probs[:,row_n])
        push!(predictions,max_index)
    end

    cfm = confussionmatrix(predictions, cross_val_Y_int, length(possible_labels))
    println("Confusion Matrix\n",cfm)
    println("Accuracy: ",sum(Diagonal(cfm))/sum(cfm))

    @show size(all_probs)

    resultBeliefs = Dict()
    @showprogress "Collecting results..." for (i, sample) in enumerate(all_gsms)
      probs = all_probs[:,i]
      resultBeliefs[sample] = collect(zip(predicted_terms[probs .> 1e-3], probs[probs .> 1e-3]))
    end
    # TODO confirm what history to return
    # resultBeliefs, (model[:get_weights](), predicted_terms, model[:history][:history])
    resultBeliefs, (w, predicted_terms)
end
