using ProgressMeter, JLD, StatsBase
using HDF5, PyCall
SEED = Int32(get(PARAMS, "global.seed", 123456789))

@pyimport numpy as np
np.random[:seed](SEED)
@pyimport tensorflow as tf
tf.set_random_seed(SEED)

@pyimport keras
@pyimport keras.callbacks as kC
@pyimport keras.models as kM
@pyimport keras.layers as kL
@pyimport keras.regularizers as kR


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
    y = zeros(Float32, (length(idx_dict), Nsamples))
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

function vizio_minibatch(dsize, batchsize)
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

function train_expr_sgd_file(cls::SGDExpressionClassifier, all_gsms, train_labels)
    ssize = length(all_gsms)
    # approx 5% for validation
    val_size = Int(round(ssize*0.05))
    val_ind = rand(1:ssize,val_size)
    train_ind = setdiff(1:ssize,val_ind)

    train_labels = getAtIndicesDict(train_labels,train_ind)
    val_labels = getAtIndicesDict(train_labels,val_ind)

    train_gsms = sort(collect(keys(train_labels)))
    val_gsms = sort(collect(keys(val_labels)))

    @show SEED
    srand(SEED)
    train_gsms = train_gsms[randperm(length(train_gsms))]

    possible_labels = unique(vcat(values(train_labels)...));
    possible_labels_idx = Dict([(possible_labels[i], i) for i=1:length(possible_labels)])
	predicted_terms = Int32[parse(Int64, bto[5:end]) for bto in possible_labels]

    # labels_int = Int32[parse(Int32, getAtIndicesDict(bto,5:length(bto))) for bto in possible_labels]
    Nlabels = length(possible_labels)

    minibatch_size = val_size
    epochs = 5
    mini_batches = vizio_minibatch(length(train_gsms),minibatch_size)
    num_batches = length(mini_batches)
	# 2048 = size(train_x,2)
    input = kL.Input(shape=(2048,))
    activation = kL.Dense(Nlabels, activation="softmax", input_dim=minibatch_size,
        kernel_regularizer=kR.l2(cls.L2Reg), kernel_initializer="zeros")(input)

    # TODO implement without keras
    if cls.droplr
      lrdrop = kC.ReduceLROnPlateau(monitor="val_loss", patience=5)
      early_stopping = kC.EarlyStopping(monitor="val_loss", patience=10, min_delta=0.01)
      callbacks = [lrdrop, early_stopping]
    else
      early_stopping = kC.EarlyStopping(monitor="val_loss", patience=1, min_delta=0.01)
      callbacks = [early_stopping]
    end

    model = kM.Model(inputs=input, outputs=activation)
    model[:compile](optimizer=cls.optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # train_Y = get_labels_from_dict(train_gsms, train_labels, possible_labels_idx)'
    # val_X = collect_vecs(cls.prov, val_gsms)
    val_X = collect_vecs(cls.prov, val_gsms)
    val_Y = get_labels_from_dict(val_gsms, val_labels, possible_labels_idx)'

    for e in 1:epochs
        println("Epoch: ",e,"/",epochs)
        batchn = 0
        for mb in 1:num_batches
            batchn+=1
            println("Batch No: ",batchn,"/",num_batches)
            train_gsms_mini = train_gsms[mini_batches[mb][1]:mini_batches[mb][2]]
            train_labels_mini = getAtIndicesDict(train_labels,mini_batches[mb][1]:mini_batches[mb][2])
            train_X_mini = collect_vecs(cls.prov, train_gsms_mini)
            train_Y_mini = get_labels_from_dict(train_gsms_mini, train_labels_mini, possible_labels_idx)'
            # @show size(train_X_mini), size(train_Y_mini)
            if cls.use_cw
              cw = get_class_weight(train_Y)
              model[:fit](train_X_mini, train_Y_mini, epochs=1, validation_data=(val_X,val_Y), callbacks=callbacks, class_weight=cw)
            else
              model[:fit](train_X_mini, train_Y_mini, epochs=1, validation_data=(val_X,val_Y), callbacks=callbacks)
            end
        end
        # Validation accuracy
        # val_X = collect_vecs(cls.prov, val_gsms)
        # val_probs = model[:predict](all_X, verbose=false)
        # count = 0
		# TODO validation loss, accuracy
        # for (i, vp) in enumerate(val_probs)
        #     vp_probs = val_probs[i, :]
        #     vp_label = predicted_terms[indmax(vp_probs)]
        #     count+= vp_label == val_labels[i]
        # end
        # println("Validation Acc: ",count/val_size)
    end

    all_X = collect_vecs(cls.prov, all_gsms)
    all_probs = model[:predict](all_X, verbose=true)

    @show size(all_probs)

    resultBeliefs = Dict()
    @showprogress "Collecting results..." for (i, sample) in enumerate(all_gsms)
      probs = all_probs[i, :]
      resultBeliefs[sample] = collect(zip(predicted_terms[probs .> 1e-3], probs[probs .> 1e-3]))
    end
    resultBeliefs, (model[:get_weights](), predicted_terms, model[:history][:history])
end
