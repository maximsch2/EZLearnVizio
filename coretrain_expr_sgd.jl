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


function train_expr_sgd_file(cls::SGDExpressionClassifier, all_gsms, train_labels)

    train_gsms = sort(collect(keys(train_labels)))
    @show SEED
    srand(SEED)
    train_gsms = train_gsms[randperm(length(train_gsms))]

    possible_labels = unique(vcat(values(train_labels)...));
    possible_labels_idx = Dict([(possible_labels[i], i) for i=1:length(possible_labels)])

    Nlabels = length(possible_labels)

    train_Y = get_labels_from_dict(train_gsms, train_labels, possible_labels_idx)'

    train_X = collect_vecs(cls.prov, train_gsms)
    @show size(train_X), size(train_Y)

    input = kL.Input(shape=(size(train_X, 2),))
    activation = kL.Dense(Nlabels, activation="softmax", input_dim=size(train_X, 2), 
                          kernel_regularizer=kR.l2(cls.L2Reg), kernel_initializer="zeros")(input)
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
    if cls.use_cw
      cw = get_class_weight(train_Y')
      model[:fit](train_X, train_Y, epochs=100, batch_size=cls.batch_size, validation_split=cls.val_split, callbacks=callbacks, class_weight=cw)
    else
      model[:fit](train_X, train_Y, epochs=100, batch_size=cls.batch_size, validation_split=cls.val_split, callbacks=callbacks)
    end

    all_X = collect_vecs(cls.prov, all_gsms)
    all_probs = model[:predict](all_X, verbose=true)

    @show size(all_probs)

    predicted_terms = Int32[parse(Int64, bto[5:end]) for bto in possible_labels]
    resultBeliefs = Dict()
    @showprogress "Collecting results..." for (i, sample) in enumerate(all_gsms)
      probs = all_probs[i, :]
      resultBeliefs[sample] = collect(zip(predicted_terms[probs .> 1e-3], probs[probs .> 1e-3]))
    end
    resultBeliefs, (model[:get_weights](), predicted_terms, model[:history][:history])
end

