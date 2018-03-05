using ProgressMeter, HDF5

immutable TextDataProvider
  strings::Vector{String}
  samples::Vector{String}
  samples_index::Dict
end

immutable FastTextClassifier <: TextView
    id::String
    params::String
    prov::TextDataProvider
    balance::Integer
end

function get_strings_for_samples(prov::TextDataProvider, gsms)
  idx = [prov.samples_index[uppercase(gsm)] for gsm in gsms]
  return prov.strings[idx]
end

get_all_samples(prov::TextDataProvider) = prov.samples

function load_strings_data(h5fn)
  string_dataF = h5open(h5fn, "r")
  strings_gsms = read(string_dataF["gsms"]);
  strings_gsms_idx = Dict([(uppercase(gsm), i) for (i, gsm) in enumerate(strings_gsms)]);
  strings_data = read(string_dataF["strings"]);
  return TextDataProvider(strings_data, strings_gsms, strings_gsms_idx)
end


FastTextClassifier(params, provider; balance=0) = FastTextClassifier("text", params, provider, balance)

train_and_predict(v::FastTextClassifier, labels, task) = train_text_fasttext(v, task.all_samples, labels)

get_all_samples(v::FastTextClassifier) = get_all_samples(v.prov)


function train_text_fasttext(cls::FastTextClassifier, all_gsms, train_labels)
  println("using fasttext")
  model_fn = get_FT_model(cls, train_labels)

  strings = get_strings_for_samples(cls.prov, all_gsms)
  pred_dict = mktemp() do path, io
    for s in strings
      write(io, s * "\n")
    end
    predictions = readstring(`../fastText-0.1.0/fasttext predict-prob $(model_fn).bin $path 25`)
    rm(model_fn * ".bin"; force=true)
    result = Dict()
    for (i, line) in enumerate(split(predictions, "\n"))
      if line == ""
        continue
      end
      gsm = all_gsms[i]
      preds = split(line)
      N = ceil(Int64, length(preds)/2)
      curpred = Tuple{Int64, Float32}[]
      for i=1:N
        bto = parse(Int64, preds[2*i-1][length("__label__BTO:")+1:end])
        prob = parse(Float32, preds[2*i])
        push!(curpred, (bto, prob))
      end
      result[gsm] = curpred
    end

    result
  end
  return pred_dict, 0
end


function get_label_counts(train_labels)
  all_labels = String[]
  for (gsm, labels) in train_labels
    append!(all_labels, labels)
  end

  countmap(all_labels)
end

function prepare_training_data(cls::FastTextClassifier, train_labels)
  result = String[]

  all_gsms = collect(keys(train_labels))
  println("training on ", length(all_gsms), " samples")
  all_strings = get_strings_for_samples(cls.prov, all_gsms)
  label_counts = get_label_counts(train_labels)
  @showprogress for (i, gsm) in enumerate(all_gsms)
    labels = train_labels[gsm]
    labels_str = map(x->"__label__"*x, labels)
    str = join(labels_str, " ") * " " * all_strings[i] * "\n"
    push!(result, str)
    # to balanace the dataset:
    if cls.balance == 0
      continue
    end

    for label in labels
      want_total_add = cls.balance - label_counts[label]
      if want_total_add <= 0
        continue
      end
      add_now = round(Int32, rand()*want_total_add)
      for i = 1:add_now
        push!(result, str)
      end
    end
  end
  shuffle!(result)
  println("expanded to ", length(result), " samples")
  return result
end

function get_FT_model(cls::FastTextClassifier, train_labels)
  model_fn = mktemp() do path, io
    @show path
    write(io, prepare_training_data(cls, train_labels))
    run(`../fastText-0.1.0/fasttext supervised -input $path -output $(path)_model $(split(cls.params))`)
    path * "_model"
  end

  model_fn
end
