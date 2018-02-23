using Memoize

const vizio_data_h5 = h5open("data/vizio_data.h5", "r")
const vizio_data = readmmap(vizio_data_h5["data"])
const vizio_h5_samples = read(vizio_data_h5["samples"]);
const h5_samples_dict = Dict([(sample, i) for (i, sample) in enumerate(vizio_h5_samples)]);
const h5_samples_dict_upcase = Dict([(uppercase(sample), i) for (i, sample) in enumerate(vizio_h5_samples)]);

const vizio_captions_h5 = h5open("data/vizio_captions.h5", "r")
const vizio_all_captions = read(vizio_captions_h5["captions"]);

get_caption(sample_id) = vizio_all_captions[h5_samples_dict_upcase[sample_id]]

function load_vizio_data(start_i,batch_size)
    h5_samples_dict_mini = Dict([(sample, i) for (i, sample) in enumerate(vizio_h5_samples[start_i:start_i+batch_size])]);
    h5_samples_dict_upcase_mini = Dict([(uppercase(sample), i) for (i, sample) in enumerate(vizio_h5_samples[start_i:start_i+batch_size])]);
    VecDataProvider(vizio_data[:,start_i:start_i+batch_size], h5_samples_dict_mini), TextDataProvider(vizio_all_captions[start_i:start_i+batch_size], vizio_h5_samples[start_i:start_i+batch_size], h5_samples_dict_upcase_mini)
end



@memoize function allnames(term::Term)
  result = String[]
  push!(result, lowercase(term.name))
  for s in term.synonyms
    s = s[2:end]
    i = findfirst(s, '"')
    push!(result, lowercase(s[1:i-1]))
  end
  result
end

function construct_initial_labels(all_samples, all_terms)
  result = Dict{String, Vector{Tuple{String, Float64}}}()
  @showprogress "Computing initial labels..." for sample in all_samples
  caption = lowercase(get_caption(uppercase(sample)))
    res = []
    for term in all_terms
      anames = allnames(term)::Vector{String}
      for name in anames
        if contains(caption, name)
          push!(res, (term.id, 1.0))
          break
        end
      end
    end
    if length(res) > 0
      result[sample] = res
    end
  end
  BeliefDict(result)
end
