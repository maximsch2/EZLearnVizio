using JSON 

# %%
const symbols=[:is_a,:develops_from,:part_of,:related_to]
include("eval_utils.jl")

const Brenda = OBOParse.load("brenda_ontology.obo", "BTO")
const ROOT = union([gettermbyname(Brenda, "whole body")], ancestors(Brenda, gettermbyname(Brenda, "whole body"), symbols))

# %%

if !isfile("geo_precomputed.sqlite")
  error("eval file not found")
end
evalDB = SQLite.DB("geo_precomputed.sqlite")

const test_samples = SQLite.query(evalDB, "select sample from torrente_manual_labels_v where sample not in (select sample from lee_manual_labels_v)")[:sample].values;

const ground_truth = load_samples(evalDB, "torrente_manual_labels", test_samples)
# %%
const test_eval = convert_ground_truth(ground_truth, Brenda)
@show length(test_eval)


function get_PRs_dict(eval_set, fn; iters=1:5)
  if !isfile(fn)
    error("file not found: $fn")
  end
  resultsDB = SQLite.DB(fn)
  result = Dict()
  samples = collect(keys(eval_set))
  result["EZLearn_expr"] = []
  result["EZLearn_text"] = []
  expr_PRs = []
  for i=iters
    try
      expr_dict = load_samples(resultsDB, "expr_$i", samples);
      text_dict = load_samples(resultsDB, "text_$i", samples);
      push!(result["EZLearn_expr"], get_PR_upstream(eval_set, expr_dict, linspace(0, 1), Brenda, ROOT))
      push!(result["EZLearn_text"], get_PR_upstream(eval_set, text_dict, linspace(0, 1), Brenda, ROOT))
    catch
    end
  end

  result
end



resultJSON = Dict() 

resultJSON["EZLearn"] = get_PRs_dict(test_eval, ARGS[1]; iters=[5])

const ursa_predictions = load_samples(evalDB, "ursa_predictions", test_samples)

resultJSON["URSA"] = get_PR_upstream(test_eval, ursa_predictions, linspace(0, 1), Brenda, ROOT);
@show pr2auc(resultJSON["URSA"])
@show pr2auc.(resultJSON["EZLearn"]["EZLearn_expr"])
# %%
write("geo_data.json", JSON.json(resultJSON, 2));
