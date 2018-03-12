using JSON 

# %%
const symbols=[:is_a]
include("eval_utils.jl")

const FigOnto = OBOParse.load("figure_ontology.obo", "FIG")
const ROOT = [gettermbyname(FigOnto, "Figure")]

# %%

if !isfile("vizio_precomputed.sqlite")
  error("eval file not found")
end
evalDB = SQLite.DB("vizio_precomputed.sqlite")

img_locs = SQLite.query(evalDB, "select sample from ground_truth_v")[:sample];

#dev_img_locs = img_locs_ordered[1:250];
#writecsv("dev_img_locs.csv", dev_img_locs)
const dev_img_locs = Vector{String}(readcsv("dev_img_locs.csv")[:, 1]);
const test_img_locs = collect(setdiff(img_locs, dev_img_locs));


const ground_truth = load_samples(evalDB, "ground_truth", test_img_locs)
# %%
const test_eval = convert_ground_truth(ground_truth, FigOnto)
@show length(test_eval)


function get_PRs_dict(eval_set, fn; iters=1:5)
  if !isfile(fn)
    error("file not found: $fn")
  end
  resultsDB = SQLite.DB(fn)
  result = Dict()
  img_locs = collect(keys(eval_set))
  result["EZLearn_image"] = []
  result["EZLearn_text"] = []
  expr_PRs = []
  for i=iters
    try
      expr_dict = load_samples(resultsDB, "expr_$i", img_locs);
      text_dict = load_samples(resultsDB, "text_$i", img_locs);
      push!(result["EZLearn_image"], get_PR_upstream(eval_set, expr_dict, linspace(0, 1), FigOnto, ROOT))
      push!(result["EZLearn_text"], get_PR_upstream(eval_set, text_dict, linspace(0, 1), FigOnto, ROOT))
    catch
    end
  end

  result
end



resultJSON = Dict() 

resultJSON["EZLearn"] = get_PRs_dict(test_eval, ARGS[1]; iters=[5])

const vizio_predictions = load_samples(evalDB, "viziometrics", test_img_locs)

resultJSON["Viziometrics"] = get_PR_upstream(test_eval, vizio_predictions, linspace(0, 1), FigOnto, ROOT);

# %%
write("figure_data.json", JSON.json(resultJSON, 2));
