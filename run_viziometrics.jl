using OBOParse, EZLearn

import EZLearn.train_and_predict
import EZLearn.construct_labels

abstract TextView <: EZLearn.ClassifierView
abstract ExprView <: EZLearn.ClassifierView

const PARAMS = Dict()

PARAMS["output.file"] = "data/vizio_ezlearn_keras6_generic.sqlite"
const ONTOLOGY_FILE = "figure_ontology.obo"
const FigOnto = OBOParse.load(ONTOLOGY_FILE, "FIG")

const ONTOLOGY = FigOnto
const RELS = [:is_a]

include("coretrain_expr_sgd.jl")
include("coretrain_fasttext.jl")
include("vizio_data.jl")
# Can also train on part of the data. Vizio data size (1174456))
ExprProv, TextProv = load_vizio_data(1,1174456)

include("ezlearn_runner.jl")

all_sample_ids = collect(intersect(Set(get_all_samples(ExprProv)), Set(get_all_samples(TextProv))))

# Uncomment the following to run on a subsample:
# all_sample_ids = all_sample_ids[1:5000]

#initial = get_initial_beliefs("data/init_vizio.sqlite", "text_1", all_sample_ids)
initial, precomputed = construct_initial_labels(all_sample_ids, OBOParse.allterms(ONTOLOGY))

save("data/precomp.jld","data",precomputed)
# precomputed = load("data/precomp.jld")["data"]

filtered_pc = (pc for pc in keys(precomputed) if length(precomputed[pc])==1)
filtered_precompute = Dict{String,Array{String,1}}()
for pc in filtered_pc
    filtered_precompute[pc] = [precomputed[pc][1][1]]
end
save("data/precomp_final.jld","data",filtered_precompute)

task = initialize_task(PARAMS, all_sample_ids, initial)
run_task(task)
