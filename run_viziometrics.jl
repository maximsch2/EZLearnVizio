using OBOParse, EZLearn

import EZLearn.train_and_predict
import EZLearn.construct_labels

abstract TextView <: EZLearn.ClassifierView
abstract ExprView <: EZLearn.ClassifierView

const PARAMS = Dict(
    "intersection.threshold" => 0.3,
    "text.intersect" => "append_both"
)


PARAMS["output.file"] = "/scratch/grechkin/vizio_ezlearn_keras6_generic.sqlite"
const ONTOLOGY_FILE = "figure_ontology.obo"
const FigOnto = OBOParse.load(ONTOLOGY_FILE, "FIG")

const ONTOLOGY = FigOnto
const RELS = [:is_a]

include("coretrain_expr_sgd.jl")
include("coretrain_fasttext.jl")
include("vizio_data.jl")
const ExprProv, TextProv = load_vizio_data()

include("ezlearn_runner.jl")

all_sample_ids = collect(intersect(Set(get_all_samples(ExprProv)), Set(get_all_samples(TextProv))))

# Uncomment the following to run on a subsample:
all_sample_ids = all_sample_ids[1:5000]

#initial = get_initial_beliefs("/scratch/grechkin/init_vizio.sqlite", "text_1", all_sample_ids)
initial = construct_initial_labels(all_sample_ids, OBOParse.allterms(FigOnto))

task = initialize_task(PARAMS, all_sample_ids, initial)
run_task(task)
