using OBOParse, EZLearn

import EZLearn.train_and_predict
import EZLearn.construct_labels

abstract TextView <: EZLearn.ClassifierView
abstract ExprView <: EZLearn.ClassifierView

const PARAMS = Dict(
    "intersection.threshold" => 0.3,
    "initial.text_subsample" => 1,
    "expr.sgd" => true,
    "expr.method" => "new2_newauto_val5",
    "text.valsplit" => 0.05,
    "text.intersect" => "append_both",
    "text.method" => "ft"
)

@show PARAMS


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

samples = collect(intersect(Set(get_all_samples(ExprProv)), Set(get_all_samples(TextProv))))
# Uncomment the following to run on a subsample:
# samples = samples[1:10000]

initial = get_initial_beliefs("/scratch/grechkin/init_vizio.sqlite", "text_1", samples)

task = initialize_task(PARAMS, samples, initial)
run_task(task)
