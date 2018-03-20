using OBOParse, EZLearn, Memoize

import EZLearn.train_and_predict
import EZLearn.construct_labels

abstract TextView <: EZLearn.ClassifierView
abstract ExprView <: EZLearn.ClassifierView

const PARAMS = Dict()

PARAMS["output.file"] = "data/arxiv_ezlearn_keras6_generic.sqlite"
const ONTOLOGY_FILE = "figure_ontology.obo"
const FigOnto = OBOParse.load(ONTOLOGY_FILE, "FIG")

const ONTOLOGY = FigOnto
const RELS = [:is_a,:develops_from,:part_of,:related_to]

include("coretrain_expr_sgd.jl")
include("coretrain_fasttext.jl")
include("arxiv_data.jl")

const ExprProv, TextProv = load_subset_data(1,10000)

include("ezlearn_runner.jl")

all_sample_ids = collect(intersect(Set(get_all_samples(ExprProv)), Set(get_all_samples(TextProv))))

# Uncomment the following to run on a subsample:
# all_sample_ids = all_sample_ids[1:5000]

initial = construct_initial_labels(all_sample_ids, OBOParse.allterms(ONTOLOGY))

task = initialize_task(PARAMS, all_sample_ids, initial)
run_task(task)
