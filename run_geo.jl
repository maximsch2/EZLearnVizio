using OBOParse, EZLearn

import EZLearn.train_and_predict
import EZLearn.construct_labels

abstract TextView <: EZLearn.ClassifierView
abstract ExprView <: EZLearn.ClassifierView

const PARAMS = Dict()

PARAMS["output.file"] = "data/geo_ezlearn_keras6_generic.sqlite"
const ONTOLOGY_FILE = "data/BrendaTissueOBO"
const BrendaOntology = OBOParse.load(ONTOLOGY_FILE, "BTO")

const ONTOLOGY = BrendaOntology
const RELS = [:is_a,:develops_from,:part_of,:related_to]

include("coretrain_expr_sgd.jl")
include("coretrain_fasttext.jl")


const ExprProv = load_vec_data("data/autoencoded_new.h5", "all_samples", "all_data_relu_3layer")
const TextProv = load_strings_data("data/geo_raw_strings.h5")


include("ezlearn_runner.jl")

all_sample_ids = collect(intersect(Set(get_all_samples(ExprProv)), Set(get_all_samples(TextProv))))

# Uncomment the following to run on a subsample:
all_sample_ids = all_sample_ids[1:5000]

#initial = construct_initial_labels(all_sample_ids, OBOParse.allterms(ONTOLOGY))
initial = get_initial_beliefs("data/geo_initial.sqlite", "init_lvl3", all_sample_ids)

task = initialize_task(PARAMS, all_sample_ids, initial)
run_task(task)
