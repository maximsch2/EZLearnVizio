include("coretrain_keras6_preamble.jl")


PARAMS["output.file"] = "/scratch/grechkin/vizio_ezlearn_keras6_generic2_$(oname).sqlite"
const ONTOLOGY_FILE = "figure_ontology.obo"
const FigOnto = OBOParse.load(ONTOLOGY_FILE, "FIG")

const ONTOLOGY = FigOnto
const RELS = [:is_a]


include("coretrain_expr_sgd.jl")
include("coretrain_fasttext.jl")
include("vizio_data.jl")
const ExprProv, TextProv = load_vizio_data()

include("ezlearn_runner.jl")

gsms = collect(intersect(Set(get_all_samples(ExprProv)), Set(get_all_samples(TextProv))))
initial = get_initial_beliefs("/scratch/grechkin/init_vizio.sqlite", "text_1", gsms)

task = initialize_task(PARAMS, gsms, initial)
run_task(task)
