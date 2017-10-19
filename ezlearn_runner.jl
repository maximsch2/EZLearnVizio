const DEFAULT_PARAMS = Dict(
    "intersection.threshold" => 0.3,
    "initial.text_subsample" => 1,
    "ezlearn.iterations" => 5,
    "text.lambda1" => 1e-3,
    "text.lambda2" => 1e-3,
    "text.method" => "ontology",
    "expr.ontoreg" => 0.0,
    "text.ontoreg" => 0.0,
    "text.valsplit" => 0.2
)

function parget(params, key)
    if !haskey(params, key)
        default = DEFAULT_PARAMS[key]
        warn("Missing $key parameter, using default: $default")
        return default
    end
    return params[key]
end


#include("EZLearnBase.jl")
# using EZLearn
# import EZLearn.train_and_predict
# import EZLearn.construct_labels
#
# abstract TextView <: EZLearn.ClassifierView
# abstract ExprView <: EZLearn.ClassifierView
#
# include("coretrain_fasttext.jl")
# include("coretrain_expr_sgd.jl")
#
threshold_beliefs2(beliefs, threshold) = EZLearn.threshold_beliefs_nonred(ONTOLOGY, beliefs, threshold, RELS)

remove_redundant_beliefs(beliefs) = EZLearn.remove_redundant_beliefs(ONTOLOGY, beliefs, RELS)

const OntologyIntersector = EZLearn.ontology_intersector(ONTOLOGY, RELS);

TEXT_INTERSECT_METHOD=get(PARAMS, "text.intersect", "legacy")

@show TEXT_INTERSECT_METHOD

if TEXT_INTERSECT_METHOD == "legacy"
  produce_text_beliefs(beliefs, thresh, initial) = intersect_labels_core(threshold_beliefs2(beliefs["expr"][end], thresh),
                                                              initial,
                                                      OntologyIntersector)
elseif TEXT_INTERSECT_METHOD == "no"
  produce_text_beliefs(beliefs, thresh, initial) = threshold_beliefs2(beliefs["expr"][end], thresh)
elseif TEXT_INTERSECT_METHOD == "keep_expr"
  produce_text_beliefs(beliefs, thresh, initial) = intersect_labels_core(threshold_beliefs2(beliefs["expr"][end], thresh),
                                                              initial,
                                                      OntologyIntersector; keep_left=true)
elseif TEXT_INTERSECT_METHOD == "specific_expr"
  produce_text_beliefs(beliefs, thresh, initial) = remove_redundant_beliefs(intersect_labels_core(threshold_beliefs2(beliefs["expr"][end], thresh),
                                                              initial,
                                                      OntologyIntersector; keep_left=true))
elseif TEXT_INTERSECT_METHOD == "default_expr"
  produce_text_beliefs(beliefs, thresh, initial) = intersect_labels_core(threshold_beliefs2(beliefs["expr"][end], thresh),
                                                              initial,
                                                      OntologyIntersector; default_left=true)
elseif TEXT_INTERSECT_METHOD == "append_expr"
  produce_text_beliefs(beliefs, thresh, initial) = intersect_labels_core(threshold_beliefs2(beliefs["expr"][end], thresh),
                                                              initial,
                                                      OntologyIntersector; append_left=true)

elseif TEXT_INTERSECT_METHOD == "append_both"
  produce_text_beliefs(beliefs, thresh, initial) = intersect_labels_core(threshold_beliefs2(beliefs["expr"][end], thresh),
                                                              initial,
                                                      OntologyIntersector; append_left=true, append_right=true)
elseif TEXT_INTERSECT_METHOD == "append_expr2"
  produce_text_beliefs(beliefs, thresh, initial) = remove_redundant_beliefs(intersect_labels_core(threshold_beliefs2(beliefs["expr"][end], thresh),
                                                              initial,
                                                      OntologyIntersector; append_left=true))
else
  error("unknown text intersect method: $(TEXT_INTERSECT_METHOD)")
end


EXPR_INTERSECTION_METHOD=get(PARAMS, "expr.intersect", "legacy")
@show EXPR_INTERSECTION_METHOD

if EXPR_INTERSECTION_METHOD == "legacy"
  produce_expr_beliefs(beliefs, thresh, initial) = threshold_beliefs2(beliefs["text"][end], thresh)
elseif EXPR_INTERSECTION_METHOD == "append"
  produce_expr_beliefs(beliefs, thresh, initial) = intersect_labels_core(threshold_beliefs2(beliefs["text"][end], thresh),
                                                                threshold_beliefs2(beliefs["text"][1], thresh),
                                                                OntologyIntersector; append_left=true)
elseif EXPR_INTERSECTION_METHOD == "specific"
  produce_expr_beliefs(beliefs, thresh, initial) = remove_redundant_beliefs(intersect_labels_core(threshold_beliefs2(beliefs["text"][end], thresh),
                                                                threshold_beliefs2(beliefs["text"][1], thresh),
                                                                OntologyIntersector; keep_left=true))

elseif EXPR_INTERSECTION_METHOD == "append_both"
  produce_expr_beliefs(beliefs, thresh, initial) = intersect_labels_core(threshold_beliefs2(beliefs["text"][end], thresh),
                                                                initial,
                                                                OntologyIntersector; append_left=true, append_right=true)
else
  error("unknown expr intersection method: $(EXPR_INTERSECTION_METHOD)")
end


import EZLearn.construct_labels
import EZLearn.train_and_predict

getthresh(task) = parget(task.params, "intersection.threshold")

construct_labels(v::TextView, beliefs, task) = produce_text_beliefs(beliefs, getthresh(task), task.cache["initial_expr_thresholded"])
construct_labels(v::ExprView, beliefs, task) = produce_expr_beliefs(beliefs, getthresh(task), task.cache["initial_expr_thresholded"])

TEXT_METHOD = get(PARAMS, "text.method", "ft")
@show TEXT_METHOD


if TEXT_METHOD == "ft"
  textView = FastTextClassifier("-lr 1.0 -epoch 25", TextProv)
elseif TEXT_METHOD == "ftbal"
  textView = FastTextClassifier("-lr 1.0 -epoch 25", TextProv; balance=25)
elseif TEXT_METHOD == "ftbal2"
  textView = FastTextClassifier("-lr 1.0 -epoch 25", TextProv; balance=100)
elseif TEXT_METHOD == "ft1"
  textView = FastTextClassifier("-lr 0.8 -epoch 100", TextProv)
elseif TEXT_METHOD == "ft_char"
  textView = FastTextClassifier("-lr 1.0 -epoch 25 -minn 3 -maxn 6", TextProv)
elseif TEXT_METHOD == "ft2"
  textView = FastTextClassifier("-lr 1.0 -epoch 25 -wordNgrams 2 -minCount 5", TextProv)
elseif TEXT_METHOD == "ft3"
  textView = FastTextClassifier("-lr 1.0 -epoch epoch 25 -wordNgrams 3 -minCount 5", TextProv)
else
  error("unknown text method: $(TEXT_METHOD)")
end

EXPR_METHOD = get(PARAMS, "expr.method", "new2_newauto_val5")
@show EXPR_METHOD
if EXPR_METHOD == "new2_newauto_val5"
  exprView = SGDExpressionClassifier(ExprProv)
elseif EXPR_METHOD == "new2_newauto_val5_best5"
  exprView = SGDExpressionClassifier(ExprProv; bestof=5)
elseif EXPR_METHOD == "new2_newauto_val5_best5_nr"
  exprView = SGDExpressionClassifier(ExprProv; bestof=5, reseed=false)
elseif EXPR_METHOD == "new3"
  exprView = SGDExpressionClassifier(ExprProv; bestof=3, reseed=false, optimizer="adam", droplr=true, batch_size=32, L2Reg=5e-5)
elseif EXPR_METHOD == "new2_newauto_va10_best10"
  exprView = SGDExpressionClassifier(ExprProv; val_split=0.1, bestof=10)
elseif EXPR_METHOD == "new2_newauto2_val5"
  exprView = SGDExpressionClassifier(ExprProv)
else
  error("unknown expr method: $(EXPR_METHOD)")
end


function get_partial_beliefs(all_gsms)
  all_gsms_set = Set(all_gsms)
  jf = jldopen("../data/all_distant_preds.jld", "r")
  gsms = read(jf["gsms"])
  all_distant_preds = read(jf["all_distant_preds"])
  result = Dict()
  for (i, gsm) in enumerate(gsms)
    if !(gsm in all_gsms_set)
      continue
    end

    preds = map(x->parse(Int64, x[6:end]), all_distant_preds[i])
    if length(preds) == 0
      continue
    end
    result[gsm] = collect(zip(preds, ones(length(preds))))
  end
  return result
end


function get_initial_beliefs(dbfn, table, gsms)
  db = SQLite.DB(dbfn)
  beliefSQLite = BeliefSQLite(db, table)
  result = Dict()
  @showprogress for sample in gsms
    result[sample] = get_beliefs(beliefSQLite, sample)
  end
  SQLite._close(db)
  return BeliefDict(result)
end

function initialize_task(params, all_samples, initialText, train_text_first=false)
    if isfile(params["output.file"])
        error("output file already exists")
    end

    task = EZLearnTask([textView, exprView],
                    Dict(["text" => Any[], "expr" => Any[]]),
                    Dict(["text" => Any[], "expr" => Any[]]), ONTOLOGY, params, all_samples, Dict())


    push!(task.beliefs["text"], initialText)

    init_for_expr = threshold_beliefs2(initialText, getthresh(task))
    task.cache["initial_expr_thresholded"] = init_for_expr
    @show length(init_for_expr)

    if train_text_first
      initialText, textModel = EZLearn.train_and_predict(textView, init_for_expr, task)
      push!(task.models["text"], textModel)
      push!(task.beliefs["text"], BeliefDict(initialText))
      init_for_expr = EZLearn.construct_labels(exprView, task.beliefs, task)
    end

    initialExpr, exprModel = EZLearn.train_and_predict(exprView, init_for_expr, task)

    push!(task.beliefs["expr"], BeliefDict(initialExpr))
    push!(task.models["expr"], exprModel)
    store_beliefs(task, task.params["output.file"])

    task
end

function run_task(task)
    for i=1:parget(task.params, "ezlearn.iterations")
        @show i

        ezlearn_step(task)

        store_beliefs(task, task.params["output.file"])
        models = task.models
        outfn = task.params["output.file"] * "_models.jld"
        @save outfn models
    end
end
