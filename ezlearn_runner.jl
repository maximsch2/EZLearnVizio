const DEFAULT_PARAMS = Dict(
    "intersection.threshold" => 0.3,
    "ezlearn.iterations" => 5
)

function parget(params, key)
    if !haskey(params, key)
        default = DEFAULT_PARAMS[key]
        warn("Missing $key parameter, using default: $default")
        return default
    end
    return params[key]
end



threshold_beliefs2(beliefs, threshold) = EZLearn.threshold_beliefs_nonred(ONTOLOGY, beliefs, threshold, RELS)

remove_redundant_beliefs(beliefs) = EZLearn.remove_redundant_beliefs(ONTOLOGY, beliefs, RELS)

const OntologyIntersector = EZLearn.ontology_intersector(ONTOLOGY, RELS);


produce_text_beliefs(beliefs, thresh, initial) = intersect_labels_core(threshold_beliefs2(beliefs["expr"][end], thresh),
                                                              initial,
                                                      OntologyIntersector; append_left=true, append_right=true)


produce_expr_beliefs(beliefs, thresh, initial) = threshold_beliefs2(beliefs["text"][end], thresh)

import EZLearn.construct_labels
import EZLearn.train_and_predict

getthresh(task) = parget(task.params, "intersection.threshold")

construct_labels(v::TextView, beliefs, task) = produce_text_beliefs(beliefs, getthresh(task), task.cache["initial_expr_thresholded"])
construct_labels(v::ExprView, beliefs, task) = produce_expr_beliefs(beliefs, getthresh(task), task.cache["initial_expr_thresholded"])



textView = FastTextClassifier("-lr 1.0 -epoch 25", TextProv)
# Other options:
#  textView = FastTextClassifier("-lr 1.0 -epoch 25", TextProv; balance=25)
#  textView = FastTextClassifier("-lr 1.0 -epoch 25", TextProv; balance=100)
#  textView = FastTextClassifier("-lr 0.8 -epoch 100", TextProv)
#  textView = FastTextClassifier("-lr 1.0 -epoch 25 -minn 3 -maxn 6", TextProv)
#  textView = FastTextClassifier("-lr 1.0 -epoch 25 -wordNgrams 2 -minCount 5", TextProv)
#  textView = FastTextClassifier("-lr 1.0 -epoch epoch 25 -wordNgrams 3 -minCount 5", TextProv)

exprView = SGDExpressionClassifier(ExprProv)

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
