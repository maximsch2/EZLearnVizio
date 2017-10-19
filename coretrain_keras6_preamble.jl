push!(LOAD_PATH, "../code/")

using OBOParse, EZLearn

using EZLearn
import EZLearn.train_and_predict
import EZLearn.construct_labels

abstract TextView <: EZLearn.ClassifierView
abstract ExprView <: EZLearn.ClassifierView

const PARAMS = Dict(
    "intersection.threshold" => 0.3,
    "initial.text_subsample" => 1,
    "expr.sgd" => true,
    "expr.method" => "new2_newauto_val5",
    "text.valsplit" => 0.05
)

oname = join(ARGS, "_")

PARAMS["text.intersect"] = ARGS[1]
PARAMS["text.method"] = ARGS[2]

textfirst=false
if length(ARGS) > 2
    val = ARGS[3]
    if endswith(val, "10")
        PARAMS["ezlearn.iterations"] = 10
        val=val[1:end-2]
    end
    if startswith(val, "cw")
        PARAMS["use.cw"] = true
        val=val[3:end]
    end
    if startswith(val, "old")
        PARAMS["expr.method"] = "new2_newauto_val5"
        val=val[4:end]
    end
      if startswith(val, "nr")
        PARAMS["expr.method"] = "new2_newauto_val5_best5_nr"
        val=val[3:end]
    end
    if startswith(val, "new3")
        PARAMS["expr.method"] = "new3"
        val=val[5:end]
    end

     if startswith(val, "b10")
        PARAMS["expr.method"] = "new2_newauto_val5_best10"
        val=val[4:end]
    end
   if startswith(val, "append")
      PARAMS["expr.intersect"] = "append_both"
      val = val[7:end]
    end
    if length(val) > 0
        PARAMS["global.seed"] = hash(ARGS[3]) % typemax(Int32)
    end
end
@show PARAMS
