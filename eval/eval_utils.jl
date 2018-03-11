using SQLite, OBOParse

function read_sample(db, tblname, sample_name)
    query = "select term_name as term_id, prob from $tblname, samples, terms" *
            " where $(tblname).sample_id = samples.sample_id and $(tblname).term_id = terms.term_id and sample_name='$(sample_name)'";
    rows = SQLite.query(db, query; nullable=false);
    probs = rows[:prob]
    terms = rows[:term_id]
    collect(zip(terms, probs))
end

function load_samples(db, tblname, samples)
    result = Dict()
    for sample in samples
        result[sample] = read_sample(db, tblname, sample)
    end
    result
end


function get_term(tid, ontology)
    if haskey(ontology.terms, tid)
        return ontology[tid]
    end
    return gettermbyid(ontology, parse(Int64, tid))
  end
  

function convert_ground_truth(belief, ontology)
    result = Dict()
    for (key, val) in belief
        terms = map(tid_prob -> get_term(tid_prob[1], ontology), val)
        result[key] = terms
    end
    return result
end

# %%

upstream(term::Term, ontology, root) = union([term], Vector{Term}(setdiff(ancestors(ontology, term, symbols), root)))
upstream(terms::Vector{Term}, ontology, root) = union([upstream(t, ontology, root) for t in terms]...)


function score_prec_recall_generic(correct, pred)
    n_correct_predictions = Int[]
    n_predictions = Int[]

    n_total_terms = Int[]
    for gsm in keys(correct)
        correct_terms = correct[gsm]
        push!(n_total_terms, length(correct_terms))
        if !haskey(pred, gsm)
            push!(n_predictions, 0)
            push!(n_correct_predictions, 0)
            continue
        end
        terms = pred[gsm]

        if length(terms) > 0
            push!(n_predictions, length(terms))
            push!(n_correct_predictions, length(intersect(terms, correct_terms)))
        end
    end


    sn = sum(n_predictions)
    if sn > 0
        precision = sum(n_correct_predictions)/sn
    else
        precision = 1.0
    end

    recall = sum(n_correct_predictions)/sum(n_total_terms)

    precision, recall
end

function score_prec_recall_thresh_upstream(correct, pred, thresh, ontology, root)
    pred_transformed = Dict{String, Set{Term}}()
    correct_gsms = Set(keys(correct))
    for (gsm, preds) in pred
        if !(gsm in correct_gsms)
            continue
        end

        term_ids = String[x[1] for x in preds if x[2] > thresh]
        terms = Term[get_term(id, ontology) for id in term_ids]
        pred_transformed[gsm] = Set(upstream(terms, ontology, root))
    end

    score_prec_recall_generic(correct, pred_transformed)
end


function pr2auc(PR)
    recall, precision = PR
    order = sortperm(recall)
    recall = recall[order]
    precision = precision[order]
    area = 0.0
    for i=2:length(recall)
        dx = recall[i] - recall[i-1]
        val = (precision[i] + precision[i-1])/2
        area += dx * val
    end
    area
end

function get_PR_upstream(correct, preds, bounds, ontology, root)
    P = Float64[]
    R = Float64[]
    for t in bounds
        (prec, rec) = score_prec_recall_thresh_upstream(correct, preds, t, ontology, root)
        push!(P, prec)
        push!(R, rec)
    end
    push!(P, 0)
    push!(R, 1)
    order = sortperm(R)

    R, P = R[order], P[order]

    P[1] = P[2]

    R, P
end
