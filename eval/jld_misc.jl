using JLD

type VectorOfVectorsSerializer{T}
    data::Vector{T}
    lengths::Vector{Int64}
    VectorOfVectorsSerializer{T}(val::Vector{Vector{T}}) = new(vcat(val...), Int64[length(x) for x in val])
end


JLD.writeas{T}(data::Vector{Vector{T}}) = VectorOfVectorsSerializer{T}(data)

function JLD.readas{T}(serdata::VectorOfVectorsSerializer{T})
    result = Vector{T}[]
    data = serdata.data
    idx = 1
    for len in serdata.lengths
        if len == 0
            push!(result, T[])
        else
            eidx = idx + len - 1
            push!(result, data[idx:eidx])
            idx += len
        end
    end
    result
end


type ArrayOfTuplesSeralizer{T1, T2}
    data1::Vector{T1}
    data2::Vector{T2}
end

JLD.readas{T1, T2}(serdata::ArrayOfTuplesSeralizer{T1, T2}) = collect(zip(serdata.data1, serdata.data2))
JLD.writeas{T1, T2}(data::Vector{Tuple{T1, T2}}) = ArrayOfTuplesSeralizer(T1[x[1] for x in data], T2[x[2] for x in data])
