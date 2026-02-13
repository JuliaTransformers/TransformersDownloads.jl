module TransformersDownloadsPickleExt

using TransformersDownloads
using Pickle
using HuggingFaceApi

const PYTORCH_WEIGHTS_NAME = "pytorch_model.bin"
const PYTORCH_WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"

"""
    TransformersDownloads.load_pickle_weights(model_name; possible_files=nothing, kws...)

Implementation of `load_pickle_weights` for PyTorch `.bin` files via `Pickle.jl`.
"""
function TransformersDownloads.load_pickle_weights(model_name; possible_files=nothing, kws...)
    possible_files = TransformersDownloads.ensure_possible_files(possible_files, model_name; kws...)

    if PYTORCH_WEIGHTS_INDEX_NAME in possible_files
        return load_pickle_sharded(model_name; kws...)
    elseif PYTORCH_WEIGHTS_NAME in possible_files
        return load_pickle_single(model_name; kws...)
    else
        error("Pickle weights not found in $model_name")
    end
end

"""
    load_pickle_single(model_name; kws...)

Load a single PyTorch `.bin` file using `Pickle.Torch.THload`.
"""
function load_pickle_single(model_name; kws...)
    file = TransformersDownloads.hgf_file(model_name, PYTORCH_WEIGHTS_NAME; kws...)
    return Pickle.Torch.THload(file)
end

"""
    load_pickle_sharded(model_name; kws...)

Load sharded PyTorch `.bin` files based on the index map.
"""
function load_pickle_sharded(model_name; kws...)
    index_file = TransformersDownloads.hgf_file(model_name, PYTORCH_WEIGHTS_INDEX_NAME; kws...)
    index_data = TransformersDownloads.json_load(index_file)
    weight_map = index_data["weight_map"]

    files_to_keys = Dict{String,Vector{String}}()
    for (key, filename) in weight_map
        push!(get!(files_to_keys, filename, []), key)
    end

    state_dict = Dict{String,Any}()
    for (filename, keys) in files_to_keys
        try
            file_path = TransformersDownloads.hgf_file(model_name, filename; kws...)
            tensors = Pickle.Torch.THload(file_path)
            for k in keys
                if haskey(tensors, k)
                    state_dict[k] = tensors[k]
                end
            end
        catch e
            @warn "Failed to load shard $filename: $e"
        end
    end
    return state_dict
end

end
