module TransformersDownloadsPickleExt

using Pickle
using TransformersDownloads
using TransformersDownloads: ensure_possible_files, hf_file, json_load,
    WeightFormatNotFoundError, PYTORCH_WEIGHTS_NAME, PYTORCH_WEIGHTS_INDEX_NAME

function TransformersDownloads.load_pickle_weights(repo_id; possible_files=nothing, kws...)
    files = ensure_possible_files(possible_files, repo_id; kws...)
    if isfile(repo_id)
        return _load_pickle_file(repo_id)
    elseif PYTORCH_WEIGHTS_INDEX_NAME in files
        return _load_pickle_sharded(repo_id; kws...)
    elseif PYTORCH_WEIGHTS_NAME in files
        return _load_pickle_file(hf_file(repo_id, PYTORCH_WEIGHTS_NAME; kws...))
    end
    throw(WeightFormatNotFoundError(repo_id, files))
end

_load_pickle_file(file) = Pickle.Torch.THload(file)

function _load_pickle_sharded(repo_id; kws...)
    index_file = hf_file(repo_id, PYTORCH_WEIGHTS_INDEX_NAME; kws...)
    index_data = json_load(index_file)
    weight_map = index_data["weight_map"]

    files_to_keys = Dict{String,Vector{String}}()
    sizehint!(files_to_keys, length(weight_map) ÷ 16)
    for (key, filename) in weight_map
        push!(get!(()->String[], files_to_keys, filename), key)
    end

    state_dict = Dict{String,Any}()
    sizehint!(state_dict, length(weight_map))
    for (filename, keys) in files_to_keys
        tensors = _load_pickle_file(hf_file(repo_id, filename; kws...))
        for key in keys
            haskey(tensors, key) && (state_dict[key] = tensors[key])
        end
    end
    return state_dict
end

end
