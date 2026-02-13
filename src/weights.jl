
const PYTORCH_WEIGHTS_NAME = "pytorch_model.bin"
const PYTORCH_WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
const SAFETENSOR_WEIGHTS_NAME = "model.safetensors"
const SAFETENSOR_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"

# Extension point for Pickle weights.
# This is populated by TransformersDownloadsPickleExt when Pickle.jl is loaded.
function load_pickle_weights end

"""
    hgf_model_weights(model_name; kws...)

Download the weights file (SafeTensors by default) for a given model.
"""
hgf_model_weights(model_name; kws...) = hgf_file(model_name, SAFETENSOR_WEIGHTS_NAME; kws...)

"""
    load_state_dict(model_name; possible_files=nothing, kws...)

Load the weights (state dictionary) for a given model.

It automatically detects the available format, prioritizing SafeTensors.
If only PyTorch Pickle format (`.bin`) is available, it requires `Pickle.jl`
to be loaded to enable the extension.
"""
function load_state_dict(model_name; possible_files=nothing, kws...)
    possible_files = ensure_possible_files(possible_files, model_name; kws...)

    # 1. Check for SafeTensors (Preferred)
    if SAFETENSOR_WEIGHTS_INDEX_NAME in possible_files
        return load_safetensor_sharded(model_name; kws...)
    elseif SAFETENSOR_WEIGHTS_NAME in possible_files
        return load_safetensor_single(model_name; kws...)
    end

    # 2. Check for PyTorch Pickle (Legacy)
    if PYTORCH_WEIGHTS_INDEX_NAME in possible_files || PYTORCH_WEIGHTS_NAME in possible_files
        if isempty(methods(load_pickle_weights))
            error("Weights are in PyTorch Pickle format, but `Pickle.jl` is not loaded. Please `using Pickle` to enable support.")
        end
        return load_pickle_weights(model_name; possible_files, kws...)
    end

    error("No supported weight format found for $model_name")
end

function load_safetensor_single(model_name; kws...)
    file = hgf_file(model_name, SAFETENSOR_WEIGHTS_NAME; kws...)
    return load_safetensor_file(file)
end

function load_safetensor_sharded(model_name; kws...)
    index_file = hgf_file(model_name, SAFETENSOR_WEIGHTS_INDEX_NAME; kws...)
    index_data = json_load(index_file)
    weight_map = index_data["weight_map"]

    # Group by filename to avoid redownloading shards multiple times
    files_to_keys = Dict{String,Vector{String}}()
    for (key, filename) in weight_map
        push!(get!(files_to_keys, filename, []), key)
    end

    state_dict = Dict{String,Any}()
    for (filename, keys) in files_to_keys
        file_path = hgf_file(model_name, filename; kws...)
        tensors = SafeTensors.deserialize(file_path)
        for k in keys
            state_dict[k] = tensors[k]
        end
    end
    return state_dict
end

function load_safetensor_file(file)
    tensors = SafeTensors.deserialize(file)
    return Dict{String,Any}(k => v for (k, v) in tensors)
end
