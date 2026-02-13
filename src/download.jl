using HuggingFaceApi

const CONFIG_NAME = "config.json"
const PYTORCH_WEIGHTS_NAME = "pytorch_model.bin"
const PYTORCH_WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
const SAFETENSOR_WEIGHTS_NAME = "model.safetensors"
const SAFETENSOR_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"

hgf_file_url(model_name, file_name; revision="main") =
    HuggingFaceURL(model_name, file_name; repo_type=nothing, revision=something(revision, "main"))

function _hgf_download(
    hgfurl::HuggingFaceURL; local_files_only::Bool=false, cache::Bool=true,
    auth_token=HuggingFaceApi.get_token(), kw...
)
    return hf_hub_download(
        hgfurl.repo_id, hgfurl.filename;
        repo_type=hgfurl.repo_type, revision=hgfurl.revision,
        auth_token, local_files_only, cache
    )
end

"""
    hgf_file(model_name, file_name; revision="main", kws...)

Download a specific file from a Hugging Face repository or return the local path
if `model_name` is an existing directory.
"""
function hgf_file(model_name, file_name; revision="main", kws...)
    isdir(model_name) && return joinpath(model_name, file_name)
    return _hgf_download(hgf_file_url(model_name, file_name; revision); kws...)
end

"""
    hgf_download(args...; kws...)

Alias for [`hgf_file`](@ref).
"""
hgf_download(args...; kws...) = hgf_file(args...; kws...)

"""
    hgf_model_config(model_name; kws...)

Download the `config.json` for a given model.
"""
hgf_model_config(model_name; kws...) = hgf_file(model_name, CONFIG_NAME; kws...)

"""
    load_config(model_name; kws...)

Download and parse the `config.json` for a given model.
"""
load_config(model_name; kws...) = json_load(hgf_model_config(model_name; kws...))
