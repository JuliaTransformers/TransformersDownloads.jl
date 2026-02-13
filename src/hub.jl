using HuggingFaceApi

"""
    hf_file_url(repo_id, filename; revision="main")

Construct a `HuggingFaceURL` for a given repository ID and file name.
"""
hf_file_url(repo_id, filename; revision="main") =
    HuggingFaceURL(repo_id, filename; repo_type=nothing, revision=something(revision, "main"))

function _hf_download(
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
    hf_file(repo_id, filename; revision="main", kws...)

Download a specific file from a Hugging Face repository or return the local path
if `repo_id` is an existing directory.
"""
function hf_file(repo_id, filename; revision="main", kws...)
    isdir(repo_id) && return joinpath(repo_id, filename)
    return _hf_download(hf_file_url(repo_id, filename; revision); kws...)
end
