"""
    hf_file_url(repo_id, filename; revision="main")

Construct a `HuggingFaceURL` for a given repository ID and file name.
"""
@inline hf_file_url(repo_id, filename; revision="main") =
    HuggingFaceURL(repo_id, filename; repo_type=nothing, revision=something(revision, "main"))

function _hf_download(repo_id, filename; revision="main", local_files_only::Bool=false,
                      cache::Bool=true, auth_token=get_token())
    return hf_hub_download(repo_id, filename;
        repo_type=nothing, revision=something(revision, "main"),
        auth_token, local_files_only, cache)
end

"""
    hf_file(repo_id, filename; revision="main", kws...) -> String

Return the local path of `filename` inside `repo_id`.

`repo_id` may be

  * a Hugging Face repository id such as `"google-bert/bert-base-uncased"`, in
    which case the file is downloaded (and cached) on demand;
  * a path to a local directory containing `filename`; or
  * a path to a local file, in which case that file is returned directly,
    regardless of `filename` (useful for files with non-standard names).

A string that looks like a path but does not exist throws a
[`PathNotFoundError`](@ref) instead of being sent to the Hub as a repository id.

Supported keyword arguments are `revision`, `auth_token`, `cache` and
`local_files_only` (the latter two are only meaningful for remote files).
"""
hf_file(repo_id::AbstractString, filename; kws...) =
    hf_file(classify_source(repo_id), filename; kws...)

function hf_file(source::RepoSource, filename; revision="main", kws...)
    return _hf_download(source.repo_id, filename; revision, kws...)
end

function hf_file(source::PathSource, filename; kws...)
    path = source.path
    isfile(path) && return path
    full = joinpath(path, filename)
    isfile(full) || throw(LocalFileNotFoundError(path, filename))
    return full
end
