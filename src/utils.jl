"""
    json_load(file)

Load and parse a JSON file into a Julia dictionary.
"""
json_load(file) = JSON.parse(read(file, String))

# Download-only keywords accepted by `ensure_possible_files` and ignored while
# listing files. Anything else is a user error and must not be swallowed.
const _LISTING_IGNORED_KWARGS = (:cache, :local_files_only)

function _check_listing_kwargs(kws)
    unsupported = setdiff(keys(kws), _LISTING_IGNORED_KWARGS)
    isempty(unsupported) || throw(ArgumentError(
        "unsupported keyword argument(s) $(unsupported); supported are " *
        "`revision`, `auth_token`, `cache` and `local_files_only`"))
    return nothing
end

"""
    ensure_possible_files(possible_files, repo_id; revision=nothing,
                          auth_token=HuggingFaceApi.get_token(), kws...) -> Vector{String}

Return the names of the files available in `repo_id`, which may be a Hugging
Face repository id, a local directory or a local file.

If `possible_files` is not `nothing` it is returned unchanged, allowing callers
that already know the repository contents to skip the remote listing (and its
network round trip).
"""
function ensure_possible_files(possible_files, repo_id; kws...)
    return ensure_possible_files(possible_files, classify_source(repo_id); kws...)
end

function ensure_possible_files(possible_files, source::AbstractSource;
                               revision=nothing, auth_token=get_token(), kws...)
    _check_listing_kwargs(kws)
    isnothing(possible_files) || return possible_files
    return _list_files(source; revision, auth_token)
end

_list_files(source::PathSource; kws...) =
    isfile(source.path) ? [basename(source.path)] : sort!(readdir(source.path))

_list_files(source::RepoSource; revision=nothing, auth_token=get_token()) =
    list_model_files(source.repo_id; revision, token=auth_token)
