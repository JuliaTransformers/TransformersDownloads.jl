"""
    AbstractSource

Supertype describing where an asset comes from: a Hugging Face Hub repository
([`RepoSource`](@ref)) or a local file/directory ([`PathSource`](@ref)).

The public API accepts a plain string for `repo_id`; the string is classified
once with [`classify_source`](@ref) and all internal logic dispatches on the
resulting source type. Advanced users can construct a source explicitly and pass
it to `load_*` functions to bypass classification.
"""
abstract type AbstractSource end

"""
    RepoSource(repo_id)

An asset living in the Hugging Face Hub repository `repo_id`
(e.g. `"google-bert/bert-base-uncased"`).
"""
struct RepoSource <: AbstractSource
    repo_id::String
end

"""
    PathSource(path)

An asset living at the local filesystem path `path` (a file or a directory).
"""
struct PathSource <: AbstractSource
    path::String
end

_display_name(source::RepoSource) = source.repo_id
_display_name(source::PathSource) = source.path

"""
    is_path_like(x::AbstractString) -> Bool

Return `true` when `x` unambiguously denotes a local path: an absolute path, a
path starting with `.` or `~`, or a Windows drive path. Hugging Face repository
ids never look like this.
"""
function is_path_like(x::AbstractString)
    isempty(x) && return false
    isabspath(x) && return true
    first(x) in ('.', '~') && return true
    return occursin(r"^[A-Za-z]:[\\/]", x)
end

"""
    classify_source(x::AbstractString) -> AbstractSource

Classify `x` as a local [`PathSource`] or a remote [`RepoSource`]:

  * an existing file or directory is a `PathSource`;
  * a string that looks like a path (`/...`, `./...`, `~/...`) but does not
    exist throws a [`PathNotFoundError`](@ref) instead of being sent to the Hub;
  * anything else is treated as a Hugging Face repository id.

To force a relative path that does not start with `.` or `~` to be treated as a
local path, prefix it with `./`.
"""
function classify_source(x::AbstractString)
    path = startswith(x, '~') ? expanduser(x) : String(x)
    isfile(path) && return PathSource(path)
    isdir(path) && return PathSource(path)
    is_path_like(x) && throw(PathNotFoundError(String(x)))
    return RepoSource(String(x))
end

classify_source(source::AbstractSource) = source
