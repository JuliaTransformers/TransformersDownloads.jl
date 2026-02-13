using Mmap
using JSON
using HuggingFaceApi: list_model_files

"""
    mmap_open(file, fsz=filesize(file))

Memory-map a file as a `Vector{UInt8}`.
"""
function mmap_open(file, fsz=filesize(file))
    return open(f -> Mmap.mmap(f, Vector{UInt8}, (Int(fsz),)), file, "r")
end

"""
    json_load(file)

Load and parse a JSON file into a Julia dictionary.
"""
function json_load(file)
    return JSON.parse(read(file, String))
end

_ensure(f::Function, A, args...; kwargs...) = isnothing(A) ? f(args...; kwargs...) : A

"""
    ensure_possible_files(possible_files, model_name; revision=nothing, auth_token=HuggingFaceApi.get_token(), kw...)

Ensure a list of available files in the repo/directory. If `model_name` is a local directory,
it returns its contents. Otherwise, it searches the Hugging Face Hub.
"""
function ensure_possible_files(possible_files, model_name; revision=nothing, auth_token=HuggingFaceApi.get_token(), kw...)
    isdir(model_name) && return readdir(model_name)
    return _ensure(list_model_files, possible_files, model_name; revision, token=auth_token)
end
