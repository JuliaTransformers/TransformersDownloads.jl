"""
    TransformersDownloadsError

Abstract supertype of all exceptions thrown by `TransformersDownloads`.
Downstream packages can catch this to handle any failure of this library, or
catch a concrete subtype to react to a specific failure mode.
"""
abstract type TransformersDownloadsError <: Exception end

"""
    PathNotFoundError(path)

Thrown when a string was classified as a local path (absolute, or starting with
`.` or `~`) but no file or directory exists at that location.
"""
struct PathNotFoundError <: TransformersDownloadsError
    path::String
end

"""
    LocalFileNotFoundError(dir, filename)

Thrown when `filename` is requested from an existing local directory `dir` that
does not contain it.
"""
struct LocalFileNotFoundError <: TransformersDownloadsError
    dir::String
    filename::String
end

"""
    WeightFormatNotFoundError(source, files)

Thrown by [`load_weights`](@ref) when no supported weight file is found in
`source`. `files` lists the file names that are present.
"""
struct WeightFormatNotFoundError <: TransformersDownloadsError
    source::String
    files::AbstractVector{<:AbstractString}
end

"""
    VocabFileNotFoundError(source, files)

Thrown by [`load_vocab`](@ref)/[`hf_vocab`](@ref) when neither `vocab.txt` nor
`vocab.json` is found in `source`. `files` lists the file names that are present.
"""
struct VocabFileNotFoundError <: TransformersDownloadsError
    source::String
    files::AbstractVector{<:AbstractString}
end

"""
    UnsupportedVocabFormatError(name)

Thrown when a vocabulary file was found but its format is unknown. Extend
[`TransformersDownloads.vocab_format`](@ref) and
[`TransformersDownloads.load_vocab_file`](@ref) to support a new format.
"""
struct UnsupportedVocabFormatError <: TransformersDownloadsError
    name::String
end

"""
    PickleSupportNotLoadedError()

Thrown when PyTorch Pickle (`.bin`) weights are requested but `Pickle.jl` is not
loaded. Run `using Pickle` (or add `Pickle` to the active project) and retry.
"""
struct PickleSupportNotLoadedError <: TransformersDownloadsError end

Base.showerror(io::IO, e::PathNotFoundError) = print(io,
    "local path `$(e.path)` does not exist. If you meant a Hugging Face repo id, ",
    "it must not start with `.`, `~` or `/`; if you meant a local path, check the spelling.")

Base.showerror(io::IO, e::LocalFileNotFoundError) = print(io,
    "`$(e.filename)` was not found in local directory `$(e.dir)`.")

function Base.showerror(io::IO, e::WeightFormatNotFoundError)
    print(io, "no supported weight file (model.safetensors, pytorch_model.bin or a ",
              "sharded index) found in `$(e.source)`.")
    isempty(e.files) || print(io, " Files present: ", join(e.files, ", "), ".")
    return nothing
end

function Base.showerror(io::IO, e::VocabFileNotFoundError)
    print(io, "no vocabulary file (vocab.txt or vocab.json) found in `$(e.source)`.")
    isempty(e.files) || print(io, " Files present: ", join(e.files, ", "), ".")
    return nothing
end

Base.showerror(io::IO, e::UnsupportedVocabFormatError) = print(io,
    "could not determine the format of vocabulary file `$(e.name)`. Define a ",
    "`TransformersDownloads.vocab_format` method for this name and a matching ",
    "`TransformersDownloads.load_vocab_file` method to add support.")

Base.showerror(io::IO, ::PickleSupportNotLoadedError) = print(io,
    "PyTorch Pickle (`.bin`) weights require `Pickle.jl`, which is not loaded.\n",
    "Run `using Pickle` (or add `Pickle` to your project) and retry.")
