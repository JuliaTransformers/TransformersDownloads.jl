const TOKENIZER_NAME = "tokenizer.json"
const VOCAB_TXT_NAME = "vocab.txt"
const VOCAB_JSON_NAME = "vocab.json"
const MERGES_NAME = "merges.txt"

"""
    hf_tokenizer(repo_id; kws...)

Download the `tokenizer.json` for a given repository ID or resolve it from a
local path. Returns the local path. See [`hf_file`](@ref) for the accepted
sources and keyword arguments.
"""
hf_tokenizer(repo_id; kws...) = hf_file(repo_id, TOKENIZER_NAME; kws...)

"""
    vocab_format(name::AbstractString) -> Union{Nothing, Symbol}

Determine the vocabulary format from a file name. Returns `:txt` for `.txt`
files, `:json` for `.json` files, or `nothing` when the format is unknown.

This is an extension point. To support a new vocabulary format, define a
`vocab_format` method for your own string type (avoiding type piracy on
`String`) and a matching [`load_vocab_file`](@ref) method.
"""
function vocab_format(name::AbstractString)
    endswith(name, ".txt") && return :txt
    endswith(name, ".json") && return :json
    return nothing
end

"""
    load_vocab_file(path, ::Val{format})

Load a vocabulary file whose format was determined by [`vocab_format`](@ref).
The default implementation returns a `Vector{String}` for `:txt` and a
dictionary for `:json`.
"""
load_vocab_file(path::AbstractString, ::Val{:txt}) = readlines(path)
load_vocab_file(path::AbstractString, ::Val{:json}) = json_load(path)

function _vocab_name(source::AbstractSource, files)
    if source isa PathSource && isfile(source.path)
        return basename(source.path)
    end
    VOCAB_TXT_NAME in files && return VOCAB_TXT_NAME
    VOCAB_JSON_NAME in files && return VOCAB_JSON_NAME
    throw(VocabFileNotFoundError(_display_name(source), files))
end

"""
    hf_vocab(repo_id; possible_files=nothing, kws...)

Find and download the vocabulary file for a given repository ID or local path.
It prioritizes `vocab.txt` (WordPiece) and then `vocab.json` (BPE). Returns the
local path. See [`hf_file`](@ref) for the accepted sources and keyword
arguments.
"""
function hf_vocab(repo_id; possible_files=nothing, kws...)
    source = classify_source(repo_id)
    files = ensure_possible_files(possible_files, source; kws...)
    name = _vocab_name(source, files)
    return hf_file(source, name; kws...)
end

"""
    hf_merges(repo_id; kws...)

Download the `merges.txt` for a given repository ID or resolve it from a local
path. Returns the local path. See [`hf_file`](@ref) for the accepted sources and
keyword arguments.
"""
hf_merges(repo_id; kws...) = hf_file(repo_id, MERGES_NAME; kws...)

"""
    load_tokenizer(repo_id; kws...)

Download and parse the `tokenizer.json` for a given repository ID or local path.
Returns a dictionary.
"""
load_tokenizer(repo_id; kws...) = json_load(hf_tokenizer(repo_id; kws...))

"""
    load_vocab(repo_id; possible_files=nothing, kws...)

Download and load the vocabulary for a given repository ID or local path.
Returns a `Vector{String}` for `.txt` vocabularies or a dictionary for `.json`
vocabularies.
"""
function load_vocab(repo_id; possible_files=nothing, kws...)
    source = classify_source(repo_id)
    files = ensure_possible_files(possible_files, source; kws...)
    name = _vocab_name(source, files)
    format = vocab_format(name)
    isnothing(format) && throw(UnsupportedVocabFormatError(name))
    path = hf_file(source, name; kws...)
    return load_vocab_file(path, Val(format))
end

"""
    load_merges(repo_id; kws...)

Download and read the `merges.txt` file for a given repository ID or local path.
Returns the lines of the file as a `Vector{String}`.
"""
load_merges(repo_id; kws...) = readlines(hf_merges(repo_id; kws...))
