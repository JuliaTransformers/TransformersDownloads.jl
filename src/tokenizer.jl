const TOKENIZER_NAME = "tokenizer.json"
const VOCAB_TXT_NAME = "vocab.txt"
const VOCAB_JSON_NAME = "vocab.json"
const MERGES_NAME = "merges.txt"

"""
    hf_tokenizer(repo_id; kws...)

Download the `tokenizer.json` for a given repository ID. Returns the local path.
"""
hf_tokenizer(repo_id; kws...) = hf_file(repo_id, TOKENIZER_NAME; kws...)

"""
    hf_vocab(repo_id; possible_files=nothing, kws...)

Find and download the vocabulary file for a given repository ID.
It prioritizes `vocab.txt` (WordPiece) and then `vocab.json` (BPE).
"""
function hf_vocab(repo_id; possible_files=nothing, kws...)
    possible_files = ensure_possible_files(possible_files, repo_id; kws...)

    if VOCAB_TXT_NAME in possible_files
        return hf_file(repo_id, VOCAB_TXT_NAME; kws...)
    elseif VOCAB_JSON_NAME in possible_files
        return hf_file(repo_id, VOCAB_JSON_NAME; kws...)
    end

    error("No vocabulary file (vocab.txt or vocab.json) found for $repo_id")
end

"""
    hf_merges(repo_id; kws...)

Download the `merges.txt` for a given repository ID. Returns the local path.
"""
hf_merges(repo_id; kws...) = hf_file(repo_id, MERGES_NAME; kws...)

"""
    load_tokenizer(repo_id; kws...)

Download and parse the `tokenizer.json` for a given repository ID. Returns a dictionary.
"""
load_tokenizer(repo_id; kws...) = json_load(hf_tokenizer(repo_id; kws...))

"""
    load_vocab(repo_id; possible_files=nothing, kws...)

Download and load the vocabulary for a given repository ID.
Returns a `Vector{String}` for `.txt` files or a `Dict{String, Any}` for `.json` files.
"""
function load_vocab(repo_id; possible_files=nothing, kws...)
    possible_files = ensure_possible_files(possible_files, repo_id; kws...)

    if VOCAB_TXT_NAME in possible_files
        path = hf_file(repo_id, VOCAB_TXT_NAME; kws...)
        return readlines(path)
    elseif VOCAB_JSON_NAME in possible_files
        path = hf_file(repo_id, VOCAB_JSON_NAME; kws...)
        return json_load(path)
    end

    error("No vocabulary file (vocab.txt or vocab.json) found for $repo_id")
end

"""
    load_merges(repo_id; kws...)

Download and read the `merges.txt` file for a given repository ID.
Returns the lines of the file as as `Vector{String}`.
"""
load_merges(repo_id; kws...) = readlines(hf_merges(repo_id; kws...))
