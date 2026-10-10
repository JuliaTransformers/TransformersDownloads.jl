# TransformersDownloads.jl

`TransformersDownloads.jl` downloads and loads Transformer model assets
(configurations, weights and tokenizers) from the
[Hugging Face Hub](https://huggingface.co) or from local paths. It is the
asset-loading substrate of the Julia Transformers ecosystem (DistilBERT.jl and
friends).

## Features

- Hugging Face Hub integration: download and cache files by repository id.
- SafeTensors first: memory-mapped, zero-copy weights, including sharded
  checkpoints via a lazy `ShardedStateDict`.
- Legacy support: PyTorch Pickle (`.bin`) weights through the optional
  `Pickle.jl` extension.
- Local paths: directories or specific files, with clear errors when something
  is missing.
- Extensibility: `vocab_format`, `load_vocab_file` and `load_pickle_weights`
  are documented extension points.

## Installation

Requires Julia 1.10 or later (package extensions).

```julia
using Pkg
Pkg.add("TransformersDownloads")
```

## Quick start

```julia
using TransformersDownloads

config    = load_config("google-bert/bert-base-uncased")
weights   = load_weights("google-bert/bert-base-uncased")
vocab     = load_vocab("google-bert/bert-base-uncased")
tokenizer = load_tokenizer("google-bert/bert-base-uncased")
```

Every `load_*` function accepts a local directory or a direct file path in place
of a repository id:

```julia
config  = load_config("./my_model")
weights = load_weights("/path/to/custom_weights.safetensors")
```

## What you can load

| Function | Returns |
|---|---|
| `hf_file(repo_id, filename)` | local path, downloading and caching as needed |
| `hf_config`, `hf_weights`, `hf_tokenizer`, `hf_vocab`, `hf_merges` | local path of the matching asset |
| `load_config` | `config.json` as an `AbstractDict` |
| `load_weights` | state dict as `AbstractDict{String,<:AbstractArray}` |
| `load_tokenizer` | `tokenizer.json` as an `AbstractDict` |
| `load_vocab` | `Vector{String}` (`.txt`) or `AbstractDict` (`.json`) |
| `load_merges` | `merges.txt` as a `Vector{String}` |

`load_weights` returns a `SafeTensors.SafeTensor` (single file) or a
`ShardedStateDict` (sharded checkpoint), both memory-mapped, or a
`Dict{String, Any}` for PyTorch Pickle checkpoints.

See the [Ecosystem Guide](@ref) for the weights contract, the error taxonomy and
the extension points.

## API reference

```@autodocs
Modules = [TransformersDownloads]
```
