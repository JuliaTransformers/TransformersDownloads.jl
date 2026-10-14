# TransformersDownloads.jl

`TransformersDownloads.jl` downloads model files from the
[Hugging Face Hub](https://huggingface.co) or reads them from local paths, and
returns them as Julia data: `config.json`, SafeTensors and PyTorch weight
files, `tokenizer.json`, vocabularies and `merges.txt`.

The package contains no model or tokenization code; model architectures live in
packages such as DistilBERT.jl, and tokenization algorithms are planned for a
separate `TransformersTokenizers.jl` package.

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

# Tiny model keeps the examples light; swap in any Hugging Face repository id.
model_id = "hf-internal-testing/tiny-random-bert"

config    = load_config(model_id)
weights   = load_weights(model_id)
vocab     = load_vocab(model_id)
tokenizer = load_tokenizer(model_id)
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
