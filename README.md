# TransformersDownloads.jl

[![Docs stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaTransformers.github.io/TransformersDownloads.jl/stable)
[![Docs dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaTransformers.github.io/TransformersDownloads.jl/dev)
[![Build Status](https://github.com/JuliaTransformers/TransformersDownloads.jl/workflows/CI/badge.svg)](https://github.com/JuliaTransformers/TransformersDownloads.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![Coverage](https://codecov.io/gh/JuliaTransformers/TransformersDownloads.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaTransformers/TransformersDownloads.jl)

`TransformersDownloads.jl` downloads and loads Transformer model assets
(configurations, weights and tokenizers) from the [Hugging Face Hub](https://huggingface.co)
or from local paths. It is the asset-loading substrate of the Julia Transformers
ecosystem (BERT.jl, DistilBERT.jl, ...).

## Features

- **Hugging Face Hub integration**: download files, configurations, weights and
  tokenizers from repository ids.
- **SafeTensors first**: memory-mapped, zero-copy weights, including sharded
  checkpoints (`model.safetensors.index.json`).
- **Legacy support**: load PyTorch Pickle (`.bin`) weights via the optional
  `Pickle.jl` extension.
- **Local paths**: point at a local directory or a specific file, with clear
  errors when something is missing.
- **Extensible**: `vocab_format`, `load_vocab_file` and `load_pickle_weights`
  are documented extension points.

## Installation

```julia
using Pkg
Pkg.add("TransformersDownloads")
```

## Quick start

```julia
using TransformersDownloads

config = load_config("google-bert/bert-base-uncased")     # AbstractDict
weights = load_weights("google-bert/bert-base-uncased")   # AbstractDict{String,<:AbstractArray}
vocab = load_vocab("google-bert/bert-base-uncased")       # Vector{String}
tokenizer = load_tokenizer("google-bert/bert-base-uncased") # AbstractDict
```

### Loading weights

`load_weights` detects the available format automatically, preferring
SafeTensors over PyTorch Pickle. The returned object is always an
`AbstractDict{String, <:AbstractArray}`:

- a `SafeTensors.SafeTensor` for a single `model.safetensors` file (arrays are
  memory-mapped views, no tensor data is copied into RAM);
- a `TransformersDownloads.ShardedStateDict` for sharded checkpoints, a lazy
  view over the per-shard files;
- a `Dict{String, Any}` for PyTorch Pickle checkpoints.

To load legacy `.bin` weights, load `Pickle.jl` to enable the extension:

```julia
using TransformersDownloads
using Pickle

weights = load_weights("some-legacy-model")
```

### Local paths

Every `load_*` function accepts a local directory or a direct file path in place
of a repository id:

```julia
config = load_config("./my_model")                 # directory with config.json
weights = load_weights("./my_model")               # directory with model.safetensors
weights = load_weights("/path/to/custom_weights.safetensors")  # direct file
vocab = load_vocab("/path/to/custom_vocab.txt")    # direct file, any name
```

Classification rules:

- an existing file or directory is used locally;
- a string starting with `/`, `.` or `~` that does not exist throws a
  `TransformersDownloads.PathNotFoundError` instead of being sent to the Hub;
- anything else is treated as a Hugging Face repository id. Prefix relative
  paths with `./` to force local treatment.

### Error handling

All exceptions thrown by the package subtype
`TransformersDownloads.TransformersDownloadsError`, so downstream packages can
react to specific failures:

```julia
using TransformersDownloads: TransformersDownloadsError, WeightFormatNotFoundError

try
    load_weights("./my_model")
catch err
    if err isa WeightFormatNotFoundError
        @warn "no weights found; initializing randomly" err.files
    else
        rethrow()
    end
end
```

## Ecosystem

`TransformersDownloads.jl` is designed to be consumed by model packages:

```julia
using TransformersDownloads

state_dict = load_weights(repo_id)   # AbstractDict{String,<:AbstractArray}
for (name, tensor) in state_dict
    # `tensor` is an AbstractArray; copy into your model's parameters
end
```

See the [Ecosystem Guide](https://JuliaTransformers.github.io/TransformersDownloads.jl/dev/guide/)
for the API contract and extension points.

## License

MIT License
