# TransformersDownloads.jl

`TransformersDownloads.jl` downloads and loads Transformer model assets
(configurations, weights and tokenizers) from the
[Hugging Face Hub](https://huggingface.co) or from local paths. It is the
asset-loading substrate of the Julia Transformers ecosystem
(BERT.jl, DistilBERT.jl, ...).

## Installation

Requires Julia 1.10 or later (package extensions).

```julia
using Pkg
Pkg.add("TransformersDownloads")
```

## Quick start

```julia
using TransformersDownloads

config = load_config("google-bert/bert-base-uncased")
weights = load_weights("google-bert/bert-base-uncased")
vocab = load_vocab("google-bert/bert-base-uncased")
tokenizer = load_tokenizer("google-bert/bert-base-uncased")
```

Every `load_*` function accepts a local directory or a direct file path in place
of a repository id:

```julia
config = load_config("./my_model")
weights = load_weights("/path/to/custom_weights.safetensors")
```

See the [Ecosystem Guide](@ref) for the API contract and extension points.

## API reference

```@autodocs
Modules = [TransformersDownloads]
```
