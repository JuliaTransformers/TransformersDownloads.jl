# Ecosystem Guide

This guide is for authors of downstream model packages (BERT.jl,
DistilBERT.jl, ...) that build on `TransformersDownloads.jl`.

## The weights contract

`load_weights` always returns an `AbstractDict{String, <:AbstractArray}`
mapping parameter names (as found in the checkpoint) to tensor arrays. The
concrete type depends on the checkpoint layout:

| Checkpoint | Return type | Tensor storage |
|---|---|---|
| `model.safetensors` | `SafeTensors.SafeTensor` | memory-mapped, zero-copy |
| `model.safetensors.index.json` (sharded) | `TransformersDownloads.ShardedStateDict` | memory-mapped, lazy |
| `pytorch_model.bin` (+ `Pickle.jl`) | `Dict{String, Any}` | copied into RAM |

Consume it through the `AbstractDict` interface only:

```julia
state_dict = load_weights(repo_id)
for (name, tensor) in state_dict
    # `tensor` is an AbstractArray with the checkpoint's dimensions
end
```

Do **not** assume `Dict{String, Any}`; in particular, avoid `convert`-ing the
result to a `Dict`, which would copy every tensor and defeat memory mapping.

## Mapping weights into a model

A typical downstream loader reads the configuration to build the model, then
assigns tensors by name:

```julia
using TransformersDownloads

config = load_config(repo_id)
state_dict = load_weights(repo_id)

hidden_size = config["hidden_size"]
# build model from `config` ...
# then, for each parameter `p`:
#     copyto!(p, state_dict[name])
```

Because single-file and sharded SafeTensors return memory-mapped views, the
first access to a tensor may page it in from disk; copying into your own
parameter arrays is the intended use.

## Sources and local paths

`TransformersDownloads` classifies the `repo_id` argument into an
`AbstractSource`:

- `PathSource` for existing files and directories, and for strings that look
  like paths (`/...`, `./...`, `~/...`) — a missing path throws
  `PathNotFoundError` instead of producing a Hub 404;
- `RepoSource` for everything else, treated as a Hugging Face repository id.

Prefix relative paths with `./` to force local treatment.

## Error taxonomy

All exceptions subtype `TransformersDownloads.TransformersDownloadsError`:

| Exception | Meaning |
|---|---|
| `PathNotFoundError` | a path-like string does not exist locally |
| `LocalFileNotFoundError` | a directory does not contain the requested file |
| `WeightFormatNotFoundError` | no supported weights found |
| `VocabFileNotFoundError` | neither `vocab.txt` nor `vocab.json` found |
| `UnsupportedVocabFormatError` | a vocabulary file exists but its format is unknown |
| `PickleSupportNotLoadedError` | `.bin` weights requested without `Pickle.jl` |

Catch concrete subtypes to implement fallbacks (for example, initialize random
weights when `WeightFormatNotFoundError` is thrown).

## Extension points

### New vocabulary formats

To support a vocabulary format not built in (for example SentencePiece),
define a `vocab_format` method for your own string type and a matching
`load_vocab_file` method:

```julia
struct SentencePieceName <: AbstractString
    name::String
end
Base.String(x::SentencePieceName) = x.name

TransformersDownloads.vocab_format(name::SentencePieceName) =
    endswith(String(name), "spiece.model") ? :sentencepiece : nothing

TransformersDownloads.load_vocab_file(path::AbstractString, ::Val{:sentencepiece}) =
    # parse the sentencepiece model
    ...
```

### Custom weight formats

`TransformersDownloads.load_pickle_weights` is the extension point used by the
`Pickle.jl` extension. A downstream package can provide an analogous loader
for another format by defining a method for it (and routing to it from its own
`load_weights` wrapper).

### Explicit sources

`TransformersDownloads.RepoSource` and `TransformersDownloads.PathSource` can be
constructed explicitly and passed to the file-listing utilities when
classification heuristics are not desired.
