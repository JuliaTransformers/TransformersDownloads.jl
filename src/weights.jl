const PYTORCH_WEIGHTS_NAME = "pytorch_model.bin"
const PYTORCH_WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
const SAFETENSOR_WEIGHTS_NAME = "model.safetensors"
const SAFETENSOR_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"

const SAFETENSOR_EXTENSIONS = (".safetensors",)
const PICKLE_EXTENSIONS = (".bin", ".pt", ".pth")

_has_extension(name::AbstractString, extensions) =
    any(ext -> endswith(name, ext), extensions)

_weights_kind_from_name(name::AbstractString) =
    _has_extension(name, SAFETENSOR_EXTENSIONS) ? :safetensor_single :
    _has_extension(name, PICKLE_EXTENSIONS) ? :pickle_single : nothing

function _weights_kind(files::AbstractVector{<:AbstractString})
    SAFETENSOR_WEIGHTS_INDEX_NAME in files && return :safetensor_sharded
    SAFETENSOR_WEIGHTS_NAME in files && return :safetensor_single
    PYTORCH_WEIGHTS_INDEX_NAME in files && return :pickle_sharded
    PYTORCH_WEIGHTS_NAME in files && return :pickle_single
    return nothing
end

# A direct local file is classified by its extension, not by a hard-coded name.
function _weights_kind(source::PathSource, files)
    isfile(source.path) && return _weights_kind_from_name(basename(source.path))
    return _weights_kind(files)
end

_weights_kind(::RepoSource, files) = _weights_kind(files)

"""
    load_pickle_weights(repo_id; possible_files=nothing, kws...)

Load PyTorch Pickle (`.bin`) weights for `repo_id`. This is an extension point:
the method is provided by `TransformersDownloadsPickleExt` when `Pickle.jl` is
loaded. When `Pickle.jl` is not loaded, this fallback throws a
[`PickleSupportNotLoadedError`](@ref).
"""
load_pickle_weights(args...; kwargs...) = throw(PickleSupportNotLoadedError())

"""
    ShardedStateDict <: AbstractDict{String, AbstractArray}

Lazy dictionary-like view over a sharded SafeTensors checkpoint
(`model.safetensors.index.json`). Tensor bytes remain memory-mapped in the
per-shard `SafeTensor` objects; only the index (parameter name to shard) is held
in memory.

Supports the full `AbstractDict` interface (`keys`, `values`, `getindex`,
`haskey`, iteration, `length`).
"""
struct ShardedStateDict <: AbstractDict{String, AbstractArray}
    shards::Vector{Pair{String, SafeTensors.SafeTensor}}
    key_to_shard::Dict{String, Tuple{Int, String}}
end

Base.length(d::ShardedStateDict) = length(d.key_to_shard)
Base.keys(d::ShardedStateDict) = keys(d.key_to_shard)
Base.haskey(d::ShardedStateDict, key::AbstractString) = haskey(d.key_to_shard, key)

function Base.getindex(d::ShardedStateDict, key::AbstractString)
    i, local_key = d.key_to_shard[key]
    return d.shards[i][2][local_key]
end

function Base.iterate(d::ShardedStateDict, state...)
    y = iterate(d.key_to_shard, state...)
    y === nothing && return nothing
    key = y[1].first
    return (key => d[key], y[2])
end

function _load_safetensor_file(file::AbstractString)
    return SafeTensors.deserialize(file)
end

function _load_safetensor_sharded(repo_id; kws...)
    index_file = hf_file(repo_id, SAFETENSOR_WEIGHTS_INDEX_NAME; kws...)
    index_data = json_load(index_file)
    weight_map = index_data["weight_map"]

    files_to_keys = Dict{String,Vector{String}}()
    sizehint!(files_to_keys, length(weight_map) ÷ 16)
    for (key, filename) in weight_map
        push!(get!(()->String[], files_to_keys, filename), key)
    end

    shards = Vector{Pair{String,SafeTensors.SafeTensor}}(undef, length(files_to_keys))
    key_to_shard = Dict{String,Tuple{Int,String}}()
    sizehint!(key_to_shard, length(weight_map))
    for (i, (filename, keys)) in enumerate(files_to_keys)
        tensors = SafeTensors.deserialize(hf_file(repo_id, filename; kws...))
        shards[i] = filename => tensors
        for key in keys
            key_to_shard[key] = (i, key)
        end
    end
    return ShardedStateDict(shards, key_to_shard)
end

"""
    hf_weights(repo_id; kws...)

Download the weights file (SafeTensors by default) for a given repository ID.
Returns the local path. See [`hf_file`](@ref) for the accepted sources and
keyword arguments.
"""
hf_weights(repo_id; kws...) = hf_file(repo_id, SAFETENSOR_WEIGHTS_NAME; kws...)

"""
    load_weights(repo_id; possible_files=nothing, kws...) -> AbstractDict{String, <:AbstractArray}

Load the weights (state dictionary) for a given repository ID or local path.

The available format is detected automatically, prioritizing SafeTensors over
PyTorch Pickle (`.bin`), the latter requiring `Pickle.jl` to be loaded.

The returned object maps parameter names to tensor arrays and is always an
`AbstractDict{String, <:AbstractArray}`:

  * a `SafeTensors.SafeTensor` for a single `model.safetensors` file. The arrays
    are memory-mapped views, so tensor data is not copied into RAM;
  * a [`ShardedStateDict`](@ref) for sharded checkpoints
    (`model.safetensors.index.json`), a lazy view over the per-shard files;
  * a `Dict{String, Any}` for PyTorch Pickle checkpoints.

Consumers should rely on the `AbstractDict` interface rather than on the
concrete return type.
"""
function load_weights(repo_id; possible_files=nothing, kws...)
    source = classify_source(repo_id)
    files = ensure_possible_files(possible_files, source; kws...)
    kind = _weights_kind(source, files)
    isnothing(kind) && throw(WeightFormatNotFoundError(_display_name(source), files))

    if kind === :safetensor_sharded
        return _load_safetensor_sharded(source; kws...)
    elseif kind === :safetensor_single
        return _load_safetensor_file(hf_file(source, SAFETENSOR_WEIGHTS_NAME; kws...))
    end
    return load_pickle_weights(_display_name(source); possible_files=files, kws...)
end
