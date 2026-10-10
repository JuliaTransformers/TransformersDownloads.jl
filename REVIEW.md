# TransformersDownloads.jl — Code Review & Architectural Audit

**Target:** v0.1.x — first release, intended as the asset-loading substrate for an ecosystem of downstream packages (DistilBERT.jl, BERT.jl, …).
**Scope:** `src/`, `ext/`, `test/`, `docs/`, `Project.toml`, CI workflows.
**Method:** Static analysis + live runtime experiments against `HuggingFaceApi` (local depot version) and `SafeTensors`.

---

## 1. Executive Architectural Summary

**Overall verdict: the core concept is right, and the package is small enough that it can be made production-grade in a single focused pass.**

What the package gets *right*, by Julia standards:

- **Package split by concern** (`hub.jl`, `config.jl`, `weights.jl`, `tokenizer.jl`) is clean and maps 1:1 to the file formats involved. This is a good foundation for the ecosystem goal.
- **Weak-dependency/extension pattern for Pickle** (`function load_pickle_weights end` + `ext/TransformersDownloadsPickleExt.jl`) is the idiomatic Julia 1.9+ way to make an optional heavy dependency. This is exactly the right architecture for keeping the core dependency-light.
- **Minimal, consistent API surface**: 10 exported names, `hf_*` = "give me the path", `load_*` = "give me the data". This mirrors `transformers` (Python) ergonomics, which downstream users will appreciate.
- **SafeTensors-first** with Mmap-backed lazy views via `SafeTensors.deserialize(file; mmap=true)` — zero-copy weights is the single most important performance property for a BERT/DistilBERT ecosystem, and it's already the default path.

What is *missing or wrong* for a 0.1 release:

1. **Multiple dispatch is essentially unused.** Every public function is a monolithic `f(repo_id::String; …)` that internally re-implements `isfile`/`isdir` duck-typing. This works, but it means type-stability issues, untestable branches, and no natural extension points. The fix is not to abandon the ergonomic String API — it is to route internal logic through small, dispatched traits (`Format`, `Source`) so downstream packages (and the Pickle extension) extend via methods, not strings.
2. **Three confirmed behavioral bugs** (verified by execution, §2.1):
   - `load_weights`/`load_vocab` on a **direct local file with a non-standard name** fails with `No supported weight format found`, directly contradicting the README ("…even if it has a non-standard name").
   - A **non-existent local path** (typo, cleaned `/tmp`, etc.) is silently treated as a repo ID and surfaces as a cryptic `request status HTTP/2 404` from the Hub instead of a local file error.
   - `load_weights(dir)` on a directory works, but only because of lucky `readdir` matching — and `hf_file` on a local directory returns a `joinpath` that is never checked for existence, so failures surface as raw `open()` errors far from the cause.
3. **Copy-paste boilerplate from Example.jl is still in the tree**: `docs/make.jl` (builds module `Example`, deploys to `JuliaLang/Example.jl`), `docs/src/index.md`, the LICENSE copyright line ("Stefan Karpinski, Iain Dunning"), README doc badges. All of these are wrong for a 0.1 release and the docs CI job will currently fail.
4. **`compat julia = "1"` is incompatible with the extension architecture.** `[weakdeps]`/`[extensions]` require Julia ≥ 1.9; declaring `julia = "1"` tells the resolver (and CI's `min` matrix entry) that Julia 1.0 is supported, where the Pickle path can never load. For an ecosystem release, pin to the current LTS.
5. **An unused dependency** (`DataStructures`) and an unused function (`mmap_open`) are dead weight. A zero-weight dependency surface is a selling point for downstream packages.
6. **The eager `Dict{String,Any}` state dict** copies every tensor out of the Mmap-backed `SafeTensor` view — the package *downloads lazily but loads eagerly*. For multi-GB sharded checkpoints this is both the biggest allocation in the codebase and a type-instability tax on every downstream consumer (`Any` values = dynamic dispatch in every BERT/DistilBERT layer loader).
7. **Tests are network-bound, unsharded, and partially dead** (`safetensors_model` defined and never used; the only weights test exercises the Pickle path). There are no tests for local paths — the exact behavior the README markets — and no tests for sharded loading, which is the production path for any real model.

**Architectural diagnosis in one sentence:** the package has an excellent *surface* design (names, split, extension) sitting on an under-typed *core*, and for 0.1 the work is to replace string duck-typing with dispatchable traits, make the hot path zero-copy, fix the three bugs, and professionalize the release plumbing (compat, docs, tests, license).

---

## 2. Key Design Pattern Enhancements

Prioritized as **High** (do before 0.1), **Medium** (0.1.x), **Low** (0.2+).

### 2.1 HIGH — Fix the local-path semantics (bug fix + design)

**Current Approach** (`src/hub.jl`, `src/utils.jl`):

```julia
function hf_file(repo_id, filename; revision="main", kws...)
    isfile(repo_id) && return repo_id
    isdir(repo_id) && return joinpath(repo_id, filename)
    return _hf_download(hf_file_url(repo_id, filename; revision); kws...)
end

function ensure_possible_files(possible_files, model_name; revision=nothing, auth_token=HuggingFaceApi.get_token(), kw...)
    isfile(model_name) && return [basename(model_name)]
    isdir(model_name) && return readdir(model_name)
    return _ensure(list_model_files, possible_files, model_name; revision, token=auth_token)
end
```

Problems (all three reproduced at runtime):

1. `load_weights("/path/to/custom.safetensors")` → `ensure_possible_files` returns `["custom.safetensors"]`, which matches none of the hard-coded names → `No supported weight format found` (README promises this works).
2. `load_weights("/path/that/does/not/exist")` → `isfile`/`isdir` both false → the string is treated as a *repo ID* → 404 from the Hub. The user can't tell the difference between "typo'd local path" and "wrong repo ID".
3. `hf_file("./model_dir", "config.json")` returns `./model_dir/config.json` unconditionally; a missing file explodes later in `JSON.parse` with an unrelated error.

**Idiomatic Production Julia** — distinguish the two *sources* with a tiny trait, exactly once:

```julia
# src/hub.jl
abstract type AbstractSource end
struct RepoSource  <: AbstractSource; repo_id::String; end
struct PathSource  <: AbstractSource; path::String; end

function classify_source(x::AbstractString)
    isfile(x) && return PathSource(x)
    isdir(x) && return PathSource(x)
    isabspath(x) && error(ArgumentError("local path `$x` does not exist. " *
        "If this was meant to be a Hugging Face repo ID, it must not start with `/`."))
    return RepoSource(x)
end

# Download from a repo, resolve from a directory, or return a file itself.
function hf_file(source::RepoSource, filename; revision="main", kws...)
    return _hf_download(hf_file_url(source.repo_id, filename; revision); kws...)
end
function hf_file(source::PathSource, filename; kws...)
    path = source.path
    isfile(path) && return path              # direct file: caller intent wins
    full = joinpath(path, filename)
    isfile(full) && return full
    error(ArgumentError("`$full` does not exist in local path `$path`"))
end
# Keep the ergonomic String entry point:
hf_file(repo_id::AbstractString, filename; kws...) = hf_file(classify_source(repo_id), filename; kws...)
```

And the format-detection half, so a direct file is accepted by extension, not by hard-coded name:

```julia
# src/weights.jl — replace hard-coded membership tests
const _SAFETENSOR_EXT = (".safetensors",)
const _PICKLE_EXT     = (".bin", ".pt", ".pth")

function _weights_kind(files::AbstractVector{<:AbstractString}, repo_id)
    # Direct-file case: trust the file extension over the filename
    (isfile(repo_id) && !isnothing(kind = _kind_from_extension(basename(repo_id)))) && return kind
    SAFETENSOR_WEIGHTS_INDEX_NAME in files && return :safetensor_sharded
    SAFETENSOR_WEIGHTS_NAME       in files && return :safetensor_single
    PYTORCH_WEIGHTS_INDEX_NAME    in files && return :pickle_sharded
    PYTORCH_WEIGHTS_NAME          in files && return :pickle_single
    return nothing
end
```

Why this matters for the ecosystem: downstream packages will call `load_weights` with a mix of repo IDs, directories, and direct paths, and the current code can't tell them apart from error messages. The trait also gives you a stable, internal vocabulary that `Base.getindex`-style extensions (custom sources, e.g. S3 buckets later) can hook into *without modifying your source*.

### 2.2 HIGH — Return zero-copy views instead of eager `Dict{String,Any}`

**Current Approach** (`src/weights.jl`):

```julia
function load_safetensor_file(file)
    tensors = SafeTensors.deserialize(file)          # Mmap-backed, lazy view
    return Dict{String,Any}(k => v for (k, v) in tensors)  # copies everything
end

function load_safetensor_sharded(repo_id; kws...)
    ...
    state_dict = Dict{String,Any}()
    for (filename, keys) in files_to_keys
        file_path = hf_file(repo_id, filename; kws...)
        tensors = SafeTensors.deserialize(file_path)
        for k in keys
            state_dict[k] = tensors[k]               # copy again
        end
    end
    return state_dict
end
```

Issues: (a) every tensor is copied out of the Mmap view — for a 400 MB–multi-GB checkpoint this is the dominant allocation; (b) `Dict{String,Any}` makes every downstream `weights["layer.0.weight"]` dynamically dispatched — the exact opposite of what BERT/DistilBERT layer loaders want; (c) the sharded loader holds the entire state dict in RAM even though the per-shard `SafeTensor` objects are lazy.

**Idiomatic Production Julia** — let the type system do the work. `SafeTensor{D} <: AbstractDict{String, AbstractArray}`, so the single-file case can return the mmapped dict *as-is* (zero copies, zero allocations), and the sharded case should be a lazy wrapper:

```julia
# src/weights.jl
function load_safetensor_file(file)
    return SafeTensors.deserialize(file)::SafeTensor  # already an AbstractDict; no copy
end

"""
    ShardedStateDict(shards::Vector{Pair{String, SafeTensor}}, weight_map)

Lazy `AbstractDict{String,AbstractArray}` view over sharded SafeTensors checkpoints.
Tensor bytes stay mmapped; only the index is in memory.
"""
struct ShardedStateDict <: AbstractDict{String, AbstractArray}
    shards     :: Vector{Pair{String, SafeTensor}}   # filename => lazy dict
    key_to_idx :: Dict{String, Tuple{Int,String}}    # tensor key => (shard index, local key)

    function ShardedStateDict(shards, key_to_idx)
        for (filename, st) in shards
            filename isa String || throw(ArgumentError("shard filenames must be String"))
        end
        return new(shards, key_to_idx)
    end
end

Base.length(d::ShardedStateDict) = length(d.key_to_idx)
Base.iterate(d::ShardedStateDict, state...) = iterate(d.key_to_idx, state...)
function Base.getindex(d::ShardedStateDict, k::String)
    i, local_k = d.key_to_idx[k]
    return d.shards[i][2][local_k]
end
Base.keys(d::ShardedStateDict) = keys(d.key_to_idx)
```

with the loader:

```julia
function load_safetensor_sharded(repo_id; kws...)
    index_data = json_load(hf_file(repo_id, SAFETENSOR_WEIGHTS_INDEX_NAME; kws...))
    weight_map = index_data["weight_map"]

    files_to_keys = Dict{String,Vector{String}}()
    sizehint!(files_to_keys, length(weight_map) ÷ 16)   # shards ≪ keys
    for (key, filename) in weight_map
        push!(get!(()->String[], files_to_keys, filename), key)
    end

    shards   = Vector{Pair{String,SafeTensor}}(undef, length(files_to_keys))
    key_to_i = Dict{String,Tuple{Int,String}}()
    sizehint!(key_to_i, length(weight_map))
    for (i, (filename, keys)) in enumerate(files_to_keys)
        tensors = SafeTensors.deserialize(hf_file(repo_id, filename; kws...))
        shards[i] = filename => tensors
        for k in keys
            key_to_i[k] = (i, k)
        end
    end
    return ShardedStateDict(shards, key_to_i)
end
```

Memory result: loading a sharded BERT checkpoint allocates only the index maps (a few hundred KB) instead of the full tensor set. Because both return types satisfy `AbstractDict{String, AbstractArray}`, downstream code like

```julia
for (name, θ) in weights
    layer[:W] .= θ   # concrete via the AbstractArray, still fast enough for the copy loop
end
```

keeps working, and `load_weights`'s return-type union (`SafeTensor`/`ShardedStateDict`/Pickle dict) is a documented, stable contract.

> Note: this *is* a return-type change, but that is precisely what 0.1.0 exists to absorb. Document it in the changelog.

### 2.3 HIGH — Make the Pickle-extension detection robust and reflection-free

**Current Approach** (`src/weights.jl`):

```julia
function load_pickle_weights end   # extension point, no methods yet

# inside load_weights:
if isempty(methods(load_pickle_weights))
    error("Weights are in PyTorch Pickle format, but `Pickle.jl` is not loaded. ...")
end
```

The `methods()` reflection works, but it is evaluated on *every* `load_weights` call (it forces a dynamic lookup through the method table), and it silently breaks if the extension ever loads after `__init__` in a way the reflection cannot see (world-age-sensitive tooling like Revise can produce confusing states).

**Idiomatic Production Julia** — a lazily-computed capability flag plus a fallback method:

```julia
# src/weights.jl
const _PICKLE_SUPPORT = Ref{Union{Nothing,Bool}}(nothing)

_has_pickle_support() = get!(_PICKLE_SUPPORT) do
    isdefined(Base, :get_extension) || return false
    ext = Base.get_extension(@__MODULE__, :TransformersDownloadsPickleExt)
    return ext !== nothing
end

# Fallback: always defined, so `load_weights` needs no branch
load_pickle_weights(args...; kwargs...) =
    error(ArgumentError(
        "Weights are in PyTorch Pickle (`.bin`) format, but `Pickle.jl` is not loaded.\n" *
        "Run `using Pickle` in your session or add Pickle to your project, then retry."))
```

The extension then simply overloads `load_pickle_weights` (as it already does), and `load_weights` calls it unconditionally — dispatch resolves to the real method when loaded, or the informative fallback when not. No `methods()` reflection, no `isempty` guard, one less branch.

### 2.4 HIGH — Correct compat, remove dead weight, fix release artifacts

| Artifact | Problem | Fix |
|---|---|---|
| `Project.toml` `compat julia = "1"` | Extensions require ≥ 1.9; CI `min` job will target 1.0 where `[weakdeps]` is unsupported | `julia = "1.10"` (current LTS; every ecosystem package should match) |
| `Project.toml` `[deps] DataStructures` | Never used anywhere in `src/` or `ext/` | Remove from deps + Manifest (smaller dependency surface = easier adoption) |
| `LICENSE` | Example.jl boilerplate: "Copyright (c) 2013-2014: Stefan Karpinski, Iain Dunning" | Replace with `Copyright (c) 2026 AbrJA` (or the org), keep MIT terms |
| `docs/make.jl`, `docs/src/index.md` | Build module `Example`, deploy to `JuliaLang/Example.jl` | Rewrite for `TransformersDownloads`; deploy repo `github.com/JuliaTransformers/TransformersDownloads.jl` |
| README doc badges | Point at `JuliaLang.github.io/TransformersDownloads.jl` | Point at `JuliaTransformers.github.io/TransformersDownloads.jl` |

### 2.5 MEDIUM — A `Format` trait for tokenizers/weights so the ecosystem can extend without editing your source

**Current Approach:** `load_vocab` hard-codes two file names (`vocab.txt`, `vocab.json`) and branches on the name inside the function. A downstream package that needs `sentencepiece.bpe.model` or `spiece.model` (very common in multilingual BERTs!) has no hook.

**Idiomatic Production Julia:**

```julia
# src/tokenizer.jl
"""
    vocab_format(::AbstractString) -> Union{Nothing, Symbol}

Determine the vocabulary format from a filename. Extendable by downstream
packages without touching TransformersDownloads source:

    TransformersDownloads.vocab_format(name::String) =
        endswith(name, "spiece.model") ? :sentencepiece : invoke(...)
"""
vocab_format(name::AbstractString) = name in (VOCAB_TXT_NAME, VOCAB_JSON_NAME) ? Symbol(name) : nothing
```

`load_vocab` then dispatches on `vocab_format` (a single method per format, not an `if/elseif` chain), and `load_weights`/`hf_vocab` grow by adding methods, not by editing the core. This is the difference between a library and a *platform* — which is the stated ecosystem goal.

### 2.6 MEDIUM — Centralize imports; kill the stray `using` statements and `_hf_download` indirection

**Current Approach:** `hub.jl` opens with `using HuggingFaceApi`, `utils.jl` with `using Mmap; using JSON; using HuggingFaceApi: list_model_files`. Included files importing dependencies is fragile (include order matters, and it hides the dependency surface from the module top). Also, `hf_file_url` + `_hf_download` is a two-layer indirection whose only effect is field-splatting.

**Idiomatic Production Julia:**

```julia
# src/TransformersDownloads.jl
module TransformersDownloads

using HuggingFaceApi
using HuggingFaceApi: hf_hub_download, list_model_files, HuggingFaceURL, get_token
using JSON
using SafeTensors
using Mmap

# src/hub.jl — collapse to one function
function _hf_download(repo_id, filename; revision="main", kws...)
    return hf_hub_download(repo_id, filename; revision=something(revision, "main"), kws...)
end
```

(`something(revision, "main")` is also worth noting: it's dead guard code given the default, but it *does* normalize an explicit `revision=nothing` — one line, keep it, add a comment.)

### 2.7 MEDIUM — Error handling: custom exception taxonomy with actionable messages

**Current Approach:** `error("No supported weight format found for $repo_id")` — three problems: (a) untyped, so downstream can't `catch` selectively; (b) no remediation guidance; (c) indistinguishable from genuine bugs.

**Idiomatic Production Julia** (minimal, non-boilerplate):

```julia
# src/errors.jl
"""
    WeightFormatNotFound <: Exception
    FileNotFoundInRepo <: Exception
    ...
"""
struct WeightFormatNotFound <: Exception; repo_id::String; files::Vector{String}; end
struct TokenizerFileNotFound <: Exception; repo_id::String; tried::Vector{String}; end

Base.showerror(io::IO, e::WeightFormatNotFound) = print(io,
    "No supported weight file (model.safetensors, pytorch_model.bin, or sharded index) " *
    "found in `$(e.repo_id)`. Files present: $(join(e.files, ", ")).")
```

Three small structs + three `showerror` methods is all this package needs — not a framework. Custom types (not just `error(String)`) are what let BERT.jl catch "weights missing" vs "download failed" differently.

### 2.8 MEDIUM — Normalize the `possible_files` API and its silent kwarg swallowing

**Current Approach:** `ensure_possible_files(possible_files, model_name; revision=nothing, auth_token=get_token(), kw...)` — the catch-all `kw...` silently *discards* unknown keywords. Consequences:

- `load_weights(repo; cache=false, local_files_only=true)` — the flags are dropped for the file-*listing* API call but honored for the download. Inconsistent and surprising.
- Typos (`revison="x"`) silently do nothing — the worst failure mode for a network library.

**Idiomatic Production Julia:**

```julia
function ensure_possible_files(possible_files, repo_id; revision=nothing, auth_token=get_token(), kws...)
    isempty(kws) || throw(ArgumentError("unsupported keyword(s): $(keys(kws))"))
    ...
end
```

and consider renaming `possible_files` → `files_in_repo` in the public docstrings (keep the kwarg name for 0.1 to avoid churn; revisit in 0.2).

### 2.9 LOW — Struct-based configuration accessors (post-0.1)

Returning `Dict{String,Any}` from `load_config` is the right call for 0.1 (it is what `transformers` returns, and it keeps you format-agnostic). But the ecosystem will repeatedly write the same 10 lines:

```julia
config["hidden_size"] isa Int || error(...)
```

Do **not** build a `Config` struct now. Instead, consider a tiny accessor layer in 0.2:

```julia
hidden_size(config::AbstractDict) = get(config, "hidden_size", nothing)
```

This preserves the Dict contract (zero risk) while giving downstream packages one canonical place for defaults/validation.

### 2.10 LOW — Docstring discipline: `_`-prefix internals, add missing docs

`load_safetensor_single`, `load_safetensor_sharded`, `load_safetensor_file`, `_hf_download`, `_ensure` are all internal implementation details but several carry public-style names and docstrings. Rename internals with `_` (except the documented extension point `load_pickle_weights`) so `names(TransformersDownloads)` self-documents the API surface. Run a `Docs.undocumented_names` check in CI (Documenter's `checkdocs`) before release.

---

## 3. Performance Optimization Blueprint

The package's hot paths are **I/O-bound**, so classic micro-optimization matters less than *allocation* and *type inference* — because these determine the quality of every downstream layer loader. Function-by-function:

### 3.1 `load_safetensor_file` (src/weights.jl:76) — **biggest win in the package**

```julia
# Current: copies every tensor out of the mmapped view
return Dict{String,Any}(k => v for (k, v) in tensors)
# Fix: return the SafeTensor itself — zero allocations, zero copies
return SafeTensors.deserialize(file)
```

Allocation eliminated: **O(total checkpoint bytes)** per call. `SafeTensor <: AbstractDict{String, AbstractArray}`, so all existing `keys`/`getindex`/`iterate` consumer code is unaffected. The `Any` value type disappears, making downstream `for (k, θ) in weights` loops infer concrete array types.

### 3.2 `load_safetensor_sharded` (src/weights.jl:54-74)

| Line | Issue | Fix |
|---|---|---|
| 62 | `push!(get!(files_to_keys, filename, []), key)` — `[]` allocates a fresh `Vector{Any}` on **every** iteration even when the key exists; it is then converted to `Vector{String}` | `push!(get!(()->String[], files_to_keys, filename), key)` — lazy default, zero garbage |
| 60,65 | No pre-sizing despite known lengths (`length(weight_map)`) | `sizehint!(files_to_keys, length(weight_map) ÷ 16)`; `sizehint!(key_to_i, length(weight_map))` |
| 65-72 | `state_dict[k] = tensors[k]` copies every tensor | Replace with the `ShardedStateDict` wrapper of §2.2 — allocation drops from O(bytes) to O(nkeys) |

### 3.3 `ensure_possible_files` / `_ensure` (src/utils.jl:23-35)

- `_ensure(f::Function, A, ...)` is fine (the closure is static, no capture — Julia specializes through it).
- `readdir(model_name)` for local dirs is fine, but note it returns *unsorted, relative* names; sort them once (`sort!`) so error messages in §2.7 are deterministic.
- Real issue: the function is called **twice** for Pickle weights (once in `load_weights`, once inside the extension when `possible_files === nothing` is re-derivable). Since `load_weights` already computed `possible_files`, have the extension *require* it (make it a positional arg in the 0.2 refactor) — one fewer Hub API round-trip per cold load.

### 3.4 `load_weights` (src/weights.jl:28-47)

- `isempty(methods(load_pickle_weights))` runs method-table reflection per call → replace with `_PICKLE_SUPPORT` flag (§2.3).
- Return-type union (`SafeTensor | ShardedStateDict | Dict`) is now *intentional and documented* — consumers should not branch on it; they should rely on the `AbstractDict` interface. Add this as an explicit contract note in the docstring so nobody "fixes" it by boxing to `Dict{String,Any}` again.

### 3.5 `hf_file` / `_hf_download` (src/hub.jl:8-33)

- With the `Source` trait (§2.1), `classify_source` is trivially inferable (`Union{RepoSource, PathSource}`), and the download branch no longer contains `isfile`/`isdir` syscall branches.
- `hf_file_url(repo_id, filename; revision="main") = HuggingFaceURL(repo_id, filename; repo_type=nothing, revision=something(revision, "main"))` — keep, it is cheap and does double duty as the public "construct a URL" utility; add `@inline` since it appears in the (relatively) hot path of `hf_file`.

### 3.6 `json_load` (src/utils.jl:19-21)

```julia
JSON.parse(read(file, String))
```

For multi-MB `tokenizer.json` files (BERT's is ~700 KB, GPT's ~2 MB), `read`+`parse` is acceptable; but `JSON.parsefile` exists for exactly this and avoids materializing the full `String` when the parser can stream — **minor** gain, only worth doing if `JSON.jl`'s `parsefile` shows a win in `@btime` on a real tokenizer file. Otherwise leave it: simplicity wins.

### 3.7 `mmap_open` (src/utils.jl:10-12) — dead code

Unused anywhere in `src/` or `ext/`. Either delete it (recommended — SafeTensors already mmaps internally) or use it in `json_load` for consistency. Dead code in a 0.1 package reads as unfinished.

### 3.8 Verification tooling (add to CI)

```julia
using JET
@report_opt load_weights("hf-internal-testing/tiny-random-bert")   # expect: no Any in the non-IO sections
@test_opt hf_file("google-bert/bert-base-uncased", "config.json")
using Aqua
Aqua.test_all(TransformersDownloads)
```

Both belong in the test suite (network-tagged) and will catch every regression of the fixes above.

---

## 4. Refactored Module Structure

The current split is already 80% right — this is an endorsement with surgical changes, not a rewrite:

```
TransformersDownloads.jl/
├── Project.toml                    # julia = "1.10"; drop DataStructures
├── LICENSE                         # correct copyright
├── README.md                       # fix badges; document the local-path contract honestly
├── REVIEW.md                       # this document
├── CHANGELOG.md                    # NEW — required for a real release
├── src/
│   ├── TransformersDownloads.jl    # module docstring, ALL `using` imports, exports, include() list
│   ├── sources.jl                  # NEW — AbstractSource, RepoSource, PathSource, classify_source (§2.1)
│   ├── hub.jl                      # hf_file_url, _hf_download, hf_file (pure I/O plumbing)
│   ├── utils.jl                    # json_load, ensure_possible_files, _ensure
│   ├── errors.jl                   # NEW — exception structs + showerror (§2.7)
│   ├── config.jl                   # hf_config, load_config
│   ├── weights.jl                  # format constants, load_pickle_weights stub, _weights_kind,
│   │                               # ShardedStateDict, load_weights, _load_safetensor_*
│   └── tokenizer.jl                # hf_tokenizer, hf_vocab, hf_merges, load_* (+ vocab_format trait §2.5)
├── ext/
│   └── TransformersDownloadsPickleExt.jl   # unchanged in role; overload load_pickle_weights only
├── test/
│   ├── runtests.jl                 # thin: includes the files below
│   ├── test_local.jl               # NEW — local-path semantics, runs OFFLINE with generated fixtures
│   ├── test_network.jl             # existing content, tagged/ENV-gated, moved here
│   ├── fixtures/                   # NEW — tiny safetensors config/vocab fixtures generated in-test
│   └── benchmarks/                 # NEW — bench/weights.jl, bench/json.jl (BenchmarkTools)
├── docs/
│   ├── make.jl                     # rewritten for TransformersDownloads
│   └── src/
│       ├── index.md                # README-derived overview + @autodocs
│       └── guide.md                # NEW — "Using with BERT.jl / DistilBERT.jl" ecosystem guide
└── .github/workflows/
    ├── ci.yml                      # drop `min`; add Julia 1.10 + 1 + pre; keep coverage
    ├── CompatHelper.yml            # present — good
    └── TagBot.yml                  # present — good
```

Key deltas from today: one new `sources.jl`, one new `errors.jl`, tests split offline/online, benchmarks added, CHANGELOG added, imports centralized. No existing public function is renamed or moved in 0.1.

---

## 5. Actionable Roadmap — 0.1.x Release Plan

### Phase 1 — Blockers (must land before tagging 0.1.0)

1. **Fix compat & deps** — `julia = "1.10"`; remove `DataStructures`; regenerate Manifest.
2. **Fix local-path semantics** (§2.1) — implement `Source` traits; make direct-file loading extension-based, not name-based; add the "path doesn't exist" `ArgumentError`.
3. **Zero-copy weights** (§2.2, §3.1-3.2) — return `SafeTensor` for single-file; add `ShardedStateDict`; update docstrings with the `AbstractDict` contract.
4. **Pickle detection** (§2.3) — `_PICKLE_SUPPORT` flag + informative fallback method.
5. **Release artifacts** (§2.4) — LICENSE, `docs/make.jl`, `docs/src/index.md`, README badges, add CHANGELOG.
6. **Error taxonomy** (§2.7) — three exception structs; update all `error()` call sites.
7. **Silent-kwarg guard** (§2.8) — reject unknown kwargs in `ensure_possible_files`.
8. **Tests: offline-first** — generate a tiny `model.safetensors` + `config.json` + `vocab.txt` in `test/fixtures/` and test: directory loading, direct-file loading (standard *and* non-standard names), non-existent-path error, sharded loading (2-shard fixture). Move existing network tests behind `haskey(ENV, "TD_TEST_HUB")` (default off in CI PRs, on for scheduled runs).

### Phase 2 — Quality gate (before 0.1.0, same PR series)

9. `Aqua.test_all` + `JET.@test_opt` on the core loaders (§3.8).
10. `Documenter` `checkdocs=:exports` so every exported name is documented; add the ecosystem guide page.
11. CI: matrix `["1.10", "1", "pre"]` on ubuntu/windows + macOS; keep codecov; make the docs job actually build (it currently would fail).
12. Benchmarks directory with `@btime load_safetensor_file(...)` on a realistic BERT checkpoint (pre-seeded into the local HF cache) — this becomes the regression guard for §3.

### Phase 3 — Release mechanics

13. Tag `v0.1.0` (TagBot will do it via GitHub releases); verify CompatHelper config.
14. **Register in General**; afterwards, freeze the *exported* API until 0.2: the extension point (`load_pickle_weights`), the `AbstractDict` return contract, and the `Source` trait are the ecosystem's three load-bearing seams — write them into the docs explicitly.

### Phase 4 — Ecosystem hardening (0.1.x patch releases)

15. Downstream smoke test: a minimal `BERT.jl` PR that does `weights = load_weights("google-bert/bert-base-uncased")` and assigns into a `Flux`/`NNlib` layer, timing the load. This will immediately expose any remaining `Any`-typed paths.
16. Add `sentencepiece`/`spiece` format hook via `vocab_format` (§2.5) on demand from downstream.
17. Consider `hf_weights(; subfolder=...)` support for gated repos (common in the ecosystem) — currently `repo_type`/`subfolder` are not plumbed through `hf_file`.

### Suggested version contract for the ecosystem

| Promise | Until |
|---|---|
| Exported names (10) and their kwarg signatures | 0.2 |
| `load_weights` returns `AbstractDict{String,<:AbstractArray}` | 0.2 |
| `load_pickle_weights` extension point | 0.2 |
| `Dict{String,Any}` from `load_config`/`load_tokenizer` | 1.0 |
| Local-path-vs-repo-ID classification rules | 1.0 |

---

*Report generated from static analysis and live runtime experiments (Julia 1.13.1) against the current tree at commit `7b00b73`.*
