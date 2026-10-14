# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-09

First release. `TransformersDownloads.jl` downloads and loads Transformer model
assets (config, weights, tokenizers) from the Hugging Face Hub or from local
paths.

### Added

- `hf_file`, `hf_config`, `hf_weights`, `hf_tokenizer`, `hf_vocab`, `hf_merges`
  to resolve (and download/cache) asset paths.
- `load_config`, `load_weights`, `load_tokenizer`, `load_vocab`, `load_merges`
  to load assets into Julia objects.
- SafeTensors support, including sharded checkpoints via
  `model.safetensors.index.json`, with memory-mapped, zero-copy tensor views.
- `TransformersDownloads.ShardedStateDict`, a lazy `AbstractDict` view over
  sharded SafeTensors checkpoints.
- Optional PyTorch Pickle (`.bin`) support via the `Pickle.jl` weak dependency
  and the `TransformersDownloadsPickleExt` extension.
- `TransformersDownloads.AbstractSource`, `RepoSource` and `PathSource` for
  explicit local-vs-remote source handling.
- Exception types (`PathNotFoundError`, `LocalFileNotFoundError`,
  `WeightFormatNotFoundError`, `VocabFileNotFoundError`,
  `UnsupportedVocabFormatError`, `PickleSupportNotLoadedError`) rooted at
  `TransformersDownloadsError`.
- Extension points `TransformersDownloads.vocab_format`,
  `TransformersDownloads.load_vocab_file` and
  `TransformersDownloads.load_pickle_weights`.

### Changed

- `load_weights` now returns memory-mapped `SafeTensor`/`ShardedStateDict`
  values instead of eagerly copying into a `Dict{String, Any}`. The result is
  always an `AbstractDict{String, <:AbstractArray}`.
- A local path is only interpreted as a Hugging Face repo id when it does not
  look like a path; strings starting with `.`, `~` or `/` that do not exist now
  throw `PathNotFoundError` instead of producing a Hub 404.
- Direct local files are recognized by extension (`.safetensors`, `.bin`,
  `.pt`, `.pth`) rather than by hard-coded file names.
- `hf_file` on a local directory now checks that the requested file exists.
- Unknown keyword arguments passed to the file-listing step now raise an
  `ArgumentError` instead of being silently ignored.
- `julia` compatibility raised to `1.10` (required for package extensions).
- Removed the unused `DataStructures` and `Mmap` dependencies.

### Fixed

- Loading weights/vocabularies from a local file with a non-standard name now
  works, as documented.
- Missing local paths no longer fall through to the Hugging Face Hub.
