# Dependency-free benchmark of the asset loading path.
#
#     julia --project=. benchmarks/run.jl
#
# The key property being checked is that loading SafeTensors weights does not
# allocate memory proportional to the checkpoint size: tensors stay
# memory-mapped and are only paged in on access.

using Printf
using SafeTensors
using TransformersDownloads

function make_single(dir)
    tensors = Dict("layer.$i.weight" => rand(Float32, 256, 256) for i in 1:8)
    SafeTensors.serialize(joinpath(dir, "model.safetensors"), tensors)
    write(joinpath(dir, "config.json"), """{"hidden_size":256,"num_hidden_layers":8}""")
    return sum(sizeof, values(tensors))
end

function make_sharded(dir)
    sharded = joinpath(dir, "sharded")
    mkpath(sharded)
    SafeTensors.serialize(joinpath(sharded, "model-00001-of-00002.safetensors"),
                          Dict("a.weight" => rand(Float32, 256, 256)))
    SafeTensors.serialize(joinpath(sharded, "model-00002-of-00002.safetensors"),
                          Dict("b.weight" => rand(Float32, 256, 256)))
    write(joinpath(sharded, "model.safetensors.index.json"),
          """{"metadata":{},"weight_map":{"a.weight":"model-00001-of-00002.safetensors","b.weight":"model-00002-of-00002.safetensors"}}""")
    return sharded
end

function bench(label, f)
    f()  # warmup / compile
    GC.gc()
    allocs = @allocated f()
    time = @elapsed f()
    @printf("%-28s %8.2f ms  %10.1f KiB allocated\n", label, time * 1e3, allocs / 1024)
end

function main()
    dir = mktempdir()
    checkpoint_bytes = make_single(dir)
    sharded = make_sharded(dir)

    @printf("checkpoint size: %.1f KiB\n\n", checkpoint_bytes / 1024)
    bench("load_weights (single)", () -> load_weights(dir))
    bench("load_weights (sharded)", () -> load_weights(sharded))
    bench("load_weights (direct file)",
          () -> load_weights(joinpath(dir, "model.safetensors")))
    bench("load_config", () -> load_config(dir))
end

main()
