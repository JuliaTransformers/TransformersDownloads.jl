using Documenter
using TransformersDownloads

makedocs(
    modules = [TransformersDownloads],
    sitename = "TransformersDownloads.jl",
    authors = "AbrJA and contributors",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
    ),
    checkdocs = :exports,
    pages = [
        "Home" => "index.md",
        "Ecosystem Guide" => "guide.md",
    ],
)

deploydocs(
    repo = "github.com/JuliaTransformers/TransformersDownloads.jl.git",
    target = "build",
    push_preview = true,
)
