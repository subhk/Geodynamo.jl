using Geodynamo
using Documenter

DocMeta.setdocmeta!(Geodynamo, :DocTestSetup, :(using Geodynamo); recursive=true)

makedocs(;
    modules=[Geodynamo],
    authors="Subhajit Kar <subhajitkar19@gmail.com>",
    sitename="Geodynamo.jl",
    format=Documenter.HTML(;
        canonical="https://subhk.github.io/Geodynamo.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/subhk/Geodynamo.jl",
    devbranch="main",
)
