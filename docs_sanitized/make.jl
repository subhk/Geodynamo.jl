usingGeodynamo
usingDocumenter

DocMeta.setdocmeta!(Geodynamo,:DocTestSetup,:(usingGeodynamo);recursive=true)

makedocs(;
modules=[Geodynamo],
authors="SubhajitKar<subhajitkar19@gmail.com>",
sitename="Geodynamo.jl",
format=Documenter.HTML(;
canonical="https://subhk.github.io/Geodynamo.jl",
edit_link="main",
assets=String[],
),
pages=[
"Home"=>"index.md",
],
)

deploydocs(;
repo="github.com/subhk/Geodynamo.jl",
devbranch="main",
)
