nFly = parse(Int, ARGS[1])
nNeu = parse(Int, ARGS[2])
lRate = parse(Float64, ARGS[3])
aFun =  ARGS[4]
pType = ARGS[5]
Regime = ARGS[6]


using DelimitedFiles
include("Drosophila.jl")


Adp, Ab = Simulation(nFly, nNeu, lRate, aFun, pType, Regime)
Adp = hcat(collect(1:20), Adp, Ab)

open("DroSim$pType.txt", "w") do io
    writedlm(io, Adp)
end