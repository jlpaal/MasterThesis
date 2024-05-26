include("HopNetFun.jl")
include("HopNetSim.jl")

nNeu = 1200
fileHN = JLD.load("hnData$nNeu.jld")
randCorr = fileHN["randCorrMat"];
randEig = fileHN["randEigenMat"];
orthoCorr = fileHN["orthoCorrMat"];
orthoEig = fileHN["orthoEigenMat"];

data = [randCorr, randEig, orthoCorr, orthoEig]
nameData = ["randCorr" "randEig" "orthoCorr" "orthoEig"]

let name = 1

for d in data
    ElbowMethod(d, 10);
    aux = nameData[name]
    savefig("elbow$nNeu$aux.pdf")
    name += 1
end

end
