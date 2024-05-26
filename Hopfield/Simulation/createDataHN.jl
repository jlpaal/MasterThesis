using JLD
include("HopNetSim.jl")

N = 40
K = round.(Int, [0.5*N*0.13, N*0.13, 1.5*N*0.13])
numSim = 100 #Number of simulations
numExpe = 100; #Number of experiment in the simulation.

numMC = 7 #Number of MCTS 
stepSave = collect(0:1:numMC);#Simulations tu save
stepSave[1] = 1

randEigenValMat, randCorrMat = SimManyK(RandWeigthMatrix, RandIniStates, K, SimulationOne);
println("KWH")
orthoEigenValMat, orthoCorrMat = SimManyK(OrthoWeigthMatrix, RandIniStates, K, SimulationOne);
println("KWH")

save("hnData$N.jld", "randEigenMat", randEigenValMat, "randCorrMat", randCorrMat, 
                     "orthoEigenMat", orthoEigenValMat, "orthoCorrMat", orthoCorrMat);


println("La comedia é finita.")
