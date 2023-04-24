# Perfoming differents Conditioning regimes for a fly's brain
# simulated by a Perceptron. 
#
# Author: José Luis Parra Aldrete
# Advisors: Dr. Thomas Gorin & Dra. Luz Marina Reyes Barrera
# 
# 
# w = (w b), x = (x 1)
# w*x = w*x + b
# 


using LinearAlgebra
using Distributions
using StatsPlots
using Hadamard
using Random

#Elements in the fly (like a Object)
# n is the number of neurons.
# lrate is the learning rate.
# ActFun is for the activation function that the fly use for its learning.
# w is the weight vector.
# memory

mutable struct Fly
    n :: Int64
    lrate :: Float64
    ActFun :: String
    w :: Array{Float64}
    memory :: Any
    normed :: Bool
        
    function Fly(n, lrate, ActFun, normed::Bool=false)

        w = Weight(n+1, normed)
        memory = []
    
        return new(n, lrate, ActFun, w, memory, normed)
    
    end
end


# Activation function
# We have 3 activations Functions:
#   Heaviside for {0, 1}
#   atan(z) in the range [0, 1]
#   logistic regresion in the range [0, 1]
function ActFunction(fly::Fly, z)
    ActFun = fly.ActFun
    
    if ActFun == "Heaviside" || ActFun == "H"
    
        if z > 0
            return 1
        else
            return 0
        end

    elseif ActFun == "atan" 
    
        return atan(z)/pi + 0.5
    
    elseif ActFun == "logistic" || ActFun == "logi" 
    
        return 1 / (1 + exp(-z))
    
    else

        println("Error: activation function is not valid.")
    
    end

end


# Perform the output of the fly given the activation function
function PerceptronOutput(fly::Fly, vecX)

    z = fly.w' * vecX 
    return ActFunction(fly, z)

end


# This fuction return a weight vector of zero
function Weight(n, normed = false)
    
    #return zeros(n)
    if normed == true

        return NormPatterns(rand(n))

    else

        return rand(n)

    end
end


function BiasAdjustment(fly::Fly, patType)

    if patType == "rand"
        if fly.normed == true

            fly.w[end] = -3/4

        else

            fly.w[end] = -(length(fly.w) - 1)/4
        
        end

    else

        fly.w[end] = 0.0

    end

end

# To perform the differents conditioning regimens occording 
# (Wesstnitzer, 2011). 
# 1. + represents presentation of the reinforcer por the given pattern.
# 2. - represents absence of the reinforcer por the given pattern.
# 3. The letters {A, B, C, D} represent the different patterns used for 
#    each regimen. 
# 4. The combination of two patterns is represented by AB (ex. AB = A + B), where the 
#    new pattern is normalized.
function Conditioning(fly::Fly, matPat, Regime, trainType)

    if Regime == "E" #Elemental: A+ B-

        train = [1, 0]

        Trainer(fly, matPat, train, trainType)

    elseif Regime == "M" #Mixture: AB+ CD-

        AB = matPat[:,5]
        CD = matPat[:,end]
        matPat2 = hcat(AB, CD)

        Conditioning(fly, matPat2, "E", trainType)

    elseif Regime == "O" #Overlap: AB+ BC-

        AB = matPat[:,5]
        BC = matPat[:,7]
        matPat2 = hcat(AB, BC)
        
        Conditioning(fly, matPat2, "E", trainType)

    elseif Regime == "2E"

        train = [1, 1, 0]

        Trainer(fly, matPat, train, trainType)

    elseif Regime == "N" #Negative Pattering: A+, B+, AB-
        
        train = [1, 1, 0]
        AB = matPat[:,5]
        matPat2 = hcat(matPat[:,1:2], AB)
        
        Trainer(fly, matPat2, train, trainType)
       
    elseif Regime == "P" #Positive Pattering: A-, B-, AB+
        
        train = [0, 0, 1]
        AB = matPat[:,5]
        matPat2 = hcat(matPat[:,1:2], AB)

        Trainer(fly, matPat2, train, trainType)   
        #vecWeigths = Trainer("full", vecWeigths, AB, lRate, [1])        

    else Regime == "B" #Biconditional discrimination: AB+ CD+ AC- BD-
        
        train = [1, 1, 0, 0]
        AB = matPat[:,5] 
        CD = matPat[:,end] 
        AC = matPat[:,6] 
        BD = matPat[:,8] 
        matPat2 = hcat(AB, CD, AC, BD)
        
        Trainer(fly, matPat2, train, trainType)
    
    end
end

# This fuction perform the learning rule for the PerceptronOutput,
# also called the update for the weight vector.
# For a given regimen, the update is performed randomlly for the given patterns.
# i.e. For the Negative Patterning, the patterns {A, B, AB} could be shuffled as
# {A, AB, B} or {AB, B, A} or more...
# 
#
# w_{t+1} = w_{t} + \eta * (d_{i} - y_{i}) * x_{i}
function Trainer(fly::Fly, matPat, train, trainType)

    if fly.memory == [] 
        
        fly.memory = 0 
    
    end

    w0 = fly.w 
    l = length(train)

    if  trainType == "full"   
        vecOrder = randperm(l)[1:l]

        for i in vecOrder
        
            trainPat = matPat[:,i]
            y = PerceptronOutput(fly, trainPat)
            fly.w = fly.w + fly.lrate*(train[i] - y)*trainPat
        
        end

    end

    if trainType == "partial"
        for i in 1:l
            if train[i] == 1

                trainPat = matPat[:,i]
                y = PerceptronOutput(fly, trainPat)
                fly.w = fly.w + fly.lrate*(1 - y)*trainPat
            
            end
        end
    end


    if fly.w != w0

        fly.memory += 1

    end


    #for i in 1:l
    #    trainPat = matPat[:,i]
    #    y = PerceptronOutput(fly, trainPat)
    #    fly.w = fly.w + fly.lrate*(train[i] - y)*trainPat
    #end

end

# Normalize the patterns
function NormPatterns(matPat)

    for i in 1: length(matPat[1,:])
    
        matPat[1:end-1,i] = normalize(matPat[1:end-1,i]) 
    
    end

matPat
end


# It generates three kind of patterns:
#   1. rOnes: the entries of the patterns are -1 or 1.
#   2. hOnes: return a set of orthogonal patterns from the Hadamard matrix.
#   3. rand: the entries of the patterns are in the interval [-1, 1]
# All the patterns are normalized and with an extra term 1.
function Patterns(kind, nNeu, nPat, normed = false)
    matPat = Array{Float64}(undef,0)

    if kind == "rOnes"

        matPat =  rand(-1:2:1,nNeu,nPat) 

    elseif kind == "hOnes"

        if nNeu > 3 && nNeu % 4 == 0

            matPat =  hadamard(nNeu)[:, 2:5] 
       
        else
            println("Error")
        end

    elseif kind == "rand"

        #matPat = rand(Uniform(-1, 1), nNeu,nPat)

        matPat = rand(nNeu,nPat)       
    else
        println("Error")
    end

    matPat = hcat(matPat,   matPat[:,1] + matPat[:,2], 
                            matPat[:,1] + matPat[:,3], 
                            matPat[:,2] + matPat[:,3], 
                            matPat[:,2] + matPat[:,4], 
                            matPat[:,3] + matPat[:,4])

    matPat = vcat(matPat, ones(nPat + 5)')
    
    if normed == true

        matPat = NormPatterns(matPat)

    end

#M = Pat(kind, matPat)
matPat
end


# It returns the dot product of the original pattern (withput the extra term 1)
# and the bais.
# This is performed for all the regimens.
function DotProduct(fly::Fly, matPat, Regime)
    w = fly.w[1:end-1]
    Aux = Array{Float64}(undef,0)

    if Regime == "E" #Elemental: A+ B-
        
        A = matPat[1:end-1, 1]
        B = matPat[1:end-1, 2]
        
        Aux = hcat(dot(A,w), dot(B,w))

    elseif Regime == "2E" #Double Elemental: A+ B+ C-

        A = matPat[1:end-1, 1]
        B = matPat[1:end-1, 2]
        C = matPat[1:end-1, 3]
        
        Aux = hcat(dot(A,w), dot(B,w), dot(C,w))


    elseif Regime == "M" #Mixture: AB+ CD-
        
        AB = matPat[1:end-1, 5]
        CD = matPat[1:end-1, end]
        
        Aux = hcat(dot(AB, w), dot(CD, w))

    elseif Regime == "O" #Overlap: #Overlap: AB+ BC-
        
        AB = matPat[1:end-1, 5]
        BC = matPat[1:end-1, 7]

        Aux = hcat(dot(AB, w), dot(BC, w))

    elseif Regime == "N" #Negative Pattering: A+, B+, AB-
       
        A = matPat[1:end-1, 1]
        B = matPat[1:end-1, 2]
        AB = matPat[1:end-1, 5]

        Aux = hcat(dot(A, w), dot(B,w), dot(AB, w))

    elseif Regime == "P" #Positive Pattering: AB+, A-, B-
       
        A = matPat[1:end-1, 1]
        B = matPat[1:end-1, 2]
        AB = matPat[1:end-1, 5]

        Aux = hcat(dot(AB, w), dot(A, w), dot(B,w))
    
    else Regime == "B" #Biconditional discrimination: AB+ CD+ AC- BD-
       
        AB = matPat[1:end-1, 5]
        CD = matPat[1:end-1, end]
        AC = matPat[1:end-1, 6]
        BD = matPat[1:end-1, 8]

        Aux = hcat(dot(AB, w), dot(CD, w), dot(AC, w), dot(BD, w)) 

    end 

Aux, fly.w[end]
end


# Learning Index 
# Measure the Leaarning Index for a single fly, given by
# LI = S(X) - S(Y)
# where S(X) is the spike (or activity) for the pattern X. This is performed
# for different regimenes

function LearningIndex(fly::Fly, matPat, Regime)

    LI = 0.0

    if Regime == "E" #Elemental: A+ B-
        
        A = matPat[:, 1]
        B = matPat[:, 2]

        LI = PerceptronOutput(fly, A) - PerceptronOutput(fly, B)

    elseif Regime == "2E" #Double Elemental: A+ B+ C-
        A = matPat[:, 1]
        C = matPat[:, 3]

        LI = PerceptronOutput(fly, A) - PerceptronOutput(fly, C)

    elseif Regime == "M" #Mixture: AB+ CD-
        
        AB = matPat[:, 5]
        CD = matPat[:, end]

        LI = PerceptronOutput(fly, AB) - PerceptronOutput(fly, CD)

    elseif Regime == "O" #Overlap: #Overlap: AB+ BC-
        
        AB = matPat[:, 5]
        BC = matPat[:, 7]

        LI = PerceptronOutput(fly, AB) - PerceptronOutput(fly, BC)

    elseif Regime == "N" #Negative Pattering: A+, B+, AB-
       
        A = matPat[:, 1]
        AB = matPat[:, 5]

        LI = PerceptronOutput(fly, A) - PerceptronOutput(fly, AB)

    elseif Regime == "P" #Positive Pattering: AB+, A-, B-
       
        A = matPat[:, 1]
        AB = matPat[:, 5]

        LI = PerceptronOutput(fly, AB) - PerceptronOutput(fly, A)
    
    else Regime == "B" #Biconditional discrimination: AB+ CD+ AC- BD-
       
        AB = matPat[:, 5]
        AC = matPat[:, 6]

        LI = PerceptronOutput(fly, AB) - PerceptronOutput(fly, AC)

    end 

LI
end
    

# Simulation
function IniCounters(numTrng, Regime)

    if Regime == "P" || Regime == "N" || Regime == "2E"
        
        return zeros(numTrng, 3), zeros(numTrng)

    elseif Regime == "B"

        return zeros(numTrng, 4), zeros(numTrng)

    else

        return zeros(numTrng, 2), zeros(numTrng)

    end
    
end


function Simulation(numFly, numNeu, lRate,  actFun, patType, Regime, normed = false)
    
    numTrng = 100
    averDP, averBais = IniCounters(numTrng + 1, Regime)
    averOut = copy(averDP)
    vecTime = Array{Int64}(undef, 0)
    trainType = "full"

    for f in 1:numFly
        
        fly = Fly(numNeu, lRate, actFun, normed)
        BiasAdjustment(fly, patType)

        matPat = Patterns(patType, numNeu, 4, normed)

        auxdp, auxb  = IniCounters(numTrng + 1, Regime)
        auxout = copy(auxdp)
   

            for t in 1:(numTrng + 1)

                auxdp[t, :], auxb[t] = DotProduct(fly, matPat, Regime)        
                auxout[t, :] = sign.(auxdp[t, :] .+ auxb[t])
                auxout[auxout .== 0] .= -1.0
                auxout = (auxout .+ 1.0)./2.0  

                Conditioning(fly, matPat, Regime, trainType)

            end

        push!(vecTime, fly.memory)
        averDP += auxdp
        averBais += auxb
        averOut += auxout
    end

averDP./numFly , averBais./numFly, averOut./numFly, vecTime
end

function LISimulation(numFly, numNeu, lRate,  actFun, patType, numTrng, normed = false)

    Regime = ["E" "M" "O" "N" "P" "B"]
    trainType = "full"
    matLI = Array{Float64}(undef, 0)


    for r in Regime

        matPat = Patterns(patType, numNeu, 4, normed)
        auxLI = Array{Float64}(undef, 0)

        for f in 1:numFly

            fly = Fly(numNeu, lRate, actFun, normed)
            BiasAdjustment(fly, patType)

                for t in 1:numTrng 

                    Conditioning(fly, matPat, r, trainType)

                end

            push!(auxLI, LearningIndex(fly, matPat, r))

        end

        matLI = vcat(matLI, auxLI)

    end

reshape(matLI, numFly, length(Regime))    
end