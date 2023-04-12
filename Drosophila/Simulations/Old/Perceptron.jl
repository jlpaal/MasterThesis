using LinearAlgebra
using Distributions
using Plots
using Hadamard
using Random

function ActFunction(x)
    return atan(x)
end

#Cambiar nombre de las funciones
function PercetronOutput(vecWeights, vecX)
    z = vecWeights' * vecX - 0.5
    return ActFunction(z)
    #return atan(ActFunction(vecWeights, vecX))/pi + 0.5
end


function Patterns(kind, nNeu, nPat)
    if kind == "rOnes"
        #return rand(-1:2:1,nNeu,nPat)
        return vcat(rand(-1:2:1,nNeu,nPat), ones(nPat)')

    elseif kind == "hOnes"

        if nNeu > 3 && nNeu % 4 == 0
            return vcat(hadamard(nNeu)[:, 1:4], ones(nPat)')
        else
            println("Error")
        end

    elseif kind == "rand"
        #return rand(Uniform(-1, 1), nNeu,nPat)
        return vcat(rand(Uniform(-1, 1), nNeu,nPat), ones(nPat)')
    else
        println("Error")
    end
end


function DotProduct(Regime, vecWeigths, matPatterns)
    w = vecWeigths[1:end-1]

    if Regime == "E"
        A = matPatterns[1:end-1, 1]
        B = matPatterns[1:end-1, 2]
        return hcat(dot(A,w), dot(B,w)), vecWeigths[end] 

    elseif Regime == "M"
        AB = matPatterns[1:end-1, 1] + matPatterns[1:end-1, 2]
        CD = matPatterns[1:end-1, 3] + matPatterns[1:end-1, 4]
        return hcat(dot(AB, w), dot(CD, w)), vecWeigths[end]

    elseif Regime == "O"
        AB = matPatterns[1:end-1, 1] + matPatterns[1:end-1, 2] 
        BC = matPatterns[1:end-1, 2] + matPatterns[1:end-1, 3]
        return hcat(dot(AB, w), dot(BC, w)), vecWeigths[end]

    elseif Regime == "N"
        A = matPatterns[1:end-1, 1]
        AB = matPatterns[1:end-1, 1] + matPatterns[1:end-1, 2]
        return hcat(dot(A, w), dot(AB, w)), vecWeigths[end]

    elseif Regime == "P"
        AB = matPatterns[1:end-1, 1] + matPatterns[1:end-1, 2] 
        B = matPatterns[1:end-1, 2]
        return hcat(dot(AB, w), dot(B, w)), vecWeigths[end]
    
    else Regime == "B"
        AB = matPatterns[1:end-1, 1] + matPatterns[1:end-1, 2] 
        CD = matPatterns[1:end-1, 3] + matPatterns[1:end-1, 4] 
        AC =  matPatterns[1:end-1, 1] + matPatterns[1:end-1, 3]
        BD =  matPatterns[1:end-1, 2] + matPatterns[1:end-1, 4]

        return  hcat(dot(AB, w), dot(CD, w), dot(AC, w), dot(BD, w)), vecWeigths[end]
    end 
end


function CheckPattern(Regime, vecWeigths, matPatterns)

    if Regime == "E"
        patR = matPatterns[:,1]
        patNR = matPatterns[:,2]
        return PercetronOutput(vecWeigths,patR), PercetronOutput(vecWeigths,patNR)

    elseif Regime == "M"
        patR = matPatterns[:,1] + matPatterns[:,2]
        patNR = matPatterns[:,3] + matPatterns[:,4]
        return PercetronOutput(vecWeigths,patR), PercetronOutput(vecWeigths,patNR)

    elseif Regime == "O"
        patR = matPatterns[:,1] + matPatterns[:,2] 
        patNR = matPatterns[:,2] + matPatterns[:,3]
        return PercetronOutput(vecWeigths,patR), PercetronOutput(vecWeigths,patNR)

    elseif Regime == "N"
        patR = matPatterns[:,1]
        patNR = matPatterns[:,1] + matPatterns[:,2]
        return PercetronOutput(vecWeigths,patR), PercetronOutput(vecWeigths,patNR)

    elseif Regime == "P"
        patR = matPatterns[:,1] + matPatterns[:,2] 
        patNR = matPatterns[:,2]
        return PercetronOutput(vecWeigths,patR), PercetronOutput(vecWeigths,patNR)
    
    else Regime == "B"
        patAB = matPatterns[:,1] + matPatterns[:,2] #AB
        patAC = matPatterns[:,1] + matPatterns[:,3] #AC
        patBD =  matPatterns[:,2] + matPatterns[:,4] #DB
        patCD = matPatterns[:,3] + matPatterns[:,4] #CD

        return PercetronOutput(vecWeigths,patAB), PercetronOutput(vecWeigths,patAC), PercetronOutput(vecWeigths,patBD), OutputFunction(vecWeigths,patCD) 
    end 
end


function Trainer(Type, vecWeigths, matPatterns, lRate, train)
    l = length(train)
    
    if Type == "full"
        vecOrder = randperm(l)[1:l]

        for i in vecOrder
            trainPat = matPatterns[:,i]
            y = PercetronOutput(vecWeigths, trainPat)
            vecWeigths = vecWeigths + lRate*(train[i] - y)*trainPat
        end
    end

    if Type == "partial"
        for i in 1:l
            if train[i] == 1
                trainPat = matPatterns[:,i]
                y = PercetronOutput(vecWeigths, trainPat)
                vecWeigths = vecWeigths + lRate*(1 - y)*trainPat
            end
        end
    end

vecWeigths
end


function RandCond(Regime, vecWeigths, matPatterns, lRate)

    kindTrain = "full"

    if Regime == "E" #Elemental: A+ B-
        train = [1, 0]
        vecWeigths = Trainer(kindTrain, vecWeigths, matPatterns, lRate, train)

    elseif Regime == "M" #Mixture: AB+ CD-
        patR = matPatterns[:,1] + matPatterns[:,2]
        patNR = matPatterns[:,3] + matPatterns[:,4]
        matPatterns2 = hcat(patR, patNR)
        
        return RandCond("E", vecWeigths, matPatterns2, lRate)

    elseif Regime == "O" #Overlap: AB+ BC-
        patR = matPatterns[:,1] + matPatterns[:,2]
        patNR = matPatterns[:,2] + matPatterns[:,3]
        matPatterns2 = hcat(patR, patNR)
        
        return RandCond("E", vecWeigths, matPatterns2, lRate)

    elseif Regime == "N" #Negative Pattering: A+, B+, AB-
        train = [1, 1, 0]
        AB = matPatterns[:,1] + matPatterns[:,2]
        #patNR[end] = 1
        matPatterns2 = hcat(matPatterns[:,1:2], AB)
        
        vecWeigths = Trainer(kindTrain, vecWeigths, matPatterns2, lRate, train)
       
    elseif Regime == "P" #Positive Pattering: A-, B-, AB+
        train = [0, 0, 1]
        AB = matPatterns[:,1] + matPatterns[:,2]
        #patR[end] = 1
        matPatterns2 = hcat(matPatterns[:,1:2], AB)

        vecWeigths = Trainer(kindTrain, vecWeigths, matPatterns2, lRate, train)   
        #vecWeigths = Trainer("full", vecWeigths, AB, lRate, [1])        

    else Regime == "B" #Biconditional discrimination: AB+ CD+ AC- BD-
        train = [1, 1, 0, 0]
        AB = matPatterns[:,1] + matPatterns[:,2]
        CD = matPatterns[:,3] + matPatterns[:,4]
        AC = matPatterns[:,1] + matPatterns[:,3]
        BD = matPatterns[:,2] + matPatterns[:,4]
        matPatterns2 = hcat(AB, CD, AC, BD)
        
        vecWeigths = Trainer(kindTrain, vecWeigths, matPatterns2, lRate, train) 
    end

vecWeigths
end


function Simulation(Regime, Pattern, learnRate)
    numNeurons = 2 .^collect(2:2:10)                #Set for differents num. of neurons
    numTrng = 50                                   #Number of Trainings
    #collTrng = collect(1: 1: numTrng)             #Vector with a set of  training numbers
    numExp = 1                                    #Num. of experiments

    if Regime == "B"
        #To cout the recovery process
        matTestA = zeros(2, length(numNeurons))
        matTestB = zeros(2, length(numNeurons))
        matTestC = zeros(2, length(numNeurons))
        matTestD = zeros(2, length(numNeurons))

        #To save the statistics
        matAverDP = zeros(length(numNeurons), numTrng, 4)
        matStdDesDP = zeros(length(numNeurons), numTrng, 4)
        matAverBais = zeros(length(numNeurons), numTrng)
        
        pos = 0;

        for n in numNeurons
            pos += 1
            countTAPre = 0.0
            countTAPost = 0.0
            countTBPre = 0.0
            countTBPost = 0.0
            countTCPre = 0.0
            countTCPost = 0.0
            countTDPre = 0.0
            countTDPost = 0.0
            
            dotProd = zeros(numExp, numTrng, 4)
            matBais = zeros(numExp, numTrng)

            for e in 1:numExp
                P = Patterns(Pattern, n, 4)
                W = zeros(n + 1);
        
                a, b, c, d = CheckPattern(Regime, W, P)
                countTAPre += a 
                countTBPre += b 
                countTCPre += c
                countTDPre += d
    
                nW = copy(W)
                for t in 1:numTrng
                    nW = RandCond(Regime, nW, P, learnRate)

                    dp, bais = DotProduct(Regime, nW, P)
                    dotProd[e, t, :] = dp
                    matBais[e,t] = bais
                end
        
                a, b, c, d = CheckPattern(Regime, nW, P)
                countTAPost += a
                countTBPost += b 
                countTCPost += c
                countTDPost += d
            end
    
            matTestA[1, pos] = countTAPre
            matTestA[2, pos] = countTAPost
            matTestB[1, pos] = countTBPre
            matTestB[2, pos] = countTBPost  
            matTestC[1, pos] = countTCPre
            matTestC[2, pos] = countTCPost  
            matTestD[1, pos] = countTDPre
            matTestD[2, pos] = countTDPost
            
            for i in 1:4
                matAverDP[pos, :, i] = mean(eachrow(dotProd[:, :, i]))'
                matStdDesDP[pos, :, i] = mean(eachrow(dotProd[:, :, i].^2))'
            end
            matAverBais[pos, :] = mean(eachrow(matBais))'
        end

        matStdDesDP = sqrt.(matStdDesDP - matAverDP.^2)
        return hcat(numNeurons, hcat(matTestA', matTestB', matTestC', matTestD')./numExp), matAverDP, matStdDesDP, matAverBais

    else
        matTestA = zeros(2, length(numNeurons))
        matTestB = zeros(2, length(numNeurons))
        
        #Dot prodoct stats
        matAverDP = zeros(length(numNeurons), numTrng, 2)
        matStdDesDP = zeros(length(numNeurons), numTrng, 2)
        matAverBais = zeros(length(numNeurons), numTrng)

        pos = 0;

        for n in numNeurons
            pos += 1
            countTAPre = 0.0
            countTAPost = 0.0
            countTBPre = 0.0
            countTBPost = 0.0

            dotProd = zeros(numExp, numTrng, 2)
            matBais = zeros(numExp, numTrng)
    
            for e in 1:numExp
                P = Patterns(Pattern, n, 4)
                W = zeros(n + 1);
        
                a, b = CheckPattern(Regime, W, P)
                countTAPre += a 
                countTBPre  += b 
    
                nW = copy(W)
                for t in 1:numTrng
                    dp, bais = DotProduct(Regime, nW, P)
                    dotProd[e, t, :] = dp
                    matBais[e,t] = bais

                    nW = RandCond(Regime, nW, P, learnRate)
                end
        
                a, b = CheckPattern(Regime, nW, P)
                countTAPost += a
                countTBPost += b 
            end
    
            matTestA[1, pos] = countTAPre
            matTestA[2, pos] = countTAPost
            matTestB[1, pos] = countTBPre
            matTestB[2, pos] = countTBPost    
            
            for i in 1:2
                matAverDP[pos, :, i] = mean(eachrow(dotProd[:, :, i]))'
                matStdDesDP[pos, :, i] = mean(eachrow(dotProd[:, :, i].^2))'
            end
            matAverBais[pos, :] = mean(eachrow(matBais))'
        end

        matStdDesDP = sqrt.(matStdDesDP - matAverDP.^2)
        return hcat(numNeurons, hcat(matTestA', matTestB')./numExp), matAverDP, matStdDesDP, matAverBais
    end

end

function SingleFly(n, Regime, Pattern, learnRate)
    #P = Patterns(Pattern, n, 4)
    #P = [-1 1 1 -1; -1 1 -1 1; 1 1 1 1]
    P = [1 0; 1 0; 1 1]
    W = zeros(n + 1);
    nW = copy(W)
    numTrng = 10

    vecdp = zeros(numTrng, 2)
    vecb = Array{Float64}(undef,0)

    for t in 1:numTrng
        dp, b = DotProduct(Regime, nW, P)
        vecdp[t, :] = dp
        push!(vecb, b)

        nW = RandCond(Regime, nW, P, learnRate)
    end

P, nW, vecdp, vecb
end

#Test Functions

function BDTrainer(Type, vecWeigths, matPatterns, lRate, train)
    l = length(train)
    BDBais = Array{Float64}(undef,0)
    
    if Type == "full"
        vecOrder = randperm(l)[1:l]

        for i in vecOrder
            trainPat = matPatterns[:,i]
            y = OutputFunction(vecWeigths, trainPat)
            vecWeigths = vecWeigths + lRate*(train[i] - y)*trainPat

        end
    end

    if Type == "partial"
        for i in 1:l
            if train[i] == 1
                trainPat = matPatterns[:,i]
                y = OutputFunction(vecWeigths, trainPat)
                vecWeigths = vecWeigths + lRate*(1 - y)*trainPat
            end
        end
    end

vecWeigths
end


function BDRandCond(Regime, vecWeigths, matPatterns, lRate)

    kindTrain = "full"

    if Regime == "E" #Elemental: A+ B-
        train = [1, 0]
        vecWeigths = Trainer(kindTrain, vecWeigths, matPatterns, lRate, train)

    elseif Regime == "M" #Mixture: AB+ CD-
        patR = matPatterns[:,1] + matPatterns[:,2]
        patNR = matPatterns[:,3] + matPatterns[:,4]
        matPatterns2 = hcat(patR, patNR)
        
        return RandCond("E", vecWeigths, matPatterns2, lRate)

    elseif Regime == "O" #Overlap: AB+ BC-
        patR = matPatterns[:,1] + matPatterns[:,2]
        patNR = matPatterns[:,2] + matPatterns[:,3]
        matPatterns2 = hcat(patR, patNR)
        
        return RandCond("E", vecWeigths, matPatterns2, lRate)

    elseif Regime == "N" #Negative Pattering: A+, B+, AB-
        train = [1, 1, 0]
        AB = matPatterns[:,1] + matPatterns[:,2]
        #patNR[end] = 1
        matPatterns2 = hcat(matPatterns[:,1:2], AB)
        
        vecWeigths = Trainer(kindTrain, vecWeigths, matPatterns2, lRate, train)
       
    elseif Regime == "P" #Positive Pattering: A-, B-, AB+
        train = [0, 0, 1]
        AB = matPatterns[:,1] + matPatterns[:,2]
        #patR[end] = 1
        matPatterns2 = hcat(matPatterns[:,1:2], AB)

        vecWeigths = Trainer(kindTrain, vecWeigths, matPatterns2, lRate, train)   
        #vecWeigths = Trainer("full", vecWeigths, AB, lRate, [1])        

    else Regime == "B" #Biconditional discrimination: AB+ CD+ AC- BD-
        train = [1, 1, 0, 0]
        AB = matPatterns[:,1] + matPatterns[:,2]
        CD = matPatterns[:,3] + matPatterns[:,4]
        AC = matPatterns[:,1] + matPatterns[:,3]
        BD = matPatterns[:,2] + matPatterns[:,4]
        matPatterns2 = hcat(AB, CD, AC, BD)
        
        vecWeigths = Trainer(kindTrain, vecWeigths, matPatterns2, lRate, train) 
    end

vecWeigths
end