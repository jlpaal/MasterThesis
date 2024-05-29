include("Drosophila.jl")
using JLD
using Plots
using LaTeXStrings

# Plots
thesis = ["#9900c9", "#0046ce", "#74bad4", "#0bbc00", "#e8ec00", "#ea9305", "#ff0000"]
thesis2 = ["#9900c9", "#0046ce", "#0bbc00", "#ea9305", "#ff0000", "#74bad4", "#e8ec00"]
thesisg = ["#bcbcbc", "#999999", "#8e8e8e"] 
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


function Simulation(numFly, numNeu, lRate,  actFun, patType, Regime, trainType, numTrng, normed = false)
    
    averDP, averBais = IniCounters(numTrng + 1, Regime)
    averOut = copy(averDP)
    vecTime = Array{Int64}(undef, 0)
    

    for f in 1:numFly
        
        fly = Fly(numNeu, lRate, actFun, normed)
        BiasAdjustment(fly, patType)

        matPat = Patterns(patType, numNeu, 4, normed)

        auxdp, auxb  = IniCounters(numTrng + 1, Regime)
        auxout = copy(auxdp)
   

            for t in 1:(numTrng + 1)

                w0 = fly.w

                auxdp[t, :], auxb[t] = DotProduct(fly, matPat, Regime)        
                auxout[t, :] = sign.(auxdp[t, :] .+ auxb[t])
                auxout[auxout .== 0] .= -1.0
                auxout = (auxout .+ 1.0)./2.0  
                
                Conditioning(fly, matPat, Regime, trainType)

                if fly.w == w0

                     auxdp[t, :], auxb[t] = DotProduct(fly, matPat, Regime)        
                    auxout[t, :] = sign.(auxdp[t, :] .+ auxb[t])
                    auxout[auxout .== 0] .= -1.0
                    auxout = (auxout .+ 1.0)./2.0 

                    auxdp[t+1:end, :] .= auxdp[t, :]'
                    auxb[t+1:end, :] .= auxb[t]
                    auxout[t+1:end, :] .= auxout[t, :]'
                    
                    break            
                end

            end

        push!(vecTime, fly.memory)
        averDP += auxdp
        averBais += auxb
        averOut += auxout
    end

averDP./numFly , averBais./numFly, averOut./numFly, vecTime
end

function LISimulation(numFly, numNeu, lRate, actFun, patType, numTrng, normed = false)

    Regimen = ["E" "M" "O" "N" "P" "B"]
    trainType = "full"
    
    matLI = Array{Float64}(undef, 0)
    matLI2 = Array{Float64}(undef, 0)


    for r in Regimen

        matPat = Patterns(patType, numNeu, 4, normed)
        auxLI = Array{Float64}(undef, 0)
        auxLI2 = Array{Float64}(undef, 0)

        for f in 1:numFly

            fly = Fly(numNeu, lRate, actFun, normed)
            BiasAdjustment(fly, patType)

                for t in 1:numTrng 

                    w0 = fly.w

                    Conditioning(fly, matPat, r, trainType)

                    if fly.w == w0
 
                        break            
                    end


                end

            push!(auxLI, LearningIndex(fly, matPat, r))
            push!(auxLI2, LearningIndex2(fly, matPat, r))

        end

        matLI = vcat(matLI, auxLI)
        matLI2 = vcat(matLI2, auxLI2)

    end

reshape(matLI, numFly, length(Regimen)), reshape(matLI2, numFly, length(Regimen))   
end


function PISimulation(numFly, numNeu, lRate, actFun, patType, numTrng, normed = false)
return LISimulation(numFly, numNeu, lRate, actFun, patType, numTrng, normed)./numFly
end


function PerformanceIndex(numNeu, actFun, lRate, patType, Regimen, numTrng,  trainType, normed = false)
    numFly = 1000
    #trainType = "full"

    countNonReinforced = 0.0
    countReinforced = 0.0
    countFly = 0.0

    for f in 1:numFly

        matPat = Patterns(patType, numNeu, 4, normed)       # Create a set of patterns
        fly = Fly(numNeu, lRate, actFun, normed)            # create a fly
        BiasAdjustment(fly, patType)                        # and adjustment its bais
        

        for t in 1:numTrng

            w0 = fly.w            
            Conditioning(fly, matPat, Regimen, trainType)

            if fly.w == w0        
                break            
            end

        end

        singlePi = LearningIndex2(fly, matPat, Regimen)

        if singlePi == 1

            countReinforced += 1
            countFly += 1

        elseif singlePi == -1
        #else
            countNonReinforced += 1
            countFly += 1

        end

    end

Pi = (countReinforced - countNonReinforced)/(numFly) # Cambiar count pot numFLy
 
end


function EvaluationPI(numNeu, actFun, patType, normed = false, zoom = false)
    numEvaluation = 100
    trainType = ["full" "oP" "oR"]


    learnRate = collect(0.0:0.1:1.0)
    if zoom == true
        learnRate = collect(0.0:0.01:0.2)
    end

    sizeLR = length(learnRate)
    Regimen = ["E" "2E" "M" "O" "P" "N" "B"]
    numTrng = [1, 2, 2, 2, 3, 3, 3]    
    matMeanPI = zeros(7, sizeLR)
    matVarPI = zeros(7, sizeLR)

    for tT in trainType
        for r in 1:length(Regimen)
            for l in 1:sizeLR

                auxMean = 0.0
                auxVar = 0.0
                for t in 1:numEvaluation
                    PI = PerformanceIndex(numNeu, actFun, learnRate[l], patType, Regimen[r], numTrng[r], tT, normed)
                    auxMean = auxMean + PI
                    auxVar = auxVar + PI^2
                end
                
                matMeanPI[r,l] = auxMean
                matVarPI[r,l] = auxVar
            end
        end

        matMeanPI = matMeanPI./numEvaluation
        matVarPI = sqrt.(matVarPI./numEvaluation - matMeanPI.^2)

        if learnRate[end] == 1.0
            save("evaPI$tT$numNeu$patType.jld",  "matMeanPI", matMeanPI, 
            "matStdPI", matVarPI, "learningRate", learnRate);

        else
            save("zoomevaPI$tT$numNeu$patType.jld", "matMeanPI", matMeanPI, 
            "matStdPI", matVarPI, "learningRate", learnRate);

        end
        PlotEvaPI(numNeu, patType, tT, zoom)
    end
    
matMeanPI, matVarPI
end

function PlotEvaPI(numNeu, patType, trainType, zoom = false)
    Regimen = ["E" "2E" "M" "O" "P" "N" "B"]

    lvalue = false
    if trainType == "full" && zoom == false
        lvalue =:bottomright
    end

    if zoom == false

        fileEvaPI = JLD.load("evaPI$trainType$numNeu$patType.jld")
        matMeanPI = fileEvaPI["matMeanPI"]
        matStdPI = fileEvaPI["matStdPI"]
        learnRate = fileEvaPI["learningRate"]
        vecTicks = 0:0.1:learnRate[end]

        lrE = collect(0.2:0.1:0.4)
        pE = ones(3).*0.28
        eE = ones(3).*0.06

        lr2E = collect(0.0:0.1:0.2)
        p2E = ones(3).*0.22
        e2E = ones(3).*0.03
        
        lrM = collect(0.6:0.1:0.8)
        pM = ones(3).*0.34
        eM = ones(3).*0.06
        
        lrO = collect(0.8:0.1:1)
        pO = ones(3).*0.4
        eO = ones(3).*0.05
        
        lrP = collect(0.4:0.1:0.6)
        pP = ones(3).*0.32
        eP = ones(3).*0.07
        
        lrN = collect(0.0:0.1:0.5)
        pN = ones(6).*(-0.01)
        eN = ones(6).*0.05
        
        lrB = collect(0.5:0.1:1)
        pB = ones(6).*(-0.01)
        eB = ones(6).*0.06

        p = plot(lrE, pE, label = false, ribbon = eE, color = "#9900c9",  fillalpha = 0.2)
        plot!(lr2E, p2E, label = false, ribbon = e2E, color = "#0046ce",  fillalpha = 0.2)
        plot!(lrM, pM, label = false, ribbon = eM, color = "#74bad4",  fillalpha = 0.2)
        plot!(lrO, pO, label = false, ribbon = eO, color = "#0bbc00",  fillalpha = 0.2)
        plot!(lrP, pP, label = false, ribbon = eP, color = "#e8ec00",  fillalpha = 0.2)
        plot!(lrN, pN, label = false, ribbon = eN, color = "#ea9305",  fillalpha = 0.2)
        plot!(lrB, pB, label = false, ribbon = eB, color = "#ff0000",  fillalpha = 0.2)

        plot!(learnRate, matMeanPI', marker = :circle, palette = thesis, legendtitle = L"Regimen", 
        minorgrid = true, legend = lvalue, label = Regimen, 
        xlabel = L"\eta", xticks = vecTicks,
        ylabel = L"PI", ylim = (-1,1)) 

        display(Plots.plot(p))

        savefig("PI$trainType$numNeu$patType.pdf")

    else

        fileEvaPI = JLD.load("zoomevaPI$trainType$numNeu$patType.jld")
        matMeanPI = fileEvaPI["matMeanPI"]
        matStdPI = fileEvaPI["matStdPI"]
        learnRate = fileEvaPI["learningRate"]
        vecTicks = 0:0.02:learnRate[end]

        lrE = collect(0.04:0.01:0.08)
        pE = ones(5).*0.28
        eE = ones(5).*0.06

        lr2E = collect(0.0:0.01:0.04)
        p2E = ones(5).*0.22
        e2E = ones(5).*0.03
        
        lrM = collect(0.12:0.01:0.16)
        pM = ones(5).*0.34
        eM = ones(5).*0.06
        
        lrO = collect(0.16:0.01:0.2)
        pO = ones(5).*0.4
        eO = ones(5).*0.05
        
        lrP = collect(0.08:0.01:0.12)
        pP = ones(5).*0.32
        eP = ones(5).*0.07
        
        lrN = collect(0.0:0.01:0.1)
        pN = ones(11).*(-0.01)
        eN = ones(11).*0.05
        
        lrB = collect(0.1:0.01:0.2)
        pB = ones(11).*(-0.01)
        eB = ones(11).*0.06

        p = plot(lrE, pE, label = false, ribbon = eE, color = "#9900c9",  fillalpha = 0.2)
        plot!(lr2E, p2E, label = false, ribbon = e2E, color = "#0046ce",  fillalpha = 0.2)
        plot!(lrM, pM, label = false, ribbon = eM, color = "#74bad4",  fillalpha = 0.2)
        plot!(lrO, pO, label = false, ribbon = eO, color = "#0bbc00",  fillalpha = 0.2)
        plot!(lrP, pP, label = false, ribbon = eP, color = "#e8ec00",  fillalpha = 0.2)
        plot!(lrN, pN, label = false, ribbon = eN, color = "#ea9305",  fillalpha = 0.2)
        plot!(lrB, pB, label = false, ribbon = eB, color = "#ff0000",  fillalpha = 0.2)

        plot!(learnRate, matMeanPI', marker = :circle, palette = thesis, 
        minorgrid = true, label = false, 
        xlabel = L"\eta", xticks = vecTicks,
        ylabel = L" PI ", ylim = (-1,1)) 
        
        display(Plots.plot(p))

        savefig("zoomPI$trainType$numNeu$patType.pdf")
    end
    
end


function OptimzeLR(numNeu, actFun, patType, Regimen, numTrng, targetPi, error,  normed = false)
    numFly = 1000
    infPi = targetPi - error
    supPi = targetPi + error 
    trainType = "partial"
    # Regimen = ["E" "M" "O" "N" "P" "B"]
    
    lRate = 0.5


    for i in 1:100

        matPat = Patterns(patType, numNeu, 4, normed)   # Create a set of patterns
        countNonReinforced = 0.0
        countReinforced = 0.0
        countFly = 0.0

        for f in 1:numFly

            fly = Fly(numNeu, lRate, actFun, normed)    # create a fly
            BiasAdjustment(fly, patType)                # and adjustment its bais
            

            for t in 1:numTrng

                w0 = fly.w            
                Conditioning(fly, matPat, Regimen, trainType)

                if fly.w == w0        
                    break            
                end

            end

            singlePi = LearningIndex2(fly, matPat, Regimen)

            if singlePi == 1

                countReinforced += 1
                countFly += 1

            elseif singlePi == -1

                countNonReinforced += 1
                countFly += 1

            end

        end

        Pi = (countReinforced - countNonReinforced)/(countFly)

        if  Pi >= supPi

            lRate = lRate*0.5
        
        elseif Pi <= infPi

            lRate = lRate*1.5

        else
            print("$Pi $lRate")
            print("\n")
            print(i)
            break

        end


    end

lRate     
end


function TrainingConvergence(numFly, actFun, patType, numTrng, normed = false)
    numNeu = [4, 40, 360, 400, 4000]
    Regimen = ["E" "M" "O" "N" "P" "B"]
    learnRate = [0.1, 0.25, 0.5, 0.75, 1.0]
    trainType = "full"

    matAver = zeros(length(numNeu), length(learnRate), length(Regimen))
    matStdDes = zeros(length(numNeu), length(learnRate), length(Regimen))

    for i in 1:length(numNeu) 
        n = numNeu[i]
        
        for j in 1:length(learnRate)
            l = learnRate[j]

            for k in 1:length(Regimen)
                r = Regimen[k]

                auxAver = Array{Float64}(undef, 0)
        
                for f in 1:numFly
                    
                    matPat = Patterns(patType, n, 4, normed)
                    fly = Fly(n, l, actFun, normed)
                    BiasAdjustment(fly, patType)

                        for t in 1:numTrng 
                            w0 = fly.w

                            Conditioning(fly, matPat, r, trainType)

                            if fly.w == w0
                                break            
                            end

                        end

                    push!(auxAver, fly.memory)
                end

                matAver[i, j, k] = mean(auxAver)
                matStdDes[i, j, k] = mean(auxAver.^2) - (matAver[i, j, k])^2
            end
        end

        # matAver[i, :, :] = matAver[i, :, :]./n
        # matStdDes[i, :, :] = matStdDes[i, :, :]./n
    end

matAver, sqrt.(matStdDes)
end


function DotProdoctConvergence(numFly, actFun, patType, Regime, numTrng, normed = false)

    numNeu = [4, 40, 360, 400, 4000]
    learnRate = [0.1, 0.25, 0.5, 0.75, 1.0]
    trainType = "full"

    matAver = Array{Int64}(undef, 0)

    if Regime == "P" || Regime == "N" || Regime == "2E"
        
        matAver = zeros(length(numNeu), length(learnRate), 4)

    elseif Regime == "B"

        matAver = zeros(length(numNeu), length(learnRate), 5)

    else

        matAver = zeros(length(numNeu), length(learnRate), 3)

    end
    matStdDes = copy(matAver)



    for i in 1:length(numNeu)

        n = numNeu[i]

        for j in 1:length(learnRate)

            l = learnRate[j]

            for f in 1:numFly
                                        
                matPat = Patterns(patType, n, 4, normed)
                fly = Fly(n, l, actFun, normed)
                BiasAdjustment(fly, patType)

                for t in 1:numTrng

                    w0 = fly.w
                    Conditioning(fly, matPat, r, trainType)
                    if fly.w == w0
                        break            
                    end

                end

                auxDP, auxb = DotProduct(fly, matPat, Regime)  
                auxDP = hcat(auxDP, auxb)


                matAver[i, j, :] += auxDP'
                matStdDes[i, j, :] += (auxDP.^2)'

            end

        end


        # matAver[i, :, :] = matAver[i, :, :]./n
        #matStdDes[i, :, :] = matStdDes[i, :, :]./n

    end


    matAver = matAver./numFly
    matStdDes =  matStdDes./numFly - matAver.^2

matAver, sqrt.(matStdDes)
end
