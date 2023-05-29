include("Drosophila.jl")

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


function Simulation(numFly, numNeu, lRate,  actFun, patType, Regime, numTrng, normed = false)
    
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

                    auxdp[t+1:end, :] .= auxdp[t]
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


function OptimzeLR(numFly, numNeu, actFun, patType, numTrng, learnIndex, normed = false)


    Regimen = ["E" "M" "O" "N" "P" "B"]
    trainType = "partial"
    
    
end