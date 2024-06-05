include("HopNetFun.jl")
using Clustering
using JLD
using Plots
using LaTeXStrings
using Distances

# Plots
thesis = ["#9900c9", "#0046ce", "#74bad4", "#0bbc00", "#e8ec00", "#ea9305", "#ff0000"]
thesis2 = ["#9900c9", "#0046ce", "#0bbc00", "#ea9305", "#ff0000", "#74bad4", "#e8ec00"]
thesisg = ["#bcbcbc", "#999999", "#8e8e8e"] 

function Simulation(WeigthMatrix, IniStates)
    manyMat = zeros(Float64, numExpe^2, length(stepSave), numSim); #[Expe, MCTs, numSim]
    xStates = Array{Int64}(undef,0);
    promEne = zeros(Float64,length(stepSave), numSim);
    corMat = Array{Float64}(undef,0);
    eigVal = zeros(Float64,numExpe,length(stepSave), numSim);
    eigVec = zeros(Float64,numExpe^2,length(stepSave), numSim);
    eigDistMat = zeros(Float64, numExpe, numExpe, length(stepSave));

    matW,kPat = WeigthMatrix(N,K) #return the Weigth matrix and patterns.
    
    for n in 1:numSim
        X0 = IniStates(N,numExpe) #Initials states.
        xStates,Times = AsynRecovery(X0,matW,numMC,stepSave);
        xStates = SortStates(xStates,matW); #Sort the final state.
        corMat = abs.(CorrMat(xStates)); #Correlation matrix for the final states

        for s in 1:length(stepSave)
            promEne[s,n] = tr(Energy(matW, xStates[:,:,s]))/numExpe
            manyMat[:,s,n] = vec(corMat[:,:,s])
            eigVal[:,s,n] = sort(eigen(corMat[:,:,s]).values, rev = true)
            eigVec[:,s,n] = vec(eigen(corMat[:,:,s]).vectors)
        end
    end
    

    for t in 1:length(stepSave) 

        for i in 1:numExpe, j in i:numExpe
            eigDistMat[i,j,t] = sum(abs.(eigVal[:,t,i] - eigVal[:,t,j]))
        end
        eigDistMat[:,:,t] = eigDistMat[:,:,t] + eigDistMat[:,:,t]'
        
    end

    matForClu = Array{Float64}(undef,0)
    #matForClu = manyMat[:,1,:]
    #matForClu = eigVec[:,1,:]
    matForClu = eigDistMat[:,:,1]


    for i in 2:length(stepSave)
        #matForClu = hcat(matForClu,manyMat[:,i,:])
        #matForClu = hcat(matForClu,eigVec[:,i,:])
         matForClu = hcat(matForClu,eigDistMat[:,:,i])
    end

    clusRes = kmeans(matForClu, maxiter=200, nClust);
    clusTime = reshape(clusRes.assignments, (numSim,length(stepSave)))
    freqClus = zeros(length(stepSave), nClust)

    for t in 1:length(stepSave)
        for n in 1:nClust
            freqClus[t,n] = count(i->(i == n), clusTime[:,t])
        end
    end
    
    for i in 1:nClust
        maxEnt = maximum(freqClus[i,i:end])
        for p in i:nClust     
            if freqClus[i,p] == maxEnt
                swapcols!(freqClus, i,p)
                break
            end  
        end
    end

freqClus
end


# Return the eigenvalues and correlation matrix as a vector, for a simulation
# with parameters 

function SimulationOne(WeigthMatrix, IniStates, k)

    manyMat = zeros(Float64, numExpe^2, numSim, length(stepSave)); #[Expe, numSim,  MCTs]
    xStates = Array{Int64}(undef,0);
    promEne = zeros(Float64,length(stepSave), numSim);
    corMat = Array{Float64}(undef,0);
    eigVal = zeros(Float64,numExpe, numSim, length(stepSave));
    eigVec = zeros(Float64,numExpe^2,length(stepSave), numSim);

    matW, kPat = WeigthMatrix(N,k) #return the Weigth matrix and patterns.

    for n in 1:numSim

	    X0 = IniStates(N, numExpe) #Initials states.
        xStates, Times = AsynRecovery(X0, matW, numMC, stepSave);
        # This line sort the states.
        xStates = SortStates(xStates, matW); #Sort the final state.
        corMat = abs.(CorrMat(xStates)); #Correlation matrix for the final states

        for t in 1:length(stepSave)

            promEne[t,n] = tr(Energy(matW, xStates[:,:,t]))/numExpe
            manyMat[:,n,t] = vec(corMat[:,:,t])
            eigVal[:,n,t] = sort(eigen(corMat[:,:,t]).values, rev = true)
            eigVec[:,t,n] = vec(eigen(corMat[:,:,t]).vectors)

        end
    end


eigVal, manyMat
end



function SimulationTwo(WeigthMatrix,IniStates, k)
    
    manyMat = zeros(Float64, numExpe^2, numSim, length(stepSave)); #[Expe, MCTs, numSim]
    xStates = Array{Int64}(undef,0);
    promEne = zeros(Float64,length(stepSave), numSim);
    corMat = Array{Float64}(undef,0);
    eigVal = zeros(Float64,numExpe, numSim, length(stepSave));
    eigVec = zeros(Float64,numExpe^2,length(stepSave), numSim);

    for n in 1:numSim
    	
        matW,kPat = WeigthMatrix(N,k) #return the Weigth matrix and patterns.
	    X0 = IniStates(N,numExpe) #Initials states.
        xStates,Times = AsynRecovery(X0,matW,numMC,stepSave);
       # xStates = SortStates(xStates,matW); #Sort the final state.
        corMat = abs.(CorrMat(xStates)); #Correlation matrix for the final states

        for t in 1:length(stepSave)
            promEne[t,n] = tr(Energy(matW, xStates[:,:,t]))/numExpe
            manyMat[:,n,t] = vec(corMat[:,:,t])
            eigVal[:,n,t] = sort(eigen(corMat[:,:,t]).values, rev = true)
            eigVec[:,t,n] = vec(eigen(corMat[:,:,t]).vectors)
        end
    end


eigVal, manyMat
end


function ClustFixedTime(manyMat,fixTime, nClust)
    matFreq = zeros(size(manyMat)[4], nClust)

    matAux = manyMat[:,:,fixTime,1]
    for k in 2:size(manyMat)[4]
       matAux = hcat(matAux,manyMat[:,:,fixTime,k])
    end

    clusRes = kmeans(matAux, maxiter=500, nClust)
    clusAssig = reshape(clusRes.assignments, size(manyMat)[2], size(manyMat)[4])

    for k in 1:size(manyMat)[4], n in 1:nClust
        matFreq[k,n] = count(i->(i == n), clusAssig[:,k])
    end

    matFreq, clusRes.centers
end

function ClustEigFixedTime(manyMat,fixTime, nClust)
    matFreq = zeros(size(manyMat)[4], nClust)

    # For a fixed time, we choose all the eigenvalues
    # to cluster. 
    matAux = manyMat[:,:,fixTime,1]
    for k in 2:size(manyMat)[4]
       matAux = hcat(matAux,manyMat[:,:,fixTime,k])
    end

    clusRes = kmeans(matAux, maxiter=1000, nClust, distance = Cityblock())
    clusAssig = reshape(assignments(clusRes), size(manyMat)[2], size(manyMat)[4])

    for k in 1:size(manyMat)[4], n in 1:nClust
        matFreq[k,n] = count(i->(i == n), clusAssig[:,k])
    end

    matFreq, clusRes.centers
end


# For a given kind of weight matrix, initial states (IniStates),
# values of k-patterns, and one simulation (at the top),
# the function makes many simulations

function SimManyK(WeigthMatrix, IniStates, K, Simulation)

    manyMat = zeros(Float64, numExpe^2, numSim, length(stepSave), length(K)); #[Expe, numSim, MCTs, numKpat]
    manyEigVec = zeros(Float64, numExpe, numSim, length(stepSave), length(K)); #[Expe, numSim, MCTs, numKpat]
    manyEigVec[:,:,:,1], manyMat[:,:,:,1] = Simulation(WeigthMatrix, IniStates, K[1])
    
        for k in 2:length(K)
                manyEigVec[:,:,:,k], manyMat[:,:,:,k] = Simulation(WeigthMatrix, IniStates, K[k]);
        end

manyEigVec, manyMat
end


# 
function ClustEigFullTime(matEigVal, nClust)
matEigTime = Array{Float64}(undef,0)
append!(matEigTime,vec(matEigVal[:,1,2:end,1]))

    for n in 2:size(matEigVal)[2]
        matEigTime = hcat(matEigTime, vec(matEigVal[:,n,2:end,1]))
    end

    for k in 2:size(matEigVal)[4]
        for n in 1:size(matEigVal)[2]
            matEigTime = hcat(matEigTime, vec(matEigVal[:,n,2:end,k]))
        end
    end


    matFreq = zeros(size(matEigVal)[4], nClust)
    clusRes = kmeans(matEigTime,nClust, maxiter=500)
    clusAssig = reshape(assignments(clusRes), size(matEigVal)[2], size(matEigVal)[4])

    for k in 1:size(matEigVal)[4], n in 1:nClust
        matFreq[k,n] = count(i->(i == n), clusAssig[:,k])
    end

matFreq, clusRes
end


function PlotScatterCluster(clusResults, numSim, numMCTS, numKPat)

    vecAssigments = assignments(clusResults)
    lAssigments = length(vecAssigments)
    auxCorrections = Array{Float64}(undef,0)
    
    for t in 1:numMCTS
        auxCorrections = vcat(auxCorrections, repeat([:circle], 100), 
                        repeat([:square], 100), 
                        repeat([:dtriangle], 100))
    end

    #correctionAssigments = vecAssigments + auxCorrections

    # Background Plot

    numClusters = nclusters(clusResults)
    vecRange = ones(lAssigments).*0.5
    vecClus = ones(lAssigments)

    dx = collect(0:1: lAssigments-1)
    p = plot(dx, vecClus, ribbon = vecRange, fillalpha = 0.2, label = false, palette = thesis)

    for i in 2:numClusters
        vecClus = ones(lAssigments).*i
        plot!(dx, vecClus, ribbon = vecRange, fillalpha = 0.2, label = false, palette = thesis)
    end

    xticksPos = 0:numSim*numKPat:lAssigments
    yticksPos = 1:1:numClusters
    xticksCustom =(xticksPos, string.(0:numMCTS-1))

    if numMCTS == 14
        xticksCustom =(xticksPos, string.(0:2:numMCTS))
    elseif  numMCTS == 21
        xticksCustom =(xticksPos, string.(0:3:numMCTS))
    elseif  numMCTS == 35
        xticksCustom =(xticksPos, string.(0:5:numMCTS))
    end
                                                
    scatter!(dx, vecAssigments, label = false, grid = true, gridalpha=1,
            markercolor =:blue, marker = auxCorrections, markersize = 6,
            tickfontsize = 14, labelfontsize = 20,    
            xlabel = "MCTS", xticks = xticksCustom,
            ylabel = "Clusters", yticks = yticksPos)
    
    display(Plots.plot(p))

end

function PlotFrequencyClusster(clusRes, numSim, numMCTS, numKPat, countByKO = false)

    clusAssig = assignments(clusRes)
    numClusters = nclusters(clusRes)
    auxAssignments = reshape(clusAssig, numSim*numKPat, numMCTS)
    matAssignments = zeros(100, numKPat, numMCTS)
    matFreq = Array{Float64}(undef,0)

    for t in 1:numMCTS
        matAssignments[:,:, t] = reshape(auxAssignments[:, t], 100, numKPat)
    end

    if countByKO == false

        matFreq = zeros(numMCTS, numClusters)
        
        for t in 1:numMCTS
            for c in 1:numClusters

                matFreq[t, c] = count(i->(i == c), matAssignments[:, :, t])

            end
        end

        vecTicks = 0:1.0:numMCTS
        dx = collect(0:numMCTS-1)
        p = plot(dx, matFreq, markershape =:circle,  legendtitle = "Cluster", palette = thesis,
        xticks = vecTicks, xlabel = "MCTS")

        display(Plots.plot(p))
        
    else

        matFreq = zeros(numMCTS, nClust, numKPat)
        for t in 1:numMCTS
            for c in 1:nClust
                for k in 1:numKPat

                matFreq[t, c, k] = count(i->(i == c), matAssignments[:, k, t])
                
                end
            end
        end

        vecTicks = 0:1.0:numMCTS
        dx = collect(0:numMCTS-1)
        p1 = plot(dx, matFreq[:, :, 1], markershape =:circle, label = false, palette = thesis,
        xticks = vecTicks, xlabel = "MCTS")
        p2 = plot(dx, matFreq[:, :, 2], markershape =:circle, label = false, palette = thesis,
        xticks = vecTicks, xlabel = "MCTS")
        p3 = plot(dx, matFreq[:, :, 3], markershape =:circle,  label = false, palette = thesis,
        xticks = vecTicks, xlabel = "MCTS")

        display(Plots.plot(p1, p2, p3))

    end

    matFreq
end


function SimpleClustering(matData, nClust, countByKO = false)

    matVecCluster = Array{Float64}(undef,0)
    append!(matVecCluster,vec(matData[:,1,1,1]))
    numSim = size(matData)[2]
    numMCTS = size(matData)[3]
    numKPat = size(matData)[4]

        for t in 1:numMCTS, k in 1:numKPat
            matVecCluster = hcat(matVecCluster, matData[:, :, t, k])
        end

    clusRes = kmeans(matVecCluster[:, 2:end], nClust, maxiter=5000)
    PlotScatterCluster(clusRes, numSim, numMCTS, numKPat)

    #matFreq = PlotFrequencyClusster(clusRes, numSim, numMCTS, numKPat, countByKO)

clusRes

end

function SwapCluster(clustResults, c1, c2)

    positionCluster1 = findall(x -> x == c1, assignments(clustResults))
    positionCluster2 = findall(x -> x == c2, assignments(clustResults))

    assignments(clustResults)[positionCluster1] .= c2
    assignments(clustResults)[positionCluster2] .= c1

    clustResults.centers[:,c1], clustResults.centers[:,c2] = clustResults.centers[:,c2], clustResults.centers[:,c1] 

    clustResults
end

function ElbowMethod(matData, numClusters)

    matVecCluster = Array{Float64}(undef,0)
    append!(matVecCluster,vec(matData[:,1,1,1]))
    averElbowM = Array{Float64}(undef,0)
    stdElbowM = Array{Float64}(undef,0)

    numMCTS = size(matData)[3]
    numKPat = size(matData)[4]
    numSim = size(matData)[2]

    for t in 1:numMCTS, k in 1:numKPat
        matVecCluster = hcat(matVecCluster, matData[:, :, t, k])
    end

    
    for n in 2:numClusters
        
        vecAux = Array{Float64}(undef, 0)

        for s in 1:50
            clusterResults = kmeans(matVecCluster, n, maxiter=10000)
            
            for i in 1:(numMCTS*numSim)
                push!(vecAux, euclidean(matVecCluster[:,i], clusterResults.centers[:,clusterResults.assignments[i]]))
            end
        end

        push!(averElbowM, mean(vecAux))
        push!(stdElbowM, mean(vecAux.^2) - averElbowM[end]^2 )
    end

    # Plot
        dCluster = collect(2:numClusters)
    p = plot(dCluster, averElbowM, markershape =:circle,  label = false, 
        palette = thesis, ribbon = sqrt.(stdElbowM), fillalpha = 0.2, markersize = 6,
        tickfontsize = 14, labelfontsize = 20, 
        xlabel = "Clusters", xticks = 2:1:numClusters,
        ylabel = L"\langle d(C, \textbf{x}) \rangle")

    display(Plots.plot(p))

end

function SuccessMatrix(clusterResults, assignmentsToCount)

    matAssignments = reshape(clusterResults.assignments, 300, 8)
    matAssignmentsEndTime = reshape(matAssignments[:, end], 100, 3)
    lengthAssignmentsTC = length(assignmentsToCount)
    matCountAssignments = zeros(3, lengthAssignmentsTC + 1)

    # Counter assigments
    for k in 1:3 
        for c in 1:lengthAssignmentsTC
            matCountAssignments[k, c] = count(==(assignmentsToCount[c]),  matAssignmentsEndTime[:,k])
        end
        matCountAssignments[k, end] = 100 - sum(matCountAssignments[k, :]) 
    end

    matCountAssignments = matCountAssignments./100

    # Plot the success matrix
    xlabel = [L"\gamma_{k50}" L"\gamma_{k100}" L"\gamma_{k150}" L"\gamma_{other}"]
    ylabel = [L"k = 50\%" L"k = 100\%" L"k = 150\%"]
    p = heatmap(matCountAssignments, color =:BuPu, 
            tickfontsize = 14, labelfontsize = 20,
            xticks = (1:4, xlabel),
            yticks = (1:3, ylabel))
    fontsize = 14
    nrow, ncol = size(matCountAssignments)
    ann = [(j, i, text(round(matCountAssignments[i,j], digits=2), 
            fontsize, (matCountAssignments[i,j] <= 0.7 ? :black : :white), :center))
            for i in 1:nrow for j in 1:ncol]
    annotate!(ann, linecolor=:white)

    display(Plots.plot(p))


end

function SimulationClusterForMCS(numNeurons, numPatterns, numExp, numMC, numSim)
    
    stepSave = collect(0:1:numMC);#Simulations tu save
    stepSave[1] = 1
    nClust = 3;
    
    A, B = SimManyK(RandWeigthMatrix, RandIniStates, K, SimulationOne);
    HA, HB = SimManyK(OrthoWeigthMatrix, RandIniStates, K, SimulationOne);
    
    matFreq, ClusRes = ClustEigFullTime(A,3);
    HmatFreq, HClusRes = ClustEigFullTime(HA,3);
    
    freqClusEig = zeros(Float64,length(K),nClust, length(stepSave))
    CenEig = zeros(Float64, numSim,length(K), length(stepSave))

    for t in 1:length(stepSave)
        freqClusEig[:,:,t], fake = ClustFixedTime(A,t, nClust);
    end
    freqClusEig = freqClusEig./numSim;
    freqClusEig2 = copy(freqClusEig);
    freqClusEig2 = SwapClusters(freqClusEig2)
        
end


function SwapClustersFrequency(matFreq)
    time = size(matFreq)[3]
    numCluster = size(matFreq)[1]
    numPat = size(matFreq)[2]
    matResu = zeros(numCluster, numPat, time)
    
    for t in 1:time
        matAux = copy(matFreq[:,:,t])
        
        for k in 1:numCluster    
            maxIndex = argmax(matAux[k,:])
            matResu[:,k, t] = matAux[:,maxIndex]
            matAux = matAux[:, setdiff(1:end, maxIndex)]
        
        end
    end
matResu
end
