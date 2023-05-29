include("/home/joseluis/Master/Papers/Hopfield/Simulations/HopNetFun.jl")

function Simulation(WeigthMatrix,IniStates)
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


function SimulationOne(WeigthMatrix,IniStates, k)
    manyMat = zeros(Float64, numExpe^2, numSim, length(stepSave)); #[Expe, MCTs, numSim]
    xStates = Array{Int64}(undef,0);
    promEne = zeros(Float64,length(stepSave), numSim);
    corMat = Array{Float64}(undef,0);
    eigVal = zeros(Float64,numExpe, numSim, length(stepSave));
    eigVec = zeros(Float64,numExpe^2,length(stepSave), numSim);

    matW,kPat = WeigthMatrix(N,k) #return the Weigth matrix and patterns.
    for n in 1:numSim
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

    matAux = manyMat[:,:,fixTime,1]
    for k in 2:size(manyMat)[4]
       matAux = hcat(matAux,manyMat[:,:,fixTime,k])
    end

    clusRes = kmeans(matAux, maxiter=500, nClust, distance = Cityblock())
    clusAssig = reshape(assignments(clusRes), size(manyMat)[2], size(manyMat)[4])

    for k in 1:size(manyMat)[4], n in 1:nClust
        matFreq[k,n] = count(i->(i == n), clusAssig[:,k])
    end

    matFreq, clusRes.centers
end



function SimManyK(WeigthMatrix,IniStates, K, Simulation)
manyMat = zeros(Float64, numExpe^2, numSim, length(stepSave), length(K)); #[Expe, MCTs, numSim]
manyEigVec = zeros(Float64, numExpe, numSim, length(stepSave), length(K)); #[Expe, MCTs, numSim]
manyEigVec[:,:,:,1], manyMat[:,:,:,1] = Simulation(WeigthMatrix, IniStates, K[1])
   
	for k in 2:length(K)
    		manyEigVec[:,:,:,k], manyMat[:,:,:,k] = Simulation(WeigthMatrix, IniStates, K[k]);
	end

manyEigVec, manyMat
end


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
