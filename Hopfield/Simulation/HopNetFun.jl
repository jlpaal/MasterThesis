#########################################################################
#       Functions for the Random Hopfield Network Simulations           #
#       José Luis Parra Aldrete                                         #
#       Advisor: Dr. Thomas Gorin                                       #
#       Date: 07/01/2021                                                #
#########################################################################

######## Functions for develop some simulations #########
#       This functions are for develop:                 #
#       1. Two kind of Weigth Matrices                  #
#       2. Two kind of initials states:                 #
#               -Random initials States                 #
#               -Initials States with a Hamming Dist.   #
#       3. Energy function                              #
#       4. Two recovery algorithms                      #
#               -Synchronous                            #
#               -Asynchronous                           #
#       5. Math tools to analysis                       #
#                                                       #
#########################################################


using LinearAlgebra
using Random
using Statistics
using Hadamard

# Create a Weigth Matrix with orthogonal patterns
# and return the weigth Matrix and patterns used

function OrthoWeigthMatrix(N,K)
Had = hadamard(N)
kMat = zeros(N,K)
Had2 = Had[:,2:end]
Had2 = hcat(Had2,Had[:,1])

    for i in 1:K
    kMat[:,i]=Had2[:,i]
    end

    kMat=round.(Int,kMat)                   
    WMat=kMat*kMat'
    WMat=(WMat-Diagonal(WMat))./K 
    WMat,kMat
end

# Create a Weigth Matrix with random patterns
# and return the weigth Matrix and patterns used

function RandWeigthMatrix(N,K)
    bw(dim::Int64) = rand(-1:2:1,N,K)
    kMat=bw(N)			
    WMat=kMat*kMat'
    WMat=(WMat-Diagonal(WMat))./K
    WMat, kMat
end

#For a given set of patterns, it returns the
#weight matrix according the Hebb's rule

function HebbRule(kPat)
    K = size(kPat)[2]
    wMat = (kPat*kPat')./K - I
end


#For a given set of patterns, it return
#the weight matrix according the Storkey's Rule

function StorkeyRule(kPat)
    K = length(kPat[1,:])
    wMat = (kPat[:,1]*kPat[:,1]')./K
    
    for k in 2:K
        Pat = kPat[:,k]
        wMat = wMat + (Pat*Pat' - Pat*(wMat*Pat)' - (wMat*Pat)*Pat')./K
    end
    
wMat
end


# Random Initial States
# Create a set of Exp-initial states with N neurons

function RandIniStates(N,Exp)
	IniStates=rand(-1:2:1,N,Exp)
end

# Hamming Initial States
# a set of Exp-initial states with a HDis-distance

function HamIniStates(N, Exp, kMat, HDis)
States = Array{Int64}(undef, N, 0)
lK = length(kMat[N,:])
HDis = round.(Int,N*HDis)

	for e in 1:Exp
   	spinp = randperm(N)[1:HDis]
   	randk = rand(1:lK)
   	State=kMat[:,randk]

   		for l in 1:HDis
   		State[spinp[l]]=State[spinp[l]]*(-1)
		end

	States=hcat(States,State)
	end

States
end

# Function to check the equiliprium of a state given
# a weigth matrix and a state 

function CheckEquilibrium(W, X)
 N = length(X)
 Y = sign.(W*X)

dot(Y,X) - N
end

# Return the energy of a vector X given a Weigth Matrix

function Energy(WMat,State)
h=0.0
h=-(State'*WMat*State)/2
end

# Asynchronous updating to recover patterns for a set of initial states
# Return a Matrix with all initials states, middle States and final State

function AsynRecovery(IniStates,WMat,M, Save)
LSave=length(Save)
Exp=length(IniStates[1,:])
N=length(IniStates[:,1])
States = zeros(N,Exp,LSave)
States[:,:,1]=IniStates
iTime = Array{Int64}(undef,0)
t=0
flips=N*M

	for e in 1:Exp
	State=States[:,e,1]
	i=2

	  for flip in 1:flips
	    k= rand(1:N)
	    Xi=2*State[k]*(WMat[:,k]'*State)

		if ((Xi<0)||((Xi==0) &&(rand()>=0.5)))
         	State[k] *= -1
		t=flip
		end

		if (CheckEquilibrium(WMat,State) == 0)
		   States[:,e,i:end] .= State
		   break
		end

		if(flip == N*Save[i])
	 	States[:,e,i]=State
		i+=1
	 	end
	  end
	  States[:,e,LSave]=State
	  push!(iTime,t)
	end

States=round.(Int,States), iTime
end

# Synchronous updating to recover patterns for a set of initial states
# Return a Matrix with all initials states, middle States and final State

function SynRecovery(IniStates,WMat,M, Save)
LSave=length(Save)
Exp=length(IniStates[1,:])
N=length(IniStates[:,1])
States = zeros(N,Exp,LSave)
States[:,:,1]=IniStates
iTime = Array{Int64}(undef,0)
t=0
flips=N*M

        for e in 1:Exp
        State=States[:,e,1]
        i=2

        for flip in 1:LSave
            Eni = Energy(WMat,State)
            State = sign.(WMat*State)

	    for n in 1:N 
		if State[n] == 0
		State[n] = rand(-1:2:1)
		end
	    end

            if Energy(WMat,State) < Eni
               t = flip
            end
            
            if flip == Save[i]
                States[:,e,i]=State
                i+=1    
            end
            
            if (CheckEquilibrium(WMat,State) == 0)
                States[:,e,i:end] .= State
                break
            end
                    
        end
        
          States[:,e,LSave]=State
          push!(iTime,t)
        end

States=round.(Int,States), iTime
end

function ProErr(n,k)
kmax = floor(Int, (n-2)*(k-2)/2 -1)
N = (n-1)*(k-1)
sum = 0

    for i in 0:kmax
    sum = sum + exp(loggamma(N+1) - loggamma(i+1) - loggamma(N-i+1) - N*log(2))
    end

sum
end

######## Functions for Data Analysis ############
#                                               #
#       This functions must to be applied       #
#       under the states after the recovery     #
#       functions                               #
#                                               #
#################################################

# Return the correlation matrix between all the
# given states.
# Note: It is recommended to organize the states by a parameter before
#       use the function CorrMatrix
function CorrMat(Mat)
N=length(Mat[:,1,1])
Exp=length(Mat[1,:,1])
M=length(Mat[1,1,:])
CorrMat=zeros(Exp,Exp,M)

        for m in 1:M
        CorrMat[:,:,m]=(Mat[:,:,m]'*Mat[:,:,m])./N
        end
CorrMat
end

# Return the correlation matrix between all the given states.
# Note: It is recommended to organize the states by a parameter before  use the function CorrMatrix
function CorrWeigthMat(Mat,WMat)
N=length(Mat[:,1,1])
Exp=length(Mat[1,:,1])
M=length(Mat[1,1,:])
CorrMat=zeros(Exp,Exp,M)

        for m in 1:M
        CorrMat[:,:,m]=(Mat[:,:,m]'*(WMat*Mat[:,:,m]))./N
        end
CorrMat
end


# Sort the States acording the final energy
# and return them.
function SortStates(States, WMat)
N=length(States[:,1,1])
Exp=length(States[1,:,1])
M=length(States[1,1,:])
finE = Array{Float64}(undef,0)

	for e in 1:Exp
		push!(finE,Energy(WMat,States[:,e,M]))
	end

indxminMax = sortperm(finE)
StatesOrd = States[:,indxminMax,:]
end

# Measure the Hamming Distance between  the initial
# and final states.
function HamDist(States)
Exp=length(States[1,:,1])
N=length(States[:,1,1])
M=length(States[1,1,:])
estados = hcat(States[:,:,1], States[:,:,M])
diffStates=2*Exp
hammingdmatrix = zeros(diffStates, diffStates)

	for i in 1:diffStates, j in 1:diffStates
    		stdif = count(iszero, estados[:,i] .+ estados[:,j])
    		hammingdmatrix[i,j] = min(stdif, N - stdif)
	end

hammingdmatrix
end


# Measure, for two diferent times, how much states
# have been reaches to a pattern sorted in KMat.
function Frequency(IniM,FinM,KMat)
Exp=length(IniM[1,:])
K=length(KMat[1,:])
N=length(IniM[:,1])
Nm=round.(Int,N/2)

IniDist = zeros(Int64,Exp, K)
FinDist = zeros(Int64,Exp,K)
PDist = zeros(Int64,Nm+1,K)
PDistF = zeros(Int64,Nm+1,K)
PDistAlt = zeros(Int64,Nm+1,K)
IniFre = zeros(Int64,Nm+1,K)

 for i in 1:Exp, j in 1:K
        stdif = count(iszero, IniM[:,i] .+ KMat[:,j])
        IniDist[i,j] = min(stdif,N-stdif)

        stdif = count(iszero, FinM[:,i] .+ KMat[:,j])
        FinDist[i,j] = min(stdif,N-stdif)
 end

 for i in 1:K
        for j in 1:Exp
         a=FinDist[j,i]
         b=IniDist[j,i]+1
         IniFre[b,i]=IniFre[b,i].+1

         if(a==0)
         PDist[b,i]+=1
         else
                a=minimum(FinDist[j,:])
                if (a==0)
                 PDistAlt[b,i]+=1
                end
         end
        end

 end
sum(PDist),Exp-sum(PDist)
end

# Count the frecuency of states for a given procentage
# of similarity with the final state.
function FreqError(FinM,KMat)
Err=[0.0,0.02,0.04,0.06,0.08,0.1,0.15,0.2,0.3]
E=length(Err)
Exp=length(FinM[1,:])
K=length(KMat[1,:])
N=length(FinM[:,1])

FinDist = zeros(Int64,Exp,K)
Freq = zeros(Int128,E)


 for i in 1:Exp, j in 1:K
        stdif = count(iszero, FinM[:,i] .+ KMat[:,j])
        FinDist[i,j] = min(stdif,N-stdif)
 end

 for e in 1:E, j in 1:Exp
         #a=minimum(FinDist[j,:])
               #if (a<= round.(Int,N*Err[e]) )
           a=length(FinDist[j,:][FinDist[j,:].<=round.(Int,N*Err[e])])
        if(a == 1)
                  Freq[e]+=1
                end
 end

Freq
end


function Frequency2(IniM,FinM,KMat)
Exp=length(IniM[1,:])
K=length(KMat[1,:])
N=length(IniM[:,1])
Nm=round.(Int,N/2)

IniDist = zeros(Int64,Exp, K)
FinDist = zeros(Int64,Exp,K)

PDist = zeros(Int64,Nm+1,K)
PDistF = zeros(Int64,Nm+1,K)
PDistAlt = zeros(Int64,Nm+1,K)
IniFre = zeros(Int64,Nm+1,K)

 for i in 1:Exp, j in 1:K
        stdif = count(iszero, IniM[:,i] .+ KMat[:,j])
        IniDist[i,j] = min(stdif,N-stdif)

        stdif = count(iszero, FinM[:,i] .+ KMat[:,j])
        FinDist[i,j] = min(stdif,N-stdif)
 end
i=3
        for j in 1:Exp
         a=FinDist[j,i]
         b=IniDist[j,i]+1
         IniFre[b,i]=IniFre[b,i].+1

         if(a==0)
         PDist[b,i]+=1
         else
                a=minimum(FinDist[j,:])
                if (a==0)
                 PDistAlt[b,i]+=1
                end
         end
        end


sum(PDist),sum(PDistAlt)
# Exp-sum(PDist)-sum(PDistAlt)
end

function Projec(States,KMat)
K=length(KMat[1,:])
N=length(States[:,1,1])
Exp=length(States[1,:,1])
M=length(States[1,1,:])
Pro=zeros(Int128,M,2)

for m in 1:M
        for k in 1:K, e in 1:Exp
           if((dot(KMat[:,k],States[:,e,m])/N)==1)
           Pro[m,1]+=1
           end
        end
Pro[m,2]=Exp-Pro[m,1]
end

Pro
end

###
function FrecEneK(KMat, States, WMat)
K = length(KMat[1,:])
N=length(States[:,1,1])
Exp=length(States[1,:,1])
M=length(States[1,1,:])


end

function swapcols!(X::AbstractMatrix, i::Integer, j::Integer)
        X[:,i], X[:,j] = X[:,j], X[:,i]
end
