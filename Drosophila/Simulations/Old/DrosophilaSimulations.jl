using DelimitedFiles
using ColorSchemes
include("Perceptron.jl")

learnRate = [0.1, 0.5]
Regime = ["E", "M", "O", "N", "P", "B"]
#Regime = ["E", "M"]
Pattern = ["rOnes", "hOnes", "rand"]
nList = ["4" "16" "64" "256" "1024"]

for l in learnRate
    for r in Regime
        for p in Pattern
            A, Adp, Stddp, Ab = Simulation(r, p, l)

            open("$r$p$l.txt", "w") do io
                writedlm(io, A)
            end

            if r == "B"
                p1 = plot(A[:,3], label = "Test AB", title = "Training, $r, $p, $l")
                plot!( A[:,5], ylabel = "Frequency", label = "Test AC")
                plot!( A[:,7],  xlabel = "N", ylabel = "Frequency", label = "Test BD")
                plot!( A[:,9],  xlabel = "N", ylabel = "Frequency", label = "Test CD")

            else
                p1 = plot(A[:,3], label = "Test A", title = "Training, $r, $p, $l")
                plot!( A[:,5],  xlabel = "N", ylabel = "Frequency", label = "Test B")   
                
                p2 = plot(Adp[:,:,1]'; xlabel = "Trainings", title = "Dot Product, $r, $p, $l", 
                    label = nList, marker=(:circle,3), palette =:rainbow)
                plot!(Ab[:, :]', label = false, palette =:rainbow)
                plot!(Adp[:, :, 2]'; label = false, marker=(:square,3), palette =:rainbow)
            
            end
            
            plot(p1, yrange = [0,1], markershape = :circle)
            savefig("$r$p$l.pdf") 

            if r != "B"
                plot(p2)
                savefig("DotProduct$r$p$l.pdf") 
            end
            
        end
    end
end