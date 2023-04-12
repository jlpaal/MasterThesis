#!/bin/sh
for n in 16
do

    for r in "hOnes" "rOnes" "rand"
    do
        julia DrosoSimulation.jl 100 $n 0.1 "H" $r "P"

    done
    
        #gnuplot ThesisPlots.gp

done
#mv *.pdf Plots/
