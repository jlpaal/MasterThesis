
set term pdf


plot "DrosoSim.txt" u 1:2 
replot "DrosoSim.txt" u 1:3
replot "DrosoSim.txt" u 1:4
replot "DrosoSim.txt" u 1:5

set terminal pdfcairo 
set output 'test.pdf'
replot
unset output
unset terminal
