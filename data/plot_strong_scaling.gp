set xlabel 'Ncore'
set ylabel 'Flops/s'

plot for [i=1:9] 'strong_scaling_3072xx2.dat' i 0 u 1:(1/$i) title 'nth=1'

 plot 'strong_scaling_3072xx2.dat' i 0 u 1:(1/$2) title 'nth=1','' i 0 u 1:(1/$3) title 'nth=2', i 0 '' u 1:(1/$4) title 'nth=3', '' i 0 u 1:(1/$5) title 'nth=6', '' i 0 u 1:(1/$6) title 'nth=1 cco', '' i 0 u 1:(1/$7) title 'nth=2 cco', '' i 0 u 1:(1/$8) title 'nth=3 cco', '' i  0 u 1:(1/$9) title 'nth=6 cco'

