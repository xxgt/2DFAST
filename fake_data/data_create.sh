#!/bin/bash
# This is a script generate simulated FRB test data for heavy test

k=60
w=0.016
dms=(300 600 900 1200)
for (( d = 0 ; $d <= 3; d=d+1 )) ; do
   dm=${dms[$d]}
   mkdir DM_$dm
   for (( i = 1 ; $i <= 5; i=i+1 )) ; do
      mkdir DM_$dm/SNR_$i
      t=$(($k*1000))
      echo $w
      for (( j = 1 ; $j <= 100; j=j+1 )) ; do
         fake -period $t -width $w -nbits 8 -nchans 4096 -tsamp 1000 -tobs $k -fch1 1500 -foff 0.12207 -dm $dm -nosmear -snrpeak $i > ./DM_$dm/SNR_$i/noise_level_$j.fil 
      done
      echo 'tlen:'$k 'done!'
   done
done
