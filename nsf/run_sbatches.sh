#!/bin/bash

for experiment in /atlas/u/kechoi/time-score-dre/nsf/slurm/without_clamp/*.sh
do
    echo $experiment
    chmod u+x $experiment
    sbatch $experiment
    sleep 1
done

# done
echo "Done"
