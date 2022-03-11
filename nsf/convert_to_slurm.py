import numpy as np
import itertools
import glob
import os


SBATCH_PREFACE = \
"""#!/bin/bash
#SBATCH --partition=atlas --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=atlas1,atlas2,atlas3,atlas4,atlas5,atlas6,atlas7,atlas9,atlas10,atlas16,atlas17,atlas19,atlas22
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name="{}.sh"
#SBATCH --error="{}/{}_err.log"
#SBATCH --output="{}/{}_out.log"\n
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
"""

# constants for commands
method = 'joint'
lrs = [2e-4]
batch_sizes = [64]
alphas = [0]

OUTPUT_PATH="/atlas/u/kechoi/time-score-dre/nsf/slurm/without_clamp/"
# exp_id = 'rerun_rq_nsf_mnist_copula_0'
# script_fn = os.path.join(OUTPUT_PATH, '{}.sh'.format(exp_id))

counter = 0
for i in range(4):
    counter += 1
    exp_id = 'rq_nsf_mnist_copula_{}'.format(i)
    script_fn = os.path.join(OUTPUT_PATH, '{}.sh'.format(exp_id))

    base_cmd = 'python3 experiments/images_centering_copula.py \
    with experiments/image_configs/copula2/{}.json'.format(i)

    # write to file
    with open(script_fn, 'w') as f:
        print(SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH,
                                    exp_id), file=f)
        print(base_cmd, file=f)
        print('sleep 1', file=f)
print('Generated {} config files'.format(counter))

# base_cmd = 'python3 experiments/images.py with \
# experiments/image_configs/mnist-8bit-tre.json'
# base_cmd = 'python3 experiments/images.py with \
# experiments/image_configs/mnist-8bit.json'
# base_cmd = 'python3 experiments/images_centering.py with \
# experiments/image_configs/mnist-8bit-noresnet2.json'
# base_cmd = 'python3 experiments/images_centering.py with \
# experiments/image_configs/mnist-8bit-noresnet4.json'
# base_cmd = 'python3 experiments/images.py with \
# experiments/image_configs/mnist-8bit-noresnet.json'
# base_cmd = 'python3 experiments/images_centering_copula.py \
# with experiments/image_configs/mnist-8bit-copula2.json'
# base_cmd = 'python3 experiments/images_centering_copula.py \
# with experiments/image_configs/mnist-8bit-copula.json'
# base_cmd = 'python3 experiments/images_centering_copula.py \
# with experiments/image_configs/copula/0.json'
#
# # write to file
# with open(script_fn, 'w') as f:
#     print(SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id), file=f)
#     print(base_cmd, file=f)
#     print('echo "running score method"', file=f)
#     print('sleep 1', file=f)
# print('hi')



# print('Generated {} experiment files'.format(counter))
#SBATCH --nodes=1
