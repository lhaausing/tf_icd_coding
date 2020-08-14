#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu
#SBATCH --time=24:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=bert_icd_pred
#SBATCH --mail-type=END
#SBATCH --mail-user=xl3119@nyu.edu
#SBATCH --output=bert_icd_pred.txt

overlay_ext3=/scratch/xl3119/bert_icd_pred_env/overlay-10GB-400K.ext3

singularity \
    exec --nv --overlay $overlay_ext3:ro \
    /beegfs/work/public/singularity/centos-7.8.2003.sif  \
    /bin/bash -c "module load  anaconda3/5.3.1; \
                  module load cuda/10.1.105; \
                  module load gcc/6.3.0; \
                  source /share/apps/anaconda3/5.3.1/etc/profile.d/conda.sh; \
                  conda activate /ext3/cenv; \
                  python3 main_tmp.py "
