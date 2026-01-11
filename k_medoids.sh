#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --partition=EPYC
#SBATCH --job-name=UL_Project
#SBATCH -A dssc

##SBATCH --mem=20G

# bash k_medoids.sh "T_2m rh_2m p_sfc dif_precip_g" "2001-01-01 2023-12-31" 10
# bash k_medoids.sh "T_2m rh_2m dif_precip_g" "2001-01-01 2023-12-31" 10
# sbatch k_medoids.sh "dif_precip_g" "2001-01-01 2023-12-31" 10
# srun --cpus-per-task=8 python k_medoids.py "$files" "$vars" "$train_date" $n_clusters $SLURM_JOB_ID

eval "$(/u/dssc/mzampar/miniconda3/bin/conda shell.bash hook)"

conda init

in_dir="/u/dssc/mzampar/scratch/Nausica"
out_dir="/u/dssc/mzampar/UL_Project/results"


vars=$1
train_date=$2
n_clusters=$3

# Convert vars into an array
patterns=($vars)

# Collect all files in the input directory
files=($(find "$in_dir" -type f ! -name '*.npz'))

# Initialize filtered files array
filtered_files=()

# Filter files using patterns
for f in "${files[@]}"; do
    for p in "${patterns[@]}"; do
        if [[ "$f" =~ $p ]]; then
            filtered_files+=("$f")
            break  # stop checking other patterns if one matches
        fi
    done
done

echo "${filtered_files[@]}"

sorted=$(echo $vars | sort)
vars=$sorted
sorted=$(echo "${filtered_files[@]}" | sort)
files=$sorted

python k_medoids.py "$files" "$vars" "$train_date" $n_clusters $SLURM_JOB_ID


