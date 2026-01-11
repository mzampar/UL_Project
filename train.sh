#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --partition=GPU
#SBATCH --mem=100gb
#SBATCH --job-name=train_ae
#SBATCH -A dssc
#SBATCH --output=UL_ae_%j.out

# echo "Starting job $SLURM_JOB_ID"

# sbatch train.sh "T_2m rh_2m p_sfc dif_precip_g" "2001-01-01 2017-12-31" "2018-01-01 2023-12-31" 5
# sbatch train.sh "dif_precip_g rh_2m T_2m" "2001-01-01 2017-12-31" "2018-01-01 2023-12-31" 5
# sbatch train.sh "dif_precip_g" "2001-01-01 2017-12-31" "2018-01-01 2023-12-31" 5

eval "$(/u/dssc/mzampar/miniconda3/bin/conda shell.bash hook)"

conda init

in_dir="/u/dssc/mzampar/scratch/Nausica"
out_dir="/u/dssc/mzampar/UL_Project/results"

vars=$1
train_date=$2
test_date=$3
n_clusters=$4

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

python main.py "$files" "$vars" "$train_date" "$test_date" $n_clusters $SLURM_JOB_ID


