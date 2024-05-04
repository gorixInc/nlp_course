#!/bin/bash -l
#SBATCH --job-name="nlp_hw_gordei"
#SBATCH --time 00:30:00 # set an appropriate amount of time for the job to run here in HH:MM:SS format
#SBATCH --partition=gpu # set the partition to gpu
#SBATCH --gres=gpu:tesla:1 # assign a single tesla gpu



# Here you need to run train.py with python from the virtual environment where you have all the dependencies install
# You also have to pass the command line args (such as dataset name) to the script here, as well
# You may use whichever virtual environment manager you prefer (conda, venv, etc.)

source activate nlp_course

cd ~/nlp_course/hw3

python train.py --num_train_epochs 100 --dataset_name conll2003 --label_column_name ner_tags --output_dir ./output --model_name_or_path distilbert-base-uncased

conda deactivate