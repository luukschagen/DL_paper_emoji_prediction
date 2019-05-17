# DL_paper_emoji_prediction
Code supporting the paper submission for the Deep Learning course at JADS.


## Data source & cleaning
The original raw data used for this paper is taken from: https://archive.org/details/archiveteam-twitter-stream-2018-10

The Data cleaning script can be run on unzipped .tar filed for the first two weeks, in order to regenerate the data. This should generate a 'pickled' dataset which can be unpickled in the bert_classification script.

## Training script

bert_classification.py is the original training script, which upsamples the dataset to balance, and trains for 1 epoch, which takes over 5 hours on a cloud GPU. After it turned out that this very heavily overfitted, bert_classification_batches.py was created, which separates the data into batches, and trains on each one, before evaluating, and prompting the user to train another batch. This script also returns a few pickled files, which can subsequently be used for evaluation.

## Model evaluation.

The evaluation notebook contains some evaluation statistics as well as experiments in generating predictions from the saved weights of the model. These prediction files were too large to upload to github, but can be provided on request.
