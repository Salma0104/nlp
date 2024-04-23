import os
import evaluate
from nltk.translate import meteor_score
from datasets import load_dataset
import numpy as np

def evaluate_preds(predictions,references):
  # compute sacrebleu, meteor + mover score
  met_score = [meteor_score.single_meteor_score(p,r,alpha=0.9, beta=3, gamma=0.5) for p,r in zip(predictions,references)]
  scb_score = scb.compute(predictions = predictions,references = references)
  


scb = evaluate.load("sacrebleu")
data_files = {"test":"test.csv"}
test_data = load_dataset(path='./processed_data', data_files=data_files)["test"]
references = [[ref] for ref in test_data["target"]]

for file in os.listdir("./inference_results"):
  predictions = np.loadtxt(f'./training-data/{file}', dtype=str)
  evaluate_preds(predictions,references)
