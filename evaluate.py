import os
import evaluate
from nltk.translate import meteor_score
from datasets import load_dataset
import numpy as np
from moverscore_v2 import get_idf_dict, word_mover_score

def evaluate_preds(predictions,references):
  # compute sacrebleu, meteor + mover score
  met_score = [meteor_score.single_meteor_score(p,r,alpha=0.9, beta=3, gamma=0.5) for p,r in zip(predictions,references)]
  scb_score = scb.compute(predictions = predictions,references = [[ref] for ref in references])

  idf_dict_pred = get_idf_dict(predictions)
  idf_dict_ref = get_idf_dict(references)
  wm_score = word_mover_score(references, predictions, idf_dict_ref, idf_dict_pred, \
                          stop_words=[], n_gram=1, remove_subwords=True, batch_size=64)
  return met_score,scb_score,wm_score



scb = evaluate.load("sacrebleu")
data_files = {"test":"test.csv"}
test_data = load_dataset(path='./processed_data', data_files=data_files)["test"]

for file in os.listdir("./inference_results"):
  predictions = np.loadtxt(f'./training-data/{file}', dtype=str)
  ms,scbs,wms = evaluate_preds(predictions,test_data["target"])
  print(f"{file}: Meteor: {ms}, SacreBleu: {scbs}, Word Mover Score: {wms}")
