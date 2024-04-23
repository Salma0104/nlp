import os
import evaluate
from nltk.translate import meteor_score
from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer
import argparse
import transformers
import pandas as pd
transformers.set_seed(2002)

parser = argparse.ArgumentParser()
parser.add_argument('--access_token',type=str)
args = parser.parse_args()
#from moverscore_v2 import get_idf_dict, word_mover_score

def evaluate_preds(predictions,references,checkpoint):
  # compute sacrebleu, meteor + mover score
  tokenizer = AutoTokenizer.from_pretrained(checkpoint,token=args.access_token) 
  tokenized_refs = [tokenizer.tokenize(ref) for ref in references]
  tokenized_preds = [tokenizer.tokenize(pred) for pred in predictions]
  met_score = [meteor_score.single_meteor_score(p,r,alpha=0.9, beta=3, gamma=0.5) for p,r in zip(tokenized_preds,tokenized_refs)]
  scb_score = scb.compute(predictions = predictions,references = [[ref] for ref in references])

  # idf_dict_pred = get_idf_dict(predictions)
  # idf_dict_ref = get_idf_dict(references)
  wm_score = 0
  # wm_score = word_mover_score(references, predictions, idf_dict_ref, idf_dict_pred, \
  #                         stop_words=[], n_gram=1, remove_subwords=True, batch_size=64)
  return met_score,scb_score,wm_score



scb = evaluate.load("sacrebleu")
data_files = {"test":"test.csv"}
test_data = load_dataset(path='./processed_data', data_files=data_files)["test"]

for file in os.listdir("./inference_results"):
  predictions =pd.read_csv(f'./inference_results/{file}', header=None)[0].tolist()
  checkpoint = file
  if checkpoint != "falcon-rw-1b":
    checkpoint = f"google/{checkpoint}"
  else:
    checkpoint = f"Rocketknight1/{checkpoint}"
  print(len(predictions))
  print(predictions[0])
  #ms,scbs,wms = evaluate_preds(predictions,test_data["target"],checkpoint)
  #print(f"{file}: Meteor: {ms}, SacreBleu: {scbs}, Word Mover Score: {wms}")
