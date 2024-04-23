from datasets import load_dataset
from transformers import (
  AutoTokenizer,
  AutoModelForSeq2SeqLM,
  AutoModelForCausalLM,
)
import argparse
import torch
import transformers
import numpy as np
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
import os
warnings.filterwarnings('ignore')
transformers.set_seed(2002)

parser = argparse.ArgumentParser()
parser.add_argument('--access_token',type=str)
args = parser.parse_args()

#make results dir
if not Path("./inference_results").is_dir():
  os.makedirs("./inference_results")

def run_inference(model,model_name,tokenizer,data,device):
  model.to(device)
  predictions = []

  model.eval()
  for i in range(len(data)):
    print(f'{i}/{len(data)}')
    prompt = tokenizer(data["source"][i],max_length = 384,return_tensors="pt",padding=True,truncation=True)
    with torch.no_grad():
      outputs = model.generate(
        prompt["input_ids"].to(device),
        attention_mask = prompt["attention_mask"].to(device),
        max_new_tokens = 130,
        early_stopping = True,
        use_cache = True,
        num_beams = 5
      )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predictions.append(summary)
  np.savetxt(f'./inference_results/{model_name}',predictions,fmt='%s')

data_files = {"test":"test.csv"}
test_data = load_dataset(path='./processed_data', data_files=data_files)
ed_models = {"google/flan-t5-small","google/flan-t5-large"}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model_path in os.listdir("./models"):
  checkpoint = model_path.replace("-few_shot","")
  # this line no longer needed
  checkpoint = checkpoint.replace("-50","")

  if checkpoint != "falcon-rw-1b":
    checkpoint = f"google/{checkpoint}"
  else:
    checkpoint = f"Rocketknight1/{checkpoint}"

  
  tokenizer = AutoTokenizer.from_pretrained(checkpoint,token=args.access_token)


  if checkpoint not in ed_models:
    model = AutoModelForCausalLM.from_pretrained(f'./models/{model_path}',token = args.access_token)
    tokenizer.padding_side = "left"
  else:
    model = AutoModelForSeq2SeqLM.from_pretrained(f'./models/{model_path}',token = args.access_token)
  
  if tokenizer.pad_token == None:
    tokenizer.pad_token = tokenizer.eos_token 
  
  model_name = checkpoint.split("/")[-1]
  run_inference(model,model_name,tokenizer,test_data["test"],device)
