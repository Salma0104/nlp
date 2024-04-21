from datasets import load_dataset
from transformers import (
  AutoTokenizer,
  AutoModelForSeq2SeqLM,
  DataCollatorForSeq2Seq,
  DataCollatorForLanguageModeling,
  AutoModelForCausalLM,
  get_scheduler,
  pipeline,
  LlamaForCausalLM,
  LlamaTokenizer,

)
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
import transformers
import pandas as pd
import evaluate
import numpy as np
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
import os

warnings.filterwarnings('ignore')
transformers.set_seed(2002)

class CustomSciGenTrainer:

  def __init__(self,mode,checkpoint,access_token,dataset_path,dataset_type,max_input_length,max_target_length,batch_size):
    self.mode = mode
    self.access_token = access_token
    self.dataset_path = dataset_path
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.checkpoint = checkpoint 


    if self.mode == "causal":
      self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint,token=self.access_token)
      self.tokenizer.padding_side = "right"
    else:
      self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint,token=self.access_token)
    self.tokenizer.pad_token = self.tokenizer.eos_token 
    self.max_input_length = max_input_length
    self.max_target_length = max_target_length
    self.batch_size = batch_size
    self.dataset_type = dataset_type
    self.model_name = self.checkpoint.split('/')[-1]

    # Make required directories
    directories = ['./plots','./models','./training-data',f'./models/{self.model_name}-{self.dataset_type}']
    for d in directories:
      if not Path(d).is_dir():
        os.makedirs(d)

  def load_dataset(self):
    data_files = {"train":"train.csv","val":"dev.csv"}
    datasets = load_dataset(path=self.dataset_path, data_files=data_files)
    return datasets
  
  def load_model(self):
    if self.mode == "seq2seq":
      model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint,token = self.access_token)
    elif self.mode == "causal":
      model = LlamaForCausalLM.from_pretrained(self.checkpoint,token = self.access_token,)
    return model
  
  def preprocess_function(self,example):
    model_inputs = self.tokenizer(example["source"], max_length=self.max_input_length,padding=True,truncation=True)

    labels = self.tokenizer(example["target"], max_length=self.max_target_length,padding=True,truncation=True)
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs
  
  def tokenize_dataset(self,datasets):
    tokenized_datasets = datasets.map(self.preprocess_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns("source")
    tokenized_datasets = tokenized_datasets.remove_columns("target")
    tokenized_datasets.set_format("torch")
    return tokenized_datasets
  
  def create_dataloaders(self,tokenized_datasets, data_collator):
    train_dl = DataLoader(
      tokenized_datasets["train"],
      shuffle = True,
      collate_fn = data_collator,
      batch_size = self.batch_size
    )
    val_dl = DataLoader(tokenized_datasets["val"], batch_size = self.batch_size//2)
    return train_dl,val_dl
  
  def plot(self,losses,scores,model_name):
    plt.figure("Training results", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(losses))]
    y = losses
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Bleu Scores")
    x = [len(losses)//len(scores) * (i + 1) for i in range(len(scores))]
    y = scores 
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.savefig(f'plots/{model_name}-{self.dataset_type}.png', bbox_inches='tight', pad_inches=0)

  def train(self,learning_rate,epochs):
    model = self.load_model()
    model.to(self.device)
    if self.mode == "seq2seq":
      data_collator = DataCollatorForSeq2Seq(self.tokenizer,model=model)
    else:
      data_collator = DataCollatorForLanguageModeling(self.tokenizer,mlm=False)
    
    datasets = self.load_dataset()
    tokenized_datasets = self.tokenize_dataset(datasets)
    train_dl,val_dl = self.create_dataloaders(tokenized_datasets,data_collator)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    metric = evaluate.load("sacrebleu")

    
    lr_scheduler = get_scheduler(
      "linear",
      optimizer = optimizer,
      num_warmup_steps = 0,
      num_training_steps = epochs*len(train_dl),
    )
    # Actual train loop
    losses = []
    metric_scores = []
    best_result = -1
    for epoch in range(epochs):
      model.train()
      epoch_loss = []
      for step, batch in enumerate(train_dl):
        batch = {k: v.to(self.device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        epoch_loss.append(loss.item())
        print(f'Training step: {step+1}/{len(train_dl)}, Loss: {loss:.4f}')
      
      model.eval()
      for step,batch in enumerate(val_dl):
        with torch.no_grad():
          generated_tokens = model.generate(
            batch["input_ids"].to(self.device),
            attention_mask = batch["attention_mask"].to(self.device),
            num_beams = 5,
            max_length = self.max_target_length,
            length_penalty = 5.0,
            early_stopping = True,
            use_cache = True
          )

          
          labels = batch["labels"]
          labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

          decoded_preds = self.tokenizer.batch_decode(generated_tokens,skip_special_tokens=True)
          decoded_labels = self.tokenizer.batch_decode(labels,skip_special_tokens=True)
          decoded_preds = [pred.strip() for pred in decoded_preds]
          decoded_labels = [[label.strip()] for label in decoded_labels]

          metric.add_batch(predictions=decoded_preds,references=decoded_labels)
        print(f'Validation step: {step+1}/{len(val_dl)}')

      result = metric.compute()
      # save best version of model
      if result["score"] > best_result:
        model.save_pretrained(f'./models/{self.model_name}-{self.dataset_type}')
        best_result = result["score"]


      metric_scores.append(result["score"])
      mean_loss = np.mean(epoch_loss)
      losses.append(mean_loss)
      print(f'Epoch: {epoch}, mean loss:{mean_loss:.4f}, bleu_score: {result["score"]:.4f}')
    
    # plot metric scores and losses
    self.plot(losses,metric_scores,self.model_name)
    arr = np.array(list(zip(losses,metric_scores)))
    np.savetxt(f'./training-data/{self.model_name}-{self.dataset_type}',arr)
  
  # def inference(self):
  #   # model = AutoModelForSeq2SeqLM.from_pretrained(f'.models/{self.model_name}-{self.dataset_type}')
  #   tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
  #   summerizer = pipeline("summarization", f'./models/{self.model_name}-{self.dataset_type}',tokenizer=tokenizer,max_length=self.max_target_length)
  #   print("Summerization Inference \n")
  #   print(summerizer("<R> <C> Model <C> SVQA <C> TGIF-QA (*) Action <C> TGIF-QA (*) Trans.")["summary_text"])