from datasets import load_dataset
from transformers import AutoTokenizer,Seq2SeqTrainer, Seq2SeqTrainingArguments,AutoModelForSeq2SeqLM,DataCollatorForSeq2Seq
import torch
import transformers
import pandas as pd
import evaluate
import numpy as np
import warnings
warnings.filterwarnings('ignore')
transformers.set_seed(2002)

def generate(text,model,tokenizer):
  model.eval()
  input_ids = tokenizer(text,return_tensors="pt",padding=True,truncation=True)["input_ids"]
  input_ids = input_ids.to(model.device)
  outputs = model.generate(
    input_ids,
    max_length=512,
    early_stopping=True,
    use_cache=True,
    num_beams=5,
    )
  return tokenizer.decode(outputs[0], skip_special_tokens=True)


# load data
data_files = {"train":"train.csv","val":"dev.csv","test":"test.csv"}
datasets = load_dataset(path='./processed_data', data_files=data_files)

# get tokenizer and model
access_token = "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"
checkpoint = "google/flan-t5-small"
# checkpoint = "facebook/bart-large"

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,token=access_token)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint,token=access_token)

# tokenize data (train + val)
max_target_length = 384
max_input_length = 384

def preprocess_function(example):
  model_inputs = tokenizer(example["source"], max_length=max_input_length,padding=True,truncation=True)

  labels = tokenizer(example["target"], max_length=max_target_length,padding=True,truncation=True)
  model_inputs["labels"] = labels["input_ids"]

  return model_inputs

tokenized_datasets = datasets.map(preprocess_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns("source")
tokenized_datasets = tokenized_datasets.remove_columns("target")
data_collator = DataCollatorForSeq2Seq(tokenizer,model=model)


# training
batch_size = 4
num_train_epochs = 5
steps = len(tokenized_datasets["train"]) // batch_size
name = checkpoint.split('/')[-1]

args = Seq2SeqTrainingArguments(
    output_dir=f"{name}",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    optim="adamw_torch",
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    save_strategy = 'epoch',
    logging_steps=steps,
    push_to_hub=False,
    generation_max_length=max_target_length,
    load_best_model_at_end=True
) 


def compute_metrics(eval_preds):
  metric = evaluate.load("sacrebleu")
  predictions, labels = eval_preds
  predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
  labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
  # convert predictions back into text
  decoded_preds = tokenizer.batch_decode(predictions,skip_special_tokens=True)
  decoded_labels = tokenizer.batch_decode(labels,skip_special_tokens=True)
  decoded_preds = [pred.strip() for pred in decoded_preds]
  decoded_labels = [[label.strip()] for label in decoded_labels]

  # compute score
  result = metric.compute(predictions=decoded_preds,references=decoded_labels)
  return {"bleu" : result["score"]}

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["val"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
# print(trainer.evaluate())

print()
print(generate(datasets["test"]["source"][0],model,tokenizer))
print()
print(datasets["test"]["target"][0])