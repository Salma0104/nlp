import os
from pathlib import Path
import numpy as np
import pandas as pd


def load_file(file_path):
  with open(file_path,'r') as f:
    lines = f.readlines()
  return lines


#Assume you have already cloned ScieGen in this directory
base_path = './SciGen/baselines'

# Execute convert to json command for train, dev, and test
for folder,file_name in [["train","train"],["development","dev"],["test","test-Other"]]:
  if folder == "test":
    os.system(f'python {base_path}/convert_json_files.py -f {base_path}/../dataset/{folder}/{file_name}.json -s test')
  else:
    os.system(f'python {base_path}/convert_json_files.py -f {base_path}/../dataset/{folder}/few-shot/{file_name}.json -s {file_name}')

#make directory called converted_data
if not Path(f'{base_path}/converted_data').is_dir():
    os.makedirs(f'{base_path}/converted_data')

# move everything to converted_data directory
for split in ["train","dev","test"]:
  os.system(f'mv {split}.* {base_path}/converted_data')


#make directory called processed_data
if not Path('processed_data').is_dir():
    os.makedirs('processed_data')

# read in file and convert to csv and save in processed_data dir
for split in ["train","dev","test"]:
  file_path = f'{base_path}/converted_data/{split}'

  source,target = load_file(f'{file_path}.source'),load_file(f'{file_path}.target')
  df = pd.DataFrame({'source': source, 'target':target})
  df.to_csv(f'./processed_data/{split}.csv',index=False,encoding='utf-8')
