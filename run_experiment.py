import argparse
from assignment import CustomSciGenTrainer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='datasets/')
parser.add_argument('--access_token',type=str)
args = parser.parse_args()

t5_small = CustomSciGenTrainer("seq2seq","google/flan-t5-small",args.access_token,args.dataset_path,"few_shot",384,130,4)
t5_small.train(5e-5,1)
t5_small.inference()
#evaluate

# t5_small = CustomSciGenTrainer("seq2seq","google/flan-t5-large",args.access_token,args.dataset_path,384,130,4)
# t5_small.train(5e-5,1)