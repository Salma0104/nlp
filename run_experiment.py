import argparse
from assignment import CustomSciGenTrainer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='processed_data')
parser.add_argument('--access_token',type=str)
args = parser.parse_args()

t5_small = CustomSciGenTrainer("seq2seq","google/flan-t5-small",args.access_token,args.dataset_path,"few_shot-50",384,130,4)
t5_small.train(5e-5,30)

t5_large = CustomSciGenTrainer("seq2seq","google/flan-t5-large",args.access_token,args.dataset_path,"few_shot-50",384,130,4)
t5_large.train(5e-5,30)

gemma_2b = CustomSciGenTrainer("causal","google/gemma-2b",args.access_token,args.dataset_path,"few_shot",384,130,4)
gemma_2b.train(5e-5,30)

tran_x5 = CustomSciGenTrainer("causal","Rocketknight1/falcon-rw-1b",args.access_token,args.dataset_path,"few_shot",384,130,4)
tran_x5.train(5e-5,30)