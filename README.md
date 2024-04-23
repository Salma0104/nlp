# Instructions to run code and replicate results

## Set up environment
- If using HPC, load anaconda module
- Create new conda environment using the command `conda create --name nlp_env python==3.10`
- Activate your environment by running `source activate nlp_env` if using HPC and `conda activate nlp_env` otherwise
- Install the required libraries and packages by running `pip install -r requirements.txt`

## Set up data
- Clone the SciGen repository by running `git clone https://github.com/UKPLab/SciGen.git`
- Generate the required datasets by running `python preprocess_data.py`. This should create a directory called processed_data that contains train, dev and test data

## Get access to models
- Create an account on hugging face
- Generate your access token by using this link [access token page](https://huggingface.co/settings/tokens)
- Request access to google/gemma by filling out the form here [gemma consent form](https://www.kaggle.com/models/google/gemma/license/consent)

## Train Models
- To train the 4 models Gemma, T5-Large, T-Small and Falcon, run the command `python run_experiment.py --access_token={YOUR ACCESS TOKEN}`. You can comment out any models you dont want to train
- Be aware that model checkpoints will be stored locally and can take up quite a bit of space (~18GB)
- Takes ~5-6 Hours (stanage)

## Plot Train data
- To make plots of how your models performed during training run the command `python plot_train_data.py`. This should generate a plot of the loss and the sacrebleu score during training

## Run inference on all models
- To run inference and create text files containing predictions, run the command `python inference.py --access_token={YOUR ACCESS TOKEN}`.
- Takes ~2:30 hours (stanage)

## Evaluate models
- To evaluate the performance of the models run the command `python evaluate_models.py --access_token={YOUR ACCESS TOKEN}`. This takes ~5-10 minutes to run and shows you the meteor, sacrebleu and mover score of the predictions against the references (test data)

# Results
## Training results (30 epochs on few shot ds)
<img src=https://github.com/Salma0104/nlp/assets/100161578/7d9e86c9-491b-419e-a44d-a18df03c87a1 alt="Bleu Score" width="350"/>      <img src=https://github.com/Salma0104/nlp/assets/100161578/8404d238-aab2-4a28-a1fd-73a42e4a0d05 alt="Loss" width="350"/>

## Evaluation results
| Model Name    | Meteor Score | Sacrebleu Score | mean Mover Score | Median Mover Score |
|---------------|--------------|-----------------|------------------|--------------------|
| gemma-2b      | 0.078        | 1.368           | 0.508            | 0.506              |
| flan-t5-large | 0.255        | 1.859           | 0.520            | 0.514              |
| falcon-rw-1b  | 0.067        | 1.348           | 0.506            | 0.504              |
| flan-t5-small | 0.218        | 2.217           | 0.510            | 0.502              |



