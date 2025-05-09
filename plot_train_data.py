import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np

losses = []
scores = []
names = []


def plot(data,names,yaxis_name):
  fig, ax = plt.subplots(figsize=(8, 8))
  colours = ['red','blue','green','orange']
  epochs = [i + 1 for i in range(len(data[0]))]
  ax.set_title(f"Change in {yaxis_name} over {len(data[0])} epochs")
  ax.set_ylabel(yaxis_name)
  ax.set_xlabel('Number of epochs')

  for i in range(len(names)):
    plt.grid(True)
    plt.plot(epochs,data[i],colours[i],linewidth=2,label=names[i])
  yaxis_name = yaxis_name.replace(" ","_")
  plt.legend()
  plt.savefig(f'./{yaxis_name}.png', bbox_inches='tight', pad_inches=0)


for file in os.listdir("./training-data"):
  name = file.replace("-few_shot","")
  name = name.replace("-50","")
  arr = np.loadtxt(f'./training-data/{file}',delimiter=" ", dtype=float)
  names.append(name)
  losses.append([loss for loss,score in arr])
  scores.append([score for loss,score in arr])

# Plot data
plot(losses,names,"Loss")
plot(scores,names,"Bleu Score")
