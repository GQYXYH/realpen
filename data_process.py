import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast

# Log in to wandb
wandb.login()

# Initialize a new run
run = wandb.init(entity="2017920898", \
        project="penenvcom",
        name="2017920898_pen_seed"+str(1))
data = pd.read_csv('reward.csv')  # Adjust the path to your CSV file
communication_savings = data['Comsavings'].tolist()  # Ensure the column name matches your CSV file

# Read and preprocess data
total_steps = 0 
for epoch in range(50):
    
    for t in range(1000): 
        print("1")
    total_steps += t
    wandb.log({"Episode Reward": communication_savings[epoch]}, step=total_steps)


# Finish the run
wandb.finish()

