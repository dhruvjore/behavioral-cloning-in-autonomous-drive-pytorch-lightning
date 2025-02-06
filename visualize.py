import torch  
from datamodule import DataModule  # Import custom DataModule for handling data loading  

import matplotlib.pyplot as plt  
import seaborn as sns  
import numpy as np  

# Initialize the DataModule, which handles dataset preparation and loading  
dm = DataModule()  
dm.prepare_data()  # Download or process data if needed  
dm.setup()  # Set up training, validation, and test datasets  

# Get the training dataloader to fetch batches of data  
train_dataloader = dm.train_dataloader()  

# Fix potential "too many open files" error in PyTorch multiprocessing  
torch.multiprocessing.set_sharing_strategy('file_system')  

# Initialize an empty list to store steering angles from the dataset  
steering_angles = []  

# Iterate over batches of images and steering angles from the training data  
for batch_idx, (image, steering_angle) in enumerate(train_dataloader):  
    steering_angles.append(steering_angle)  # Collect steering angles  

# Set the visualization theme and ensure reproducibility  
sns.set_theme(); np.random.seed(0)  

# Convert collected steering angles to a NumPy array for easy manipulation  
steering_angles = np.array(steering_angles)  

# Plot a histogram to visualize the distribution of steering angles  
ax = sns.histplot(steering_angles)  

# Label the axes for better readability  
plt.xlabel("Steering Angle")  
plt.ylabel("Count")  

# Display the plot  
plt.show()  
