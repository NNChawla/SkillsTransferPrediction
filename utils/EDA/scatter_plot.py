import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm

def create_scatter_plot(feature):
    """Create a scatter plot for a single feature vs Score"""
    if feature == 'Score':
        return
        
    # Create a new figure
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot
    sns.scatterplot(data=features_df, 
                   x=feature,
                   y='Score',
                   alpha=0.6)
    
    # Add title and labels
    plt.title(f'{feature} vs Score', pad=15)
    plt.xlabel(feature)
    plt.ylabel('Score')
    
    # Add a trend line
    sns.regplot(data=features_df,
               x=feature,
               y='Score',
               scatter=False,
               color='red',
               line_kws={'linestyle': '--'})
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'figures/scatter_plots/{feature}_vs_score.png', 
                dpi=300, 
                bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    # Create directory for scatter plots if it doesn't exist
    os.makedirs('figures/scatter_plots', exist_ok=True)

    # Read the CSV file
    df = pd.read_csv('results/feature_exports/features_A.csv')

    # Drop PID and Gender columns
    features_df = df.drop(['PID', 'Gender'], axis=1)

    # Set the style for all plots
    plt.style.use('seaborn-v0_8')

    # Get features to plot (excluding Score)
    features_to_plot = [col for col in features_df.columns if col != 'Score']

    # Create pool of workers
    with Pool() as pool:
        # Create iterator with progress bar
        for _ in tqdm(
            pool.imap_unordered(create_scatter_plot, features_to_plot),
            total=len(features_to_plot),
            desc="Creating scatter plots"
        ):
            pass