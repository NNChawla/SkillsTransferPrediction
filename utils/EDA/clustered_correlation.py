import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import os
import random

def create_clustered_correlation(csv_path, figsize=(20, 16), features_per_group=50, shuffle=False, random_seed=42):
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Drop non-feature columns
    features_df = df.drop(['PID', 'Gender'], axis=1)
    
    # Get list of all features
    features = list(features_df.columns)
    
    # Optionally shuffle features
    if shuffle:
        random.seed(random_seed)  # For reproducibility
        random.shuffle(features)
    
    # Split features into groups
    feature_groups = [features[i:i + features_per_group] 
                     for i in range(0, len(features), features_per_group)]
    
    # Process each group
    for group_idx, feature_group in enumerate(feature_groups):
        # Select subset of features
        subset_df = features_df[feature_group]
        
        # Calculate correlation matrix
        corr_matrix = subset_df.corr()
        
        # Handle any NaN values
        corr_matrix = corr_matrix.fillna(0)
        
        # Convert correlation matrix to distance matrix
        distance_matrix = 1 - np.abs(corr_matrix)
        
        # Convert to numpy array
        distance_matrix = distance_matrix.to_numpy()
        
        # Ensure the distance matrix is symmetric and non-negative
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        distance_matrix = np.maximum(distance_matrix, 0)
        
        # Make sure diagonal is exactly zero
        np.fill_diagonal(distance_matrix, 0)
        
        # Convert distance matrix to condensed form
        condensed_dist = squareform(distance_matrix)
        
        # Perform hierarchical clustering
        linkage = hierarchy.linkage(condensed_dist, method='complete')
        
        # Get the order of features for the clustered matrix
        cluster_order = hierarchy.dendrogram(linkage, no_plot=True)['leaves']
        
        # Reorder correlation matrix
        corr_clustered = corr_matrix.iloc[cluster_order, cluster_order]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                      gridspec_kw={'height_ratios': [0.3, 0.7]})
        
        # Plot dendrogram
        hierarchy.dendrogram(linkage, ax=ax1, leaf_rotation=90)
        ax1.set_title(f'Feature Clustering Dendrogram (Group {group_idx + 1})', pad=20)
        
        # Plot clustered heatmap
        sns.heatmap(corr_clustered, 
                    cmap='coolwarm',
                    center=0,
                    vmin=-1,
                    vmax=1,
                    square=True,
                    xticklabels=True,
                    yticklabels=True,
                    ax=ax2)
        
        # Rotate labels
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
        ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
        ax2.set_title(f'Clustered Correlation Matrix (Group {group_idx + 1})', pad=20)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f'figures/clustered_correlation_group_{group_idx + 1}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()

# Example usage with shuffling
create_clustered_correlation(
    'results/feature_exports/features_A_all_trackers_all_measurements_all_all_global.csv',
    features_per_group=50,
    shuffle=True,  # Enable shuffling
    random_seed=42  # Optional: set seed for reproducibility
) 