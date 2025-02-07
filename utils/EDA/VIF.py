import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

def calculate_vif(features_df):
    """Calculate VIF for each feature"""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features_df.columns
    vif_data["VIF"] = [variance_inflation_factor(features_df.values, i) 
                       for i in range(features_df.shape[1])]
    
    # Sort by VIF value in descending order
    vif_data = vif_data.sort_values('VIF', ascending=False)
    return vif_data

def plot_vif(vif_data):
    """Create a horizontal bar plot of VIF values"""
    plt.figure(figsize=(12, max(8, len(vif_data) * 0.3)))
    
    # Create bar plot
    sns.barplot(data=vif_data, y='Feature', x='VIF')
    
    # Add title and labels
    plt.title('Variance Inflation Factors', pad=15)
    plt.xlabel('VIF')
    
    # Add a vertical line at VIF = 5 (common threshold)
    plt.axvline(x=5, color='r', linestyle='--', alpha=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('figures/vif_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    # Create directory for plots if it doesn't exist
    os.makedirs('figures', exist_ok=True)

    # Read the CSV file
    df = pd.read_csv('results/feature_exports/features_A.csv')

    # Drop PID, Gender and Score columns
    features_df = df.drop(['PID', 'Gender', 'Score'], axis=1)

    # Set the style for all plots
    plt.style.use('seaborn-v0_8')

    # Calculate VIF
    vif_data = calculate_vif(features_df)
    
    # Display VIF values
    print("\nVariance Inflation Factors:")
    print(vif_data.to_string(index=False))
    
    # Save VIF data to CSV
    vif_data.to_csv('results/vif_values.csv', index=False)
    
    # Create VIF plot
    plot_vif(vif_data)