import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

def calculate_mutual_information_binary(features_df):
    """Calculate mutual information between features and binary Score (0 vs non-0)"""
    # Separate features and target
    X = features_df.drop('Score', axis=1)
    y = features_df['Score'].apply(lambda x: '0' if x == 0 else 'Non-0')
    
    # Calculate mutual information using classifier version
    mi_scores = mutual_info_classif(X, y)
    
    # Create DataFrame with feature names and MI scores
    mi_df = pd.DataFrame({
        'Feature': X.columns,
        'Mutual Information': mi_scores
    }).sort_values('Mutual Information', ascending=False)
    
    # Create plot
    plt.figure(figsize=(12, max(8, len(mi_df) * 0.3)))
    sns.barplot(data=mi_df,
                y='Feature',
                x='Mutual Information',
                color='skyblue')
    
    plt.title('Mutual Information with Binary Score (0 vs Non-0)', pad=15)
    plt.xlabel('Mutual Information Score')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('figures/mutual_information_binary.png', 
                dpi=300, 
                bbox_inches='tight')
    plt.close()
    
    return mi_df

def calculate_mutual_information_regression(features_df):
    """Calculate mutual information between features and Score"""
    # Separate features and target
    X = features_df.drop('Score', axis=1)
    y = features_df['Score']
    
    # Calculate mutual information
    mi_scores = mutual_info_regression(X, y)
    
    # Create DataFrame with feature names and MI scores
    mi_df = pd.DataFrame({
        'Feature': X.columns,
        'Mutual Information': mi_scores
    }).sort_values('Mutual Information', ascending=False)
    
    return mi_df

def plot_mutual_information(mi_df):
    """Create a horizontal bar plot of mutual information scores"""
    plt.figure(figsize=(12, max(8, len(mi_df) * 0.3)))
    
    # Create horizontal bar plot
    sns.barplot(data=mi_df,
                y='Feature',
                x='Mutual Information',
                color='skyblue')
    
    # Add title and labels
    plt.title('Mutual Information with Score', pad=15)
    plt.xlabel('Mutual Information Score')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('figures/mutual_information.png', 
                dpi=300, 
                bbox_inches='tight')
    plt.close()

def calculate_and_plot_nonzero_mi(features_df):
    """Calculate and plot mutual information for non-zero scores only"""
    # Filter out zero scores
    nonzero_df = features_df[features_df['Score'] != 0].copy()
    
    # Calculate MI for regression on non-zero scores
    X = nonzero_df.drop('Score', axis=1)
    y = nonzero_df['Score']
    
    # Calculate mutual information
    mi_scores = mutual_info_regression(X, y)
    
    # Create DataFrame
    mi_df = pd.DataFrame({
        'Feature': X.columns,
        'Mutual Information': mi_scores
    }).sort_values('Mutual Information', ascending=False)
    
    # Create plot
    plt.figure(figsize=(12, max(8, len(mi_df) * 0.3)))
    sns.barplot(data=mi_df,
                y='Feature',
                x='Mutual Information',
                color='lightgreen')  # Different color to distinguish from binary classification
    
    plt.title('Mutual Information with Non-Zero Scores', pad=15)
    plt.xlabel('Mutual Information Score')
    plt.tight_layout()
    
    # Save the plot and data
    plt.savefig('figures/mutual_information_nonzero.png', 
                dpi=300, 
                bbox_inches='tight')
    plt.close()
    
    return mi_df

if __name__ == '__main__':
    # Create directory for plots if it doesn't exist
    os.makedirs('figures', exist_ok=True)

    # Read the CSV file
    df = pd.read_csv('results/feature_exports/features_A.csv')

    # Drop PID and Gender columns
    features_df = df.drop(['PID', 'Gender'], axis=1)

    # Set the style for all plots
    plt.style.use('seaborn-v0_8')

    # Calculate mutual information for regression (all scores)
    mi_df = calculate_mutual_information_regression(features_df)
    mi_df.to_csv('results/mutual_information_scores.csv', index=False)
    plot_mutual_information(mi_df)
    
    # Calculate and plot binary classification MI
    mi_binary_df = calculate_mutual_information_binary(features_df)
    mi_binary_df.to_csv('results/mutual_information_scores_binary.csv', index=False)
    
    # Calculate and plot MI for non-zero scores (regression)
    mi_nonzero_df = calculate_and_plot_nonzero_mi(features_df)
    mi_nonzero_df.to_csv('results/mutual_information_scores_nonzero.csv', index=False)