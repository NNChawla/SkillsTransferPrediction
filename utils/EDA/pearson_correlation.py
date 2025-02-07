import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml

def get_feature_columns_from_config(config):
    """Extract feature columns from the config file"""
    feature_columns = []
    
    # Add all global features with their statistics
    for feature_name, feature_info in config['global_features'].items():
        base_feature = feature_info['features'][0]  # Get the base feature name
        for stat in feature_info['statistics']:
            feature_columns.append(f"{base_feature}_{stat}")
    
    return feature_columns

def create_correlation_matrix(features_df, feature_columns):
    """Create and save correlation plots in a more readable format"""
    # Add 'Score' to the feature columns if it exists
    all_columns = feature_columns + ['Score']
    
    # Filter DataFrame to only include specified features and Score
    features_df = features_df[all_columns]
    
    # Calculate the correlation matrix
    corr_matrix = features_df.corr(method='pearson')
    
    # Create and save heatmap
    plt.figure(figsize=(20, 16))
    
    # Create heatmap without annotations
    sns.heatmap(corr_matrix, 
                annot=False,  # Removed annotations
                cmap='coolwarm',
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                cbar_kws={'label': 'Correlation Coefficient'})
    
    plt.title('Feature Correlation Heatmap', pad=20, size=16)
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    plt.tight_layout()
    
    # Save the heatmap
    plt.savefig('figures/correlation_heatmap.png',
                dpi=300,
                bbox_inches='tight')
    plt.close()
    
    # For each feature, get its strongest correlations
    correlations_list = []
    for feature in corr_matrix.columns:
        # Get correlations for this feature, excluding self-correlation
        feature_corrs = corr_matrix[feature].drop(feature)
        
        # Skip if all correlations are NaN
        if feature_corrs.isna().all():
            continue
            
        # Find max correlation, ignoring NaN values
        max_abs_corr = feature_corrs.abs().max(skipna=True)
        if pd.isna(max_abs_corr):
            continue
            
        # Find the feature with maximum correlation
        max_corr_feature = feature_corrs.abs().idxmax(skipna=True)
        correlation_value = feature_corrs[max_corr_feature]
        
        correlations_list.append({
            'Feature': feature,
            'Strongest_Correlation': max_abs_corr,
            'Most_Correlated_With': max_corr_feature,
            'Correlation_Value': correlation_value
        })
    
    # Convert to DataFrame and sort by absolute correlation
    corr_summary = pd.DataFrame(correlations_list)
    corr_summary = corr_summary.sort_values('Strongest_Correlation', ascending=True)
    
    # Continue with existing bar plot
    plt.figure(figsize=(12, max(8, len(corr_summary) * 0.2)))
    
    # Create horizontal bar plot
    bars = plt.barh(y=range(len(corr_summary)), 
                   width=corr_summary['Correlation_Value'],
                   color=plt.cm.coolwarm(
                       (corr_summary['Correlation_Value'] + 1) / 2))
    
    # Add feature names
    plt.yticks(range(len(corr_summary)), 
              [f"{row.Feature}\n→ {row.Most_Correlated_With}" 
               for _, row in corr_summary.iterrows()],
              fontsize=8)
    
    # Add correlation values
    for i, v in enumerate(corr_summary['Correlation_Value']):
        plt.text(v + (0.01 if v >= 0 else -0.01), 
                i,
                f'{v:.2f}',
                va='center',
                ha='left' if v >= 0 else 'right',
                fontsize=8)
    
    # Add vertical line at x=0
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # Set limits and labels
    plt.xlim(-1.1, 1.1)
    plt.xlabel('Correlation Coefficient')
    plt.title('Strongest Feature Correlations', pad=15)
    
    # Add grid
    plt.grid(True, axis='x', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('figures/correlation_matrix.png', 
                dpi=300, 
                bbox_inches='tight')
    plt.close()
    
    return corr_matrix

if __name__ == '__main__':
    # Create directory for figures if it doesn't exist
    os.makedirs('figures', exist_ok=True)

    # Read the config file
    with open('experiment_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Get feature columns from config
    feature_columns = get_feature_columns_from_config(config)

    # Read the CSV file
    df = pd.read_csv('results/feature_exports/features_A.csv')

    # Set the style for all plots
    plt.style.use('seaborn-v0_8')

    # Create correlation matrix and heatmap
    corr_matrix = create_correlation_matrix(df, feature_columns)
    
    # Optionally, save correlation values to CSV
    corr_matrix.to_csv('results/correlation_matrix.csv')