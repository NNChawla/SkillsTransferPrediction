import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
from multiprocessing import Pool
from tqdm import tqdm

def apply_transformation(data, transform_type):
    """Apply the specified transformation to the data"""
    # Remove inf values and convert to nan
    data = pd.Series(np.nan_to_num(data, nan=np.nan, posinf=np.nan, neginf=np.nan), name=data.name)
    
    # Remove NaN values
    clean_data = data.dropna()
    
    # If no valid data points remain, return None
    if len(clean_data) == 0:
        return None, None

    try:
        if transform_type == 'log':
            # Add offset to make all values positive
            offset = abs(clean_data.min()) + 1 if clean_data.min() <= 0 else 0
            return np.log1p(clean_data + offset), f'log({data.name} + {offset:.2f})'
        
        elif transform_type == 'sqrt':
            # Add offset to make all values positive
            offset = abs(clean_data.min()) if clean_data.min() < 0 else 0
            return np.sqrt(clean_data + offset), f'sqrt({data.name} + {offset:.2f})'
        
        elif transform_type == 'boxcox':
            # Box-Cox requires strictly positive values
            offset = abs(clean_data.min()) + 1 if clean_data.min() <= 0 else 0
            transformed_data, _ = stats.boxcox(clean_data + offset)
            return transformed_data, f'boxcox({data.name} + {offset:.2f})'
        
        elif transform_type == 'yeojohnson':
            # Yeo-Johnson can handle negative values directly
            transformed_data, _ = stats.yeojohnson(clean_data)
            return transformed_data, f'yeojohnson({data.name})'
        
        else:
            raise ValueError(f"Unknown transformation type: {transform_type}")
            
    except Exception as e:
        print(f"Error transforming {data.name} with {transform_type}: {str(e)}")
        return None, None

def create_feature_plots(feature_and_transform):
    """Create histogram and box plot for a single feature with specified transformation"""
    feature, transform_type = feature_and_transform
    
    if feature == 'Score':
        return
        
    # Apply transformation first to check if it's possible
    transformed_data, transform_label = apply_transformation(features_df[feature], transform_type)
    
    # Skip if transformation failed
    if transformed_data is None:
        print(f"Skipping {feature} with {transform_type} due to transformation failure")
        return
    
    try:
        # Create a figure with four subplots in a 2x2 grid
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original distribution plots
        sns.histplot(data=features_df, x=feature, kde=True, ax=ax1)
        ax1.set_title(f'Original {feature} Distribution')
        ax1.set_xlabel(feature)
        ax1.set_ylabel('Count')
        
        sns.boxplot(data=features_df, y=feature, ax=ax2)
        ax2.set_title(f'Original {feature} Box Plot')
        ax2.set_ylabel(feature)
        
        # Transformed distribution plots
        sns.histplot(data=transformed_data, kde=True, ax=ax3)
        ax3.set_title(f'{transform_type.title()}-Transformed {feature} Distribution')
        ax3.set_xlabel(transform_label)
        ax3.set_ylabel('Count')
        
        sns.boxplot(y=transformed_data, ax=ax4)
        ax4.set_title(f'{transform_type.title()}-Transformed {feature} Box Plot')
        ax4.set_ylabel(transform_label)
        
        # Add overall title
        plt.suptitle(f'{feature} Analysis - Original vs {transform_type.title()}-Transformed', y=1.02)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f'figures/feature_plots/{feature}_{transform_type}_analysis.png', 
                    dpi=300, 
                    bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error plotting {feature} with {transform_type}: {str(e)}")
        plt.close()  # Ensure figure is closed even if error occurs

if __name__ == '__main__':
    # Create directory for feature plots if it doesn't exist
    os.makedirs('figures/feature_plots', exist_ok=True)

    # Read the CSV file
    df = pd.read_csv('results/feature_exports/features_A.csv')

    # Drop PID and Gender columns
    features_df = df.drop(['PID', 'Gender'], axis=1)

    # Set the style for all plots
    plt.style.use('seaborn-v0_8')

    # Available transformations
    transformations = ['log', 'sqrt', 'boxcox', 'yeojohnson']

    # Get features to plot (excluding Score)
    features_to_plot = [col for col in features_df.columns if col != 'Score']
    
    # Create list of feature-transformation pairs
    feature_transform_pairs = [(feature, transform) 
                             for feature in features_to_plot 
                             for transform in transformations]

    # Create pool of workers
    with Pool() as pool:
        # Create iterator with progress bar
        for _ in tqdm(
            pool.imap_unordered(create_feature_plots, feature_transform_pairs),
            total=len(feature_transform_pairs),
            desc="Creating feature plots"
        ):
            pass