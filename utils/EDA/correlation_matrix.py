import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('results/feature_exports/features_A_all_trackers_all_measurements_all_all_global.csv')

# Drop PID and Gender columns
features_df = df.drop(['PID', 'Gender'], axis=1)

# Calculate correlation matrix
corr_matrix = features_df.corr()

# Create a figure with a larger size
plt.figure(figsize=(20, 16))

# Create heatmap
sns.heatmap(corr_matrix, 
            cmap='coolwarm',  # Red-blue diverging colormap
            center=0,         # Center the colormap at 0
            vmin=-1,         # Minimum correlation value
            vmax=1,          # Maximum correlation value
            square=True,     # Make the plot square-shaped
            xticklabels=True,
            yticklabels=True)

# Rotate x-axis labels for better readability
plt.xticks(rotation=90)
plt.yticks(rotation=0)

# Add title
plt.title('Feature Correlation Matrix', pad=5)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()