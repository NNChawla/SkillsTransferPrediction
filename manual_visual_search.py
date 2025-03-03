import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

combo_size = 4
results_A_tag = 'v3_A_manual_True_classification_lightgbm_False_0_5_None'
results_B_tag = results_A_tag.replace('_A_', '_B_')
results_A_B_tag = results_A_tag.replace('_A_', '_A_B_')
figures_A_tag = results_A_tag.replace('_manual_True_classification_', '_').replace('_0_5_None', f'_{combo_size}')
figures_B_tag = figures_A_tag.replace('_A_', '_B_')
figures_A_B_tag = figures_A_tag.replace('_A_', '_A_B_')

results_directory_A = f'./results/{results_A_tag}'
results_directory_B = f'./results/{results_B_tag}'

results_A = pd.read_csv(f'{results_directory_A}/manual_results.csv')
results_B = pd.read_csv(f'{results_directory_B}/manual_results.csv')

columns = ['feature_key', 'specificity', 'sensitivity', 'feature_selection_measure']

results_A = results_A[columns]
results_B = results_B[columns]

results = results_A.merge(results_B, how='inner', on='feature_key', suffixes=['_A', '_B'])
results.rename(columns={"feature_selection_measure_A": "mcc_A", "feature_selection_measure_B": "mcc_B"}, inplace=True)
results = results[['feature_key', 'specificity_A', 'specificity_B', 'sensitivity_A', 'sensitivity_B', 'mcc_A', 'mcc_B']]

# Optimizing for 0 class
results.sort_values(by=['specificity_A', 'specificity_B'], ascending=[False, False], inplace=True)

os.makedirs(f'./results/{results_A_B_tag}', exist_ok=True)
results.to_csv(f'./results/{results_A_B_tag}/manual_results_sorted_{combo_size}.csv', index=False)

# figures_directory_A = f'./figures/confusion_matrices/{figures_A_tag}'
# figures_directory_B = f'./figures/confusion_matrices/{figures_B_tag}'

# figure_names = os.listdir(figures_directory_A)
# figure_path_pairs = [[f'{figures_directory_A}/{figure_name}', f'{figures_directory_B}/{figure_name}'] for figure_name in figure_names]
# figure_path_pairs = [[i[0], i[1].replace('a_b_Time', 'b_b_Time')] if ('a_b_Time' in i[0]) else i for i in figure_path_pairs]

# figure_output_directory = f'./figures/confusion_matrices/{figures_A_B_tag}'
# os.makedirs(figure_output_directory, exist_ok=True)

# for figure_pair in figure_path_pairs:
#     # Create figure with two subplots side by side
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
#     # Load and display images
#     img_A = mpimg.imread(figure_pair[0])
#     img_B = mpimg.imread(figure_pair[1])
    
#     ax1.imshow(img_A)
#     ax2.imshow(img_B)
    
#     # Remove axes
#     ax1.axis('off')
#     ax2.axis('off')
    
#     # Set titles
#     ax1.set_title('Dataset A')
#     ax2.set_title('Dataset B')
    
#     # Get filename from path and save combined figure
#     filename = os.path.basename(figure_pair[0])
#     plt.savefig(f'{figure_output_directory}/{filename}', bbox_inches='tight', dpi=300)
#     plt.close()

# # Create a combined figure showing all confusion matrices
# output_files = sorted(os.listdir(figure_output_directory))
# n_files = len(output_files)
# n_cols = 3
# n_rows = (n_files + n_cols - 1) // n_cols  # Ceiling division to handle cases not divisible by 3

# plt.figure(figsize=(15, 5*n_rows))

# for idx, filename in enumerate(output_files):
#     plt.subplot(n_rows, n_cols, idx + 1)
#     img = mpimg.imread(f'{figure_output_directory}/{filename}')
#     plt.imshow(img)
#     plt.axis('off')
#     plt.title(f"{filename.strip('.png')}")

# plt.tight_layout()
# plt.savefig(f'{figure_output_directory}/_all_features_combined.png', bbox_inches='tight', dpi=100)
# plt.close()