import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def create_correlation_network(csv_path, threshold=0.7, figsize=(20, 20)):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Drop non-feature columns
    features_df = df.drop(['PID', 'Gender'], axis=1)
    
    # Calculate correlation matrix
    corr_matrix = features_df.corr()
    
    # Create network graph
    G = nx.Graph()
    
    # Add nodes
    for feature in features_df.columns:
        G.add_node(feature)
    
    # Add edges for correlations above threshold
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr = abs(corr_matrix.iloc[i, j])
            if corr > threshold:
                G.add_edge(corr_matrix.columns[i], 
                          corr_matrix.columns[j], 
                          weight=corr)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Calculate node sizes based on degree centrality
    centrality = nx.degree_centrality(G)
    node_sizes = [v * 3000 for v in centrality.values()]
    
    # Set up layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, 
                          node_size=node_sizes,
                          node_color='lightblue',
                          alpha=0.6)
    
    # Draw edges with varying thickness based on correlation strength
    edges = G.edges()
    weights = [G[u][v]['weight'] * 2 for u, v in edges]
    nx.draw_networkx_edges(G, pos, 
                          width=weights,
                          alpha=0.5)
    
    # Add labels with smaller font size
    nx.draw_networkx_labels(G, pos, 
                           font_size=8,
                           font_weight='bold')
    
    plt.title(f'Feature Correlation Network (|ρ| > {threshold})', 
              pad=20, 
              fontsize=16)
    plt.axis('off')
    
    # Save the plot
    plt.savefig('correlation_network.png', 
                dpi=300, 
                bbox_inches='tight')
    plt.close()

# Create the network visualization
create_correlation_network(
    'results/feature_exports/features_A_all_trackers_all_measurements_all_all_global.csv',
    threshold=0.9
) 