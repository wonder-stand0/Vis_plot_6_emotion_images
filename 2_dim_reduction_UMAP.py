import numpy as np
import pandas as pd
import umap

# Load embeddings, labels, and filenames
embeddings = np.load('embeddings.npy')
labels = np.load('labels.npy')
filenames = np.load('filenames.npy') # Load filenames

# UMAP dimensionality reduction
# Parameters from 2A_dim_reduction_comparison.py
reducer_umap = umap.UMAP(n_components=2,
                         random_state=42,
                         n_neighbors=50,
                         min_dist=0.15)
embeddings_2d_umap = reducer_umap.fit_transform(embeddings)

# Save the 2D UMAP coordinates
np.save('embeddings_2d.npy', embeddings_2d_umap)
print('Saved embeddings_2d_umap.npy (2D UMAP reduced features)')

# Save as CSV file for visualization in d3.js or other tools
csv_path_umap = 'embeddings_2d.csv'
df_umap = pd.DataFrame(embeddings_2d_umap, columns=['x', 'y'])
df_umap['label'] = labels
df_umap['filename'] = filenames # Add filename column
df_umap.to_csv(csv_path_umap, index=False)
print(f'Saved {csv_path_umap}')
