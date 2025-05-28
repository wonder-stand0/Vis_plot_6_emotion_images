import numpy as np
import pandas as pd
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# Load embeddings and labels
embeddings = np.load('embeddings.npy')
labels = np.load('labels.npy')
filenames = np.load('filenames.npy')

# Define a custom color palette based on unique labels, similar to 3_plot_scatter.py
unique_labels_arr = np.unique(labels)
num_unique_labels = len(unique_labels_arr)
# Use tab10 palette, ensuring enough colors if num_unique_labels > 10, though tab10 has 10.
# If more are needed, seaborn cycles them or one might choose a different palette.
palette = sns.color_palette('tab10', n_colors=num_unique_labels if num_unique_labels <= 10 else 10)
label_to_color_map = {label_val: palette[idx % len(palette)] for idx, label_val in enumerate(unique_labels_arr)}

# Define a function to draw confidence ellipses
def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# Create a figure to hold all subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 15))
fig.suptitle('Comparison of Dimensionality Reduction Techniques', fontsize=16)

# t-SNE
print("Running t-SNE...")
tsne = TSNE(n_components=2, 
            perplexity=40, 
            learning_rate=200, 
            max_iter=2000, 
            init='pca', 
            random_state=42, 
            n_jobs=-1)
embeddings_tsne = tsne.fit_transform(embeddings)
df_tsne = pd.DataFrame(embeddings_tsne, columns=['x', 'y'])
df_tsne['label'] = labels
sns.scatterplot(data=df_tsne, x='x', y='y', hue='label', ax=axs[0, 0], palette=palette, hue_order=unique_labels_arr, legend=True)
for label_val in unique_labels_arr:
    subset = df_tsne[df_tsne['label'] == label_val]
    if not subset.empty and len(subset['x']) > 2 and len(subset['y']) > 2:
        current_color = label_to_color_map[label_val]
        confidence_ellipse(subset['x'], subset['y'], axs[0, 0],
                           facecolor=current_color,
                           alpha=0.05,
                           edgecolor=current_color)
axs[0, 0].set_title('t-SNE')
axs[0, 0].set_xlabel('')
axs[0, 0].set_ylabel('')

# PCA
print("Running PCA...")
pca = PCA(n_components=2, random_state=42)
embeddings_pca = pca.fit_transform(embeddings)
df_pca = pd.DataFrame(embeddings_pca, columns=['x', 'y'])
df_pca['label'] = labels
sns.scatterplot(data=df_pca, x='x', y='y', hue='label', ax=axs[0, 1], palette=palette, hue_order=unique_labels_arr, legend=False)
for label_val in unique_labels_arr:
    subset = df_pca[df_pca['label'] == label_val]
    if not subset.empty and len(subset['x']) > 2 and len(subset['y']) > 2:
        current_color = label_to_color_map[label_val]
        confidence_ellipse(subset['x'], subset['y'], axs[0, 1],
                           facecolor=current_color,
                           alpha=0.05,
                           edgecolor=current_color)
axs[0, 1].set_title('PCA')
axs[0, 1].set_xlabel('')
axs[0, 1].set_ylabel('')

# UMAP
print("Running UMAP...")
reducer_umap = umap.UMAP(n_components=2, 
                         random_state=42, 
                         n_neighbors=50, 
                         min_dist=0.15)
embeddings_umap = reducer_umap.fit_transform(embeddings)
df_umap = pd.DataFrame(embeddings_umap, columns=['x', 'y'])
df_umap['label'] = labels
sns.scatterplot(data=df_umap, x='x', y='y', hue='label', ax=axs[1, 0], palette=palette, hue_order=unique_labels_arr, legend=False)
for label_val in unique_labels_arr:
    subset = df_umap[df_umap['label'] == label_val]
    if not subset.empty and len(subset['x']) > 2 and len(subset['y']) > 2:
        current_color = label_to_color_map[label_val]
        confidence_ellipse(subset['x'], subset['y'], axs[1, 0],
                           facecolor=current_color,
                           alpha=0.05,
                           edgecolor=current_color)
axs[1, 0].set_title('UMAP')
axs[1, 0].set_xlabel('')
axs[1, 0].set_ylabel('')

# Isomap
print("Running Isomap...")
isomap = Isomap(n_components=2, n_neighbors=15) # n_neighbors can be tuned
embeddings_isomap = isomap.fit_transform(embeddings)
df_isomap = pd.DataFrame(embeddings_isomap, columns=['x', 'y'])
df_isomap['label'] = labels
sns.scatterplot(data=df_isomap, x='x', y='y', hue='label', ax=axs[1, 1], palette=palette, hue_order=unique_labels_arr, legend=False)
for label_val in unique_labels_arr:
    subset = df_isomap[df_isomap['label'] == label_val]
    if not subset.empty and len(subset['x']) > 2 and len(subset['y']) > 2:
        current_color = label_to_color_map[label_val]
        confidence_ellipse(subset['x'], subset['y'], axs[1, 1],
                           facecolor=current_color,
                           alpha=0.05,
                           edgecolor=current_color)
axs[1, 1].set_title('Isomap')
axs[1, 1].set_xlabel('')
axs[1, 1].set_ylabel('')

# Adjust layout and save the figure
plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
plt.savefig('dimensionality_reduction_comparison.png')
print('Saved dimensionality_reduction_comparison.png')
plt.show()
