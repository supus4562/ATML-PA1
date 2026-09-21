from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

def run_tsne(features_clean, features_transformed, labels, backbone_name, output_dir, config):
    features_combined = np.concatenate([features_clean, features_transformed], axis=0)

    tsne = TSNE(
        n_components=2,
        perplexity=config['tsne']['perplexity'],
        max_iter=config['tsne']['n_iter'],  # renamed from n_iter in scikit-learn 1.5+
        random_state=config['tsne']['random_state']
    )

    proj = tsne.fit_transform(features_combined)

    proj_clean = proj[:len(features_clean)]
    proj_trans = proj[len(features_clean):]

    # Use a colormap with enough colors for 37 classes (tab20 has 20; use nipy_spectral for 37)
    num_classes = len(np.unique(labels))
    cmap = cm.get_cmap('nipy_spectral', num_classes)

    plt.figure(figsize=(10, 8))
    plt.scatter(proj_clean[:, 0], proj_clean[:, 1], c=labels, cmap=cmap, vmin=0, vmax=num_classes-1,
                marker='o', alpha=0.7, s=20, label='Clean')
    plt.scatter(proj_trans[:, 0], proj_trans[:, 1], c=labels, cmap=cmap, vmin=0, vmax=num_classes-1,
                marker='x', alpha=0.7, s=20, label='Grayscale')

    plt.legend()
    plt.title(f't-SNE for {backbone_name}')

    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    # Sanitize backbone_name — replace '/' so it doesn't create subdirectories in the filename
    safe_name = backbone_name.replace('/', '_').replace(' ', '_')
    plt.savefig(os.path.join(output_dir, 'figures', f'tsne_{safe_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
