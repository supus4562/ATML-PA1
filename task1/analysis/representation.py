from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

def run_tsne(features_clean, features_transformed, labels, backbone_name, output_dir, config, transform_name="Grayscale"):
    features_combined = np.concatenate([features_clean, features_transformed], axis=0)

    tsne = TSNE(
        n_components=2,
        perplexity=config['tsne']['perplexity'],
        max_iter=config['tsne']['n_iter'],
        random_state=config['tsne']['random_state']
    )

    proj = tsne.fit_transform(features_combined)

    proj_clean = proj[:len(features_clean)]
    proj_trans = proj[len(features_clean):]

    # Use a colormap with enough colors for 37 classes
    num_classes = len(np.unique(labels))
    try:
        cmap = plt.colormaps['nipy_spectral'].resampled(num_classes)
    except AttributeError:
        import matplotlib.cm as cm
        cmap = cm.get_cmap('nipy_spectral', num_classes)

    plt.figure(figsize=(10, 8))
    plt.scatter(proj_clean[:, 0], proj_clean[:, 1], c=labels, cmap=cmap, vmin=0, vmax=num_classes-1,
                marker='o', alpha=0.7, s=20, label='Clean')
    plt.scatter(proj_trans[:, 0], proj_trans[:, 1], c=labels, cmap=cmap, vmin=0, vmax=num_classes-1,
                marker='x', alpha=0.7, s=20, label=transform_name)

    plt.legend()
    plt.title(f't-SNE for {backbone_name} (Clean vs {transform_name})')

    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    safe_name = backbone_name.replace('/', '_').replace(' ', '_')
    tf_slug = transform_name.lower().replace(' ', '_')
    plt.savefig(os.path.join(output_dir, 'figures', f'tsne_{safe_name}_{tf_slug}.png'), dpi=150, bbox_inches='tight')
    # Also save the legacy filename if transform_name is Grayscale for backwards compatibility
    if transform_name.lower() == "grayscale":
        plt.savefig(os.path.join(output_dir, 'figures', f'tsne_{safe_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
