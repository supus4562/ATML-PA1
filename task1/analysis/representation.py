from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os

def run_tsne(features_clean, features_transformed, labels, backbone_name, output_dir, config):
    features_combined = np.concatenate([features_clean, features_transformed], axis=0)
    
    tsne = TSNE(
        n_components=2, 
        perplexity=config['tsne']['perplexity'],
        n_iter=config['tsne']['n_iter'],
        random_state=config['tsne']['random_state']
    )
    
    proj = tsne.fit_transform(features_combined)
    
    proj_clean = proj[:len(features_clean)]
    proj_trans = proj[len(features_clean):]
    
    plt.figure(figsize=(10, 8))
    scatter_clean = plt.scatter(proj_clean[:, 0], proj_clean[:, 1], c=labels, cmap='tab10', marker='o', alpha=0.7, label='Clean')
    scatter_trans = plt.scatter(proj_trans[:, 0], proj_trans[:, 1], c=labels, cmap='tab10', marker='x', alpha=0.7, label='Transformed')
    
    plt.legend()
    plt.title(f't-SNE for {backbone_name}')
    
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'figures', f'tsne_{backbone_name}.png'))
    plt.close()
