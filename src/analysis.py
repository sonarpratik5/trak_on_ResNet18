import torch
import numpy as np
import matplotlib.pyplot as plt
from trak import TRAKer
from trak.projectors import BasicProjector
import gc
import os
from .data import get_trak_loader
from .interpretability import GradCAMVisualizer, get_last_conv_layer

def denormalize(img):
    """Reverses CIFAR-10 normalization for visualization."""
    # Move to CPU if it's a tensor
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    
    # Transpose from (C, H, W) to (H, W, C)
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
        
    img = std * img + mean
    return np.clip(img, 0, 1)

def run_trak_analysis(net, train_set, testloader, classes, checkpoint_path, device='cuda', seed=42):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the model first.")

    print("\n" + "="*80)
    print("STARTING TRAK ATTRIBUTION ANALYSIS")
    print("="*80 + "\n", flush=True)
    
    # 1. Load Model
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state'])
    net.to(device)
    net.eval()

    # 2. Setup TRAK
    grad_dim = sum(p.numel() for p in net.parameters() if p.requires_grad)
    trainloader_trak = get_trak_loader(train_set, batch_size=32, seed=seed)

    traker = TRAKer(
        model=net, 
        task='image_classification', 
        train_set_size=len(train_set),
        save_dir='./trak_results', 
        device=device, 
        proj_dim=2048,
        projector=BasicProjector(grad_dim=grad_dim, proj_dim=2048, seed=seed, proj_type='rademacher', device=device)
    )
    
    traker.load_checkpoint(checkpoint=net.state_dict(), model_id=0)
    
    # 3. Featurize Training Data
    print("Featurizing training data...")
    for batch_idx, batch in enumerate(trainloader_trak):
        inputs, labels = batch[0].to(device), batch[1].to(device)
        traker.featurize(batch=(inputs, labels), num_samples=inputs.size(0))
        
        if batch_idx % 200 == 0 and batch_idx > 0:
            torch.cuda.empty_cache()
            gc.collect()
            
    traker.finalize_features()

    # 4. Score Test Samples
    print("Scoring test samples...")
    test_images, test_labels = next(iter(testloader))
    num_test_samples = 10
    
    test_subset = test_images[:num_test_samples].to(device)
    test_labels_subset = test_labels[:num_test_samples].to(device)

    traker.start_scoring_checkpoint(exp_name='test_attribution', checkpoint=net.state_dict(), model_id=0, num_targets=num_test_samples)
    
    for i in range(num_test_samples):
        traker.score(batch=(test_subset[i:i+1], test_labels_subset[i:i+1]), num_samples=1)
    
    scores = traker.finalize_scores(exp_name='test_attribution')
    
    if scores.shape[0] == len(train_set):
        scores = scores.T

    # 5. Print Statistics
    k = 5
    for test_i in range(num_test_samples):
        current_label = test_labels_subset[test_i].item()
        current_scores = scores[test_i]
        top_k_indices = np.argsort(current_scores)[-k:][::-1]
        
        print(f"\n{'='*60}")
        print(f"Test sample {test_i} (True label: {classes[current_label]})")
        print(f"Top {k} most influential training examples:")
        
        for rank, train_idx in enumerate(top_k_indices):
            train_label = train_set[train_idx][1] 
            match = "✓" if train_label == current_label else "✗"
            print(f"  {rank+1}. Train sample {train_idx:5d} | Label: {classes[train_label]:8s} {match} | Score: {current_scores[train_idx]:7.4f}")

    # =========================================================================
    # 6. GRAD-CAM VISUALIZATION LOGIC
    # =========================================================================
    print("\nGenerating GradCAM Visualizations...")
    
    # Initialize Visualizer
    target_layer = get_last_conv_layer(net)
    visualizer = GradCAMVisualizer(net, target_layer)

    # We will visualize the first test sample (test_idx=0)
    test_idx = 0
    test_label = test_labels_subset[test_idx].item()
    attribution_scores = scores[test_idx]
    top_k_indices = np.argsort(attribution_scores)[-k:][::-1]

    # Setup Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Shared Representations: GradCAM on Test Image & Top Influencers', fontsize=16)

    # --- A. Process Test Image ---
    test_img_tensor = test_subset[test_idx:test_idx+1]
    test_rgb = denormalize(test_img_tensor[0]) 
    
    # Generate CAM for the TRUE label of the test image
    test_cam_mask = visualizer.generate_cam(test_img_tensor, target_class=test_label)
    test_viz = visualizer.overlay_cam(test_rgb, test_cam_mask, threshold=0.3)

    # Plot Test Image (Top Left)
    axes[0, 0].imshow(test_viz)
    axes[0, 0].set_title(f'TEST TARGET\n{classes[test_label]}', fontweight='bold', color='red')
    axes[0, 0].axis('off')
    
    axes[0, 1].axis('off')
    axes[0, 2].axis('off')

    # --- B. Process Top-5 Influencers ---
    flat_axes = axes.flatten()
    
    for i, train_idx in enumerate(top_k_indices):
        train_img_tensor, train_label = train_set[train_idx]
        train_img_tensor = train_img_tensor.unsqueeze(0).to(device) 
        
        train_rgb = denormalize(train_img_tensor[0])
        
        # Verify what the model looks at for the TRAINING sample's own label
        train_cam_mask = visualizer.generate_cam(train_img_tensor, target_class=train_label)
        train_viz = visualizer.overlay_cam(train_rgb, train_cam_mask, threshold=0.3)
        
        # Plot
        ax = flat_axes[i+1] # Skip the first slot (test image)
        score = attribution_scores[train_idx]
        match_color = 'green' if train_label == test_label else 'blue'
        
        ax.imshow(train_viz)
        ax.set_title(f'Rank #{i+1}: {classes[train_label]}\nScore: {score:.3f}', color=match_color)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('gradcam_analysis.png', dpi=300)
    plt.close(fig)
    
    np.save('attribution_scores.npy', scores)
    print("✅ GradCAM visualization saved to 'gradcam_analysis.png'")
    print("✅ Attribution scores saved to 'attribution_scores.npy'")