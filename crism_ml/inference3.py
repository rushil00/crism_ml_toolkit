import os
import argparse
from crism_ml.multi_head_attention_spectral import SpectralAttentionModel
from crism_ml.train import get_ratioed_image
import numpy as np
import torch
import json
from torch.nn import functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import Image
import scipy.io

def process_spectrum(spectrum, target_length=248):
    if len(spectrum) < target_length:
        padded = np.pad(spectrum, (0, target_length - len(spectrum)), mode='edge')
        return padded
    elif len(spectrum) > target_length:
        return spectrum[:target_length]
    else:
        return spectrum

def normalize_spectrum(spectrum):
    mean = np.mean(spectrum)
    std = np.std(spectrum) + 1e-8
    return (spectrum - mean) / std

def create_rgb_from_hyperspectral(image_data):
    bands = image_data.shape[2]
    if bands >= 200:
        r_idx = int(bands * 0.8)
        g_idx = int(bands * 0.5)
        b_idx = int(bands * 0.2)
    else:
        r_idx = min(int(bands * 0.8), bands-1)
        g_idx = min(int(bands * 0.5), bands-1)
        b_idx = min(int(bands * 0.2), bands-1)
    
    rgb = np.zeros((image_data.shape[0], image_data.shape[1], 3), dtype=np.float32)
    for i, idx in enumerate([r_idx, g_idx, b_idx]):
        band = image_data[:, :, idx].copy()
        if np.any(~np.isnan(band)):
            valid_mask = ~np.isnan(band)
            min_val = np.min(band[valid_mask])
            max_val = np.max(band[valid_mask])
            if max_val > min_val:
                band = (band - min_val) / (max_val - min_val)
            band[~valid_mask] = 0
        rgb[:, :, i] = band.squeeze()
    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255).astype(np.uint8)

def create_inference_batch(image_data, batch_size=128):
    height, width, bands = image_data.shape
    pixels = image_data.reshape(-1, bands)
    for i in range(0, len(pixels), batch_size):
        batch = pixels[i:i+batch_size]
        processed_batch = []
        valid_indices = []
        for j, spectrum in enumerate(batch):
            if np.all(spectrum == 0) or np.any(np.isnan(spectrum)):
                continue
            processed = process_spectrum(spectrum, args.num_bands)
            normalized = normalize_spectrum(processed)
            processed_batch.append(normalized)
            valid_indices.append(i + j)
        if processed_batch:
            yield torch.tensor(np.array(processed_batch), dtype=torch.float32), valid_indices

def create_mineral_labels(num_classes, class_names=None):
    from crism_ml.lab import FULL_NAMES
    if class_names is None:
        class_names = [f"Mineral_{i.replace("/","_").replace(" ","")}" for i in FULL_NAMES.values()]
    if len(class_names) < num_classes:
        for i in range(len(class_names), num_classes):
            class_names.append(f"Mineral {i+1}")
    return class_names[:num_classes]

def main():
    parser = argparse.ArgumentParser(description='CRISM Mineral Classification Inference')
    parser.add_argument('--model_path', type=str, default="./results/checkpoints/model_epoch_9.pth")
    parser.add_argument('--img_path', type=str, default="./images/frt0000b7be_07_if168l_trr3.img")
    parser.add_argument('--output_dir', type=str, default='inference_results_dir')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_classes', type=int, default=40)
    parser.add_argument('--num_bands', type=int, default=228)
    parser.add_argument('--confidence_threshold', type=float, default=0.5)
    parser.add_argument('--class_names', type=str, default=None)
    parser.add_argument('--rgb_blend', type=float, default=0.8)
    
    global args
    args = parser.parse_args()
    
    # Create output directory structure
    image_name = os.path.splitext(os.path.basename(args.img_path))[0]
    output_dir = os.path.join(args.output_dir, image_name)
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    mineral_labels = create_mineral_labels(args.num_classes, args.class_names.split(',') if args.class_names else None)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = SpectralAttentionModel(args.hidden_size, args.num_classes)
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    # Load image
    try:
        datadir = "path/to/data"
        image_data = get_ratioed_image(args.img_path, datadir)
        height, width, bands = image_data.shape
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Create RGB base image
    rgb_image = create_rgb_from_hyperspectral(image_data)
    rgb_norm = rgb_image.astype(np.float32) / 255.0
    
    # Initialize results
    classification_map = np.zeros((height, width), dtype=np.int32)
    confidence_map = np.zeros((height, width), dtype=np.float32)
    class_predictions = {}
    
    # Inference
    with torch.no_grad():
        for batch, indices in create_inference_batch(image_data, args.batch_size):
            if len(batch) == 0:
                continue
            outputs, _ = model(batch.to(device))
            probabilities = F.softmax(outputs, dim=1)
            confidence, predictions = torch.max(probabilities, dim=1)
            
            for i, idx in enumerate(indices):
                y, x = idx // width, idx % width
                pred_class = predictions[i].item()
                conf_value = confidence[i].item()
                if conf_value >= args.confidence_threshold:
                    if pred_class not in class_predictions:
                        class_predictions[pred_class] = []
                    class_predictions[pred_class].append([int(x), int(y)])
                    classification_map[y, x] = pred_class + 1
                    confidence_map[y, x] = conf_value

    # Filter out mineral 2 (class index 1)
    class_predictions = {k: v for k, v in class_predictions.items() if k != 1}
    
    # Save JSON results
    output_json = os.path.join(output_dir, 'class_predictions.json')
    with open(output_json, 'w') as f:
        json.dump({
            "metadata": {
                "image_path": args.img_path,
                "image_dimensions": [height, width, bands],
                "confidence_threshold": args.confidence_threshold
            },
            "class_info": {
                mineral_labels[k]: {
                    "class_id": int(k),
                    "pixel_count": len(v),
                    "coordinates": v
                } for k, v in class_predictions.items()
            }
        }, f, indent=2)

    # Create visualizations
    # Use 40 visually distinct, bright colors for overlay (avoid grayscale/white/black)
    # Colors: red, yellow, orange, green, blue, cyan, magenta, pink, purple, turquoise, etc.
    # Generated using tab20, Set1, Set2, Set3, Accent, Paired, Pastel1, Pastel2, Dark2, and some custom
    base_colors = [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6",
        "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
        "#808000", "#ffd8b1", "#000075", "#808080", "#ff7f00", "#1f78b4", "#b2df8a", "#33a02c",
        "#fb9a99", "#e31a1c", "#fdbf6f", "#ff1493", "#6a3d9a", "#b15928", "#17becf", "#bc80bd",
        "#ccebc5", "#8dd3c7", "#bebada", "#fb8072", "#80b1d3", "#fdb462", "#b3de69", "#fccde5",
        "#d9d9d9", "#bcbd22"
    ]
    # Convert hex to RGBA (normalized 0-1)
    colors = np.zeros((args.num_classes + 1, 4))
    for i in range(1, args.num_classes + 1):
        hex_color = base_colors[(i - 1) % len(base_colors)]
        rgb = tuple(int(hex_color.lstrip('#')[j:j+2], 16)/255.0 for j in (0, 2, 4))
        colors[i, :3] = rgb
        colors[i, 3] = 1.0  # alpha
    colors[0, 3] = 0  # Transparent background
    
    # 1. Aggregate detections plot
    plt.figure(figsize=(14, 12))
    plt.imshow(rgb_norm)
    overlay = np.zeros((height, width, 4))
    for class_id in class_predictions:
        mask = classification_map == (class_id + 1)
        class_color = colors[class_id + 1]
        overlay[mask] = [*class_color[:3], args.rgb_blend]
    plt.imshow(overlay)
    plt.title('Aggregate Mineral Detections')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'aggregate_detections.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Individual class plots
    for class_id in class_predictions:
        class_name = mineral_labels[class_id]
        plt.figure(figsize=(12, 10))
        plt.imshow(rgb_norm)
        
        # Create overlay for this class only
        overlay = np.zeros((height, width, 4))
        mask = classification_map == (class_id + 1)
        class_color = colors[class_id + 1]
        overlay[mask] = [*class_color[:3], args.rgb_blend]
        
        plt.imshow(overlay)
        plt.title(f'{class_name} Detections')
        plt.axis('off')
        plt.savefig(os.path.join(plots_dir, f'class_{class_id+1}_{class_name}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Class distribution plot
    class_counts = {mineral_labels[k]: len(v) for k, v in class_predictions.items()}
    plt.figure(figsize=(12, 8))
    plt.bar(class_counts.keys(), class_counts.values(), 
            color=[colors[i+1][:3] for i in class_predictions.keys()])
    plt.xticks(rotation=45, ha='right')
    plt.title('Mineral Class Distribution')
    plt.ylabel('Pixel Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300)
    plt.close()

    print(f"Results saved to: {output_dir}")

if __name__ == '__main__':
    main()