import os
import argparse
from crism_ml.multi_head_attention_spectral import SpectralAttentionModel
from crism_ml.train import get_ratioed_image, remove_continuum
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
        class_names = [f'Mineral_{i.replace("/","_").replace(" ","")}' for i in FULL_NAMES.values()]
    if len(class_names) < num_classes:
        for i in range(len(class_names), num_classes):
            class_names.append(f"Mineral {i+1}")
    return class_names[:num_classes]

def main(img_path=None):
    parser = argparse.ArgumentParser(description='CRISM Mineral Classification Inference')
    parser.add_argument('--model_path', type=str, default="./results/checkpoints/model_epoch_9.pth")
    parser.add_argument('--img_path', type=str, default="./images/hrl00021d94_07_if180l_trr3.img")
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
    if not img_path:
        img_path = args.img_path
    # Create output directory structure
    image_name = os.path.splitext(os.path.basename(img_path))[0]
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
    import crism_ml.preprocessing as cp
    try:
        datadir = "/media/rushiil/Data/rushiil_2025/isro-2025/crism_ml_plebani/datasets"
        image_data = get_ratioed_image(img_path, datadir)
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
                "image_path": img_path,
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
    colors = plt.cm.jet(np.linspace(0, 1, args.num_classes+ 1))
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



BANDS_ = np.array([
    1.021, 1.02755, 1.0341, 1.04065, 1.0472, 1.05375, 1.0603, 1.06685,
    1.07341, 1.07996, 1.08651, 1.09307, 1.09962, 1.10617, 1.11273, 1.11928,
    1.12584, 1.13239, 1.13895, 1.14551, 1.15206, 1.15862, 1.16518, 1.17173,
    1.17829, 1.18485, 1.19141, 1.19797, 1.20453, 1.21109, 1.21765, 1.22421,
    1.23077, 1.23733, 1.24389, 1.25045, 1.25701, 1.26357, 1.27014, 1.2767,
    1.28326, 1.28983, 1.29639, 1.30295, 1.30952, 1.31608, 1.32265, 1.32921,
    1.33578, 1.34234, 1.34891, 1.35548, 1.36205, 1.36861, 1.37518, 1.38175,
    1.38832, 1.39489, 1.40145, 1.40802, 1.41459, 1.42116, 1.42773, 1.43431,
    1.44088, 1.44745, 1.45402, 1.46059, 1.46716, 1.47374, 1.48031, 1.48688,
    1.49346, 1.50003, 1.50661, 1.51318, 1.51976, 1.52633, 1.53291, 1.53948,
    1.54606, 1.55264, 1.55921, 1.56579, 1.57237, 1.57895, 1.58552, 1.5921,
    1.59868, 1.60526, 1.61184, 1.61842, 1.625, 1.63158, 1.63816, 1.64474,
    1.65133, 1.65791, 1.66449, 1.67107, 1.67766, 1.68424, 1.69082, 1.69741,
    1.70399, 1.71058, 1.71716, 1.72375, 1.73033, 1.73692, 1.74351, 1.75009,
    1.75668, 1.76327, 1.76985, 1.77644, 1.78303, 1.78962, 1.79621, 1.8028,
    1.80939, 1.81598, 1.82257, 1.82916, 1.83575, 1.84234, 1.84893, 1.85552,
    1.86212, 1.86871, 1.8753, 1.8819, 1.88849, 1.89508, 1.90168, 1.90827,
    1.91487, 1.92146, 1.92806, 1.93465, 1.94125, 1.94785, 1.95444, 1.96104,
    1.96764, 1.97424, 1.98084, 1.98743, 1.99403, 2.00063, 2.00723, 2.01383,
    2.02043, 2.02703, 2.03363, 2.04024, 2.04684, 2.05344, 2.06004, 2.06664,
    2.07325, 2.07985, 2.08645, 2.09306, 2.09966, 2.10627, 2.11287, 2.11948,
    2.12608, 2.13269, 2.1393, 2.1459, 2.15251, 2.15912, 2.16572, 2.17233,
    2.17894, 2.18555, 2.19216, 2.19877, 2.20538, 2.21199, 2.2186, 2.22521,
    2.23182, 2.23843, 2.24504, 2.25165, 2.25827, 2.26488, 2.27149, 2.2781,
    2.28472, 2.29133, 2.29795, 2.30456, 2.31118, 2.31779, 2.32441, 2.33102,
    2.33764, 2.34426, 2.35087, 2.35749, 2.36411, 2.37072, 2.37734, 2.38396,
    2.39058, 2.3972, 2.40382, 2.41044, 2.41706, 2.42368, 2.4303, 2.43692,
    2.44354, 2.45017, 2.45679, 2.46341, 2.47003, 2.47666, 2.48328, 2.4899,
    2.49653, 2.50312, 2.50972, 2.51632, 2.52292, 2.52951, 2.53611, 2.54271,
    2.54931, 2.55591, 2.56251, 2.56911, 2.57571, 2.58231, 2.58891, 2.59551,
    2.60212, 2.60872, 2.61532, 2.62192, 2.62853, 2.63513, 2.64174, 2.64834,
    2.80697, 2.81358, 2.8202, 2.82681, 2.83343, 2.84004, 2.84666, 2.85328,
    2.85989, 2.86651, 2.87313, 2.87975, 2.88636, 2.89298, 2.8996, 2.90622,
    2.91284, 2.91946, 2.92608, 2.9327, 2.93932, 2.94595, 2.95257, 2.95919,
    2.96581, 2.97244, 2.97906, 2.98568, 2.99231, 2.99893, 3.00556, 3.01218,
    3.01881, 3.02544, 3.03206, 3.03869, 3.04532, 3.05195, 3.05857, 3.0652,
    3.07183, 3.07846, 3.08509, 3.09172, 3.09835, 3.10498, 3.11161, 3.11825,
    3.12488, 3.13151, 3.13814, 3.14478, 3.15141, 3.15804, 3.16468, 3.17131,
    3.17795, 3.18458, 3.19122, 3.19785, 3.20449, 3.21113, 3.21776, 3.2244,
    3.23104, 3.23768, 3.24432, 3.25096, 3.2576, 3.26424, 3.27088, 3.27752,
    3.28416, 3.2908, 3.29744, 3.30408, 3.31073, 3.31737, 3.32401, 3.33066,
    3.3373, 3.34395, 3.35059, 3.35724, 3.36388, 3.37053, 3.37717, 3.38382,
    3.39047, 3.39712, 3.40376, 3.41041, 3.41706, 3.42371, 3.43036, 3.43701,
    3.44366, 3.45031, 3.45696, 3.46361, 3.47026, 3.47692
])



import json
import numpy as np
from matplotlib.patches import Patch
from crism_ml.train import get_ratioed_image
def plot_selected_minerals_on_rgb(json_path, minerals_to_plot, img_path, save_path=None, alpha=1, figsize=(14, 12)):
    """
    Plots selected minerals on the RGB image using detection coordinates from the JSON file.
    Also plots the mean reflectance spectrum of the ratioed image.

    Args:
        json_path (str): Path to the JSON file with detection results.
        minerals_to_plot (dict): Dictionary mapping class indices to mineral names to plot.
        img_path (str): Path to the CRISM .img file.
        save_path (str, optional): Path to save the overlay plot. If None, displays the plot.
        alpha (float): Overlay transparency.
        figsize (tuple): Figure size.
    """
    # Load detection results
    with open(json_path, 'r') as f:
        results = json.load(f)

    class_info = results['class_info']
    image_dims = results['metadata']['image_dimensions']
    height, width, bands = image_dims
    import crism_ml.preprocessing as cp
    # Load ratioed image and create RGB
    datadir = "/media/rushiil/Data/rushiil_2025/isro-2025/crism_ml_plebani/datasets"
    image_data = get_ratioed_image(img_path, datadir)
    ifm, _ = remove_continuum(image_data.reshape(-1, image_data.shape[2]))
    print(f"ifm_shape: {ifm.shape}")
    image_data_reshaped = ifm.reshape(height, width, bands)
    rgb_image = create_rgb_from_hyperspectral(image_data)
    rgb_norm = rgb_image.astype(np.float32) / 255.0

    # Prepare overlay
    overlay = np.zeros((height, width, 4), dtype=np.float32)
    colors = np.vstack([
        # Strong, darker colors that work well as overlays
        np.array([[139/255, 0, 0, 1],       # dark red
                 [0, 100/255, 0, 1],        # dark green
                 [0, 0, 139/255, 1],        # dark blue
                 [255/255, 140/255, 0, 1],  # dark orange
                 [128/255, 0, 128/255, 1],  # purple
                 [139/255, 69/255, 19/255, 1],  # saddle brown
                 [0, 139/255, 139/255, 1],  # dark cyan
                 [184/255, 134/255, 11/255, 1], # dark goldenrod
                 [165/255, 42/255, 42/255, 1],  # brown
                 [85/255, 107/255, 47/255, 1],  # dark olive green
                 [72/255, 61/255, 139/255, 1],  # dark slate blue
                 [178/255, 34/255, 34/255, 1], # firebrick
                # Additional colors for more variety
                 [70/255, 130/255, 180/255, 1],   # steel blue
                 [205/255, 92/255, 92/255, 1],      # indian red
                 [50/255, 205/255, 50/255, 1],      # lime green
                 [147/255, 112/255, 219/255, 1],    # medium purple
                 [218/255, 165/255, 32/255, 1],     # goldenrod
                 [0, 191/255, 255/255, 1],          # deep sky blue
                 [255/255, 99/255, 71/255, 1],      # tomato
                 [154/255, 205/255, 50/255, 1],     # yellow green
                 [186/255, 85/255, 211/255, 1],     # medium orchid
                 [244/255, 164/255, 96/255, 1]   # sandy brown
    ])])
    color_map = colors[:len(minerals_to_plot)]#plt.cm.tab20(np.linspace(0, 1, len(minerals_to_plot)))
    legend_handles = []

    # Create base overlay with transparency
    overlay[:, :, 3] = 0  # Set all pixels initially transparent

    for i, (class_id, mineral_name) in enumerate(minerals_to_plot.items()):
        # Find the correct key in class_info (may be int or str)
        for k in class_info:
            if (str(class_id) == str(class_info[k]['class_id'])) or (mineral_name == k):
                coords = class_info[k]['coordinates']
                break
        else:
            continue  # Skip if not found

        color = color_map[i][:3]
        for x, y in coords:
            if 0 <= y < height and 0 <= x < width:  # Ensure coordinates are within bounds
                overlay[y, x, :3] = color
                overlay[y, x, 3] += alpha
        legend_handles.append(Patch(color=color, label=mineral_name))

    # Plot overlay on RGB
    plt.figure(figsize=figsize)
    plt.imshow(rgb_norm)
    plt.imshow(overlay, alpha=1)  # Add global alpha for better visibility
    plt.title('Selected Mineral Detections Overlay')
    plt.axis('off')
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # Create spectra plots directory
    base_dir = os.path.dirname(json_path)
    spectra_dir = os.path.join(base_dir, 'spectra_plots')
    os.makedirs(spectra_dir, exist_ok=True)
    
    # Dictionary to store spectral data
    # spectral_data = {
    #     "metadata": {
    #         "bands": BANDS_[:248].tolist(),
    #         "minerals": minerals_to_plot
    #     },
    #     "mineral_spectra": {}
    # }

    # Plot aggregate overlay
    aggregate_path = os.path.join(spectra_dir, 'aggregate_overlay.png')
    plt.figure(figsize=(14, 12))
    plt.imshow(rgb_norm)
    plt.imshow(overlay, alpha=0.7)
    plt.title('Selected Mineral Detections Overlay')
    plt.axis('off')
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(aggregate_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Create subdirectories for individual and combined plots
    individual_dir = os.path.join(spectra_dir, 'individual_spectra')
    combined_dir = os.path.join(spectra_dir, 'combined_spectra')
    os.makedirs(individual_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)

    # Process each mineral's spectra
    all_mean_spectra = []
    # Using a combination of brighter colormaps
    colors = np.vstack([
        # Strong, darker colors that work well as overlays
        np.array([[139/255, 0, 0, 1],       # dark red
                 [0, 100/255, 0, 1],        # dark green
                 [0, 0, 139/255, 1],        # dark blue
                 [255/255, 140/255, 0, 1],  # dark orange
                 [128/255, 0, 128/255, 1],  # purple
                 [139/255, 69/255, 19/255, 1],  # saddle brown
                 [0, 139/255, 139/255, 1],  # dark cyan
                 [184/255, 134/255, 11/255, 1], # dark goldenrod
                 [165/255, 42/255, 42/255, 1],  # brown
                 [85/255, 107/255, 47/255, 1],  # dark olive green
                 [72/255, 61/255, 139/255, 1],  # dark slate blue
                 [178/255, 34/255, 34/255, 1]]) # firebrick
    ])
    color_map = colors[:len(minerals_to_plot)]

    for i, (class_id, mineral_name) in enumerate(minerals_to_plot.items()):
        # Find coordinates
        for k in class_info:
            if (str(class_id) == str(class_info[k]['class_id'])) or (mineral_name == k):
                coords = class_info[k]['coordinates']
                break
        else:
            continue

        # Collect spectra
        spectra = []
        for x, y in coords:
            if 0 <= y < height and 0 <= x < width:
                spectra.append(image_data_reshaped[y, x, :])
        
        if not spectra:
            continue

        spectra_array = np.stack(spectra)
        mean_spectrum = np.nanmean(spectra_array, axis=0)
        all_mean_spectra.append((mineral_name, mean_spectrum, color_map[i][:3]))

        # Store in spectral data
        # spectral_data["mineral_spectra"][mineral_name] = {
        #     "mean_spectrum": mean_spectrum.tolist(),
        #     "all_spectra": spectra_array.tolist(),
        #     "color": color_map[i][:3].tolist()
        # }

        # Individual mineral plot
        plt.figure(figsize=(10, 6))
        for spectrum in spectra[:10]:  # Plot first 10 detections
            plt.plot(BANDS_[:248], spectrum, color=color_map[i][:3], alpha=0.2)
        plt.plot(BANDS_[:248], mean_spectrum, color=color_map[i][:3], linewidth=2, label='Average')
        plt.title(f'{mineral_name} Spectra')
        plt.xlabel('Wavelength (μm)')
        plt.ylabel('Reflectance')
        plt.legend()
        plt.savefig(os.path.join(individual_dir, f'{mineral_name.replace("/", "_")}_spectra.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    # Combined plot of all mineral averages
    plt.figure(figsize=(12, 8))
    for name, mean_spec, color in all_mean_spectra[:10]:  # First 10 minerals
        plt.plot(BANDS_[:248], mean_spec, color=color, label=name, linewidth=2)
    plt.title('Average Spectra of Detected Minerals')
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Reflectance')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(combined_dir, 'combined_average_spectra.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

    # Save spectral data to JSON
    # spectral_json_path = os.path.join(spectra_dir, 'spectral_data.json')
    # with open(spectral_json_path, 'w') as f:
    #     json.dump(spectral_data, f, indent=2)

if __name__ == '__main__':
    image_name = "hrl00016cfe_07_if181l_trr3"
    # for image_name in image_names:    
    img_path = f"./images/{image_name}.img"
    main(img_path)
    MINERALS_TO_PLOT = {
    # 13: 'Ca/Fe CO3',  # Calcite, Ca/Fe carbonate
    29: 'Iron Oxide Silicate Sulfate',
    10: 'Serpentine',
    30: 'MgCO3',  # Magnesite
    31: 'Chlorite',
    32: 'Clinochlore',
    33: 'Low Ca Pyroxene',
    34: 'Olivine Forsterite',
    35: 'High Ca Pyroxene',
    36: 'Olivine Fayalite',
    # 37: 'Chloride',
    }

    # Get list of subdirectories in inference results directory
    # base_dir = "/media/rushiil/Data/rushiil_2025/isro-2025/crism_ml_plebani/inference_results_dir"
    # image_names = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    # print(image_names)
    if os.path.exists(img_path):
        save_path = f"./inference_results_dir/{image_name}/mineral_overlay.png"
        plot_selected_minerals_on_rgb(f"./inference_results_dir/{image_name}/class_predictions.json",
                    MINERALS_TO_PLOT,
                    img_path,
                    save_path=save_path
                    )
    else:
        print(f"Skipping {image_name} - image file not found")
        