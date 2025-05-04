import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io
#NOTE: ACTUAL USED MODEL!!!!
# Import the model architecture from your training script
class SpectralAttentionModel(torch.nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(hidden_size*2, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Softmax(dim=1)
        )
        self.fc = torch.nn.Linear(hidden_size*2, num_classes)
        
    def forward(self, x):
        x = x.unsqueeze(-1)  # Add channel dimension
        lstm_out, _ = self.lstm(x)
        attention_weights = self.attention(lstm_out).squeeze()
        context = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        return self.fc(context), attention_weights

def read_crism_img(img_path):
    """Read a CRISM IMG file and return the spectral data"""
    try:
        # Try to read as a standard image first (for testing with .tif or other formats)
        try:
            img = Image.open(img_path)
            data = np.array(img)
            if len(data.shape) == 3:
                # Convert RGB to spectral data - this is just for testing
                return data
        except:
            pass
            
        # Try to read as a MATLAB file (some CRISM data is saved as .mat)
        try:
            mat = scipy.io.loadmat(img_path)
            # Extract the data - adjust the key based on your data structure
            for key in mat.keys():
                if isinstance(mat[key], np.ndarray) and len(mat[key].shape) >= 2:
                    if key not in ['__header__', '__version__', '__globals__']:
                        return mat[key]
        except:
            pass
            
        # Try to read as a raw CRISM .img file
        # This is a simplified approach - real CRISM data might need more processing
        with open(img_path, 'rb') as f:
            # Header information might need to be parsed
            # For simplicity, we're assuming a simple format
            data = np.fromfile(f, dtype=np.float32)
            # Reshape based on known dimensions - you might need to adjust this
            # or read dimensions from file header
            height = int(np.sqrt(len(data)/240))  # Assuming 240 bands
            width = height  # Assuming square image
            bands = 240
            return data.reshape(height, width, bands)
            
    except Exception as e:
        print(f"Error reading CRISM image: {e}")
        return None

class CRISMInferenceDataset(Dataset):
    def __init__(self, image_data, num_bands):
        """
        Args:
            image_data: CRISM image data (height x width x bands)
            num_bands: Target number of bands (will pad/truncate to this)
        """
        self.height, self.width, self.bands = image_data.shape
        self.num_bands = num_bands
        self.spectra = []
        self.coords = []
        
        # Process each pixel's spectrum
        for y in range(self.height):
            for x in range(self.width):
                spec = image_data[y, x, :].astype(np.float32)
                
                # Pad or truncate to target length
                if len(spec) < num_bands:
                    # Pad with zeros
                    pad_width = num_bands - len(spec)
                    spec = np.pad(spec, (0, pad_width), mode='constant')
                elif len(spec) > num_bands:
                    # Truncate
                    spec = spec[:num_bands]
                    
                # Normalize
                if np.std(spec) > 0:  # Avoid division by zero
                    spec = (spec - np.mean(spec)) / (np.std(spec) + 1e-8)
                
                self.spectra.append(spec)
                self.coords.append((x, y))
            
    def __len__(self):
        return len(self.spectra)
    
    def __getitem__(self, idx):
        return torch.tensor(self.spectra[idx]), self.coords[idx]

def main():
    parser = argparse.ArgumentParser(description='CRISM Mineral Classification Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model weights')
    parser.add_argument('--img_path', type=str, required=True, help='Path to CRISM image file')
    parser.add_argument('--hidden_size', type=int, default=128, help='LSTM hidden state size used in training')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of mineral classes')
    parser.add_argument('--num_bands', type=int, required=True, help='Number of spectral bands')
    parser.add_argument('--batch_size', type=int, default=256, help='Inference batch size')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--create_visualization', action='store_true', help='Create a visualization of predictions')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the model
    model = SpectralAttentionModel(args.hidden_size, args.num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # Load and process CRISM image
    print(f"Loading CRISM image from {args.img_path}...")
    image_data = read_crism_img(args.img_path)
    
    if image_data is None:
        print("Failed to load image. Exiting.")
        return
    
    print(f"Image loaded. Shape: {image_data.shape}")
    dataset = CRISMInferenceDataset(image_data, args.num_bands)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    
    # Perform inference
    print("Running inference...")
    predictions = []
    coords_by_class = {i: [] for i in range(args.num_classes)}
    
    with torch.no_grad():
        for spectra, coords in dataloader:
            spectra = spectra.to(device)
            outputs, _ = model(spectra)
            _, predicted = torch.max(outputs.data, 1)
            
            for pred, coord in zip(predicted.cpu().numpy(), coords):
                predictions.append(pred)
                coords_by_class[int(pred)].append(coord)
    
    # Save results to JSON
    output_file = os.path.join(args.output_dir, f"mineral_predictions_{os.path.basename(args.img_path)}.json")
    result_dict = {f"class_{i}": [{"x": int(x), "y": int(y)} for x, y in coords] 
                  for i, coords in coords_by_class.items()}
    
    # Add summary statistics
    result_dict["summary"] = {
        "image_path": args.img_path,
        "image_shape": list(image_data.shape),
        "total_pixels": len(dataset),
        "class_counts": {f"class_{i}": len(coords) for i, coords in coords_by_class.items()}
    }
    
    with open(output_file, 'w') as f:
        json.dump(result_dict, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Create visualization if requested
    if args.create_visualization:
        height, width = image_data.shape[:2]
        prediction_map = np.zeros((height, width), dtype=np.int32)
        
        for (x, y), pred in zip(dataset.coords, predictions):
            prediction_map[y, x] = pred
        
        plt.figure(figsize=(10, 10))
        plt.imshow(prediction_map, cmap='viridis')
        plt.colorbar(label='Mineral Class')
        plt.title('CRISM Mineral Classification')
        plt.savefig(os.path.join(args.output_dir, f"prediction_map_{os.path.basename(args.img_path)}.png"))
        plt.close()
        print(f"Visualization saved to {args.output_dir}")

if __name__ == "__main__":
    main()