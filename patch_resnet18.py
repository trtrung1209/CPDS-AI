import json

def patch_model_in_notebook():
    nb_path = "notebooks/01_audio_training.ipynb"
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except FileNotFoundError:
        print(f"Error: {nb_path} not found.")
        return

    new_model_code = """class AudioCNN(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        import torchvision.models as models
        # Load pre-trained ResNet18 (Transfer Learning)
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Replace the final fully connected layer to match our classes (2 classes: noise, cry)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        # x comes in as (Batch, 1 Channel, 128, 63)
        # ResNet18 expects 3 channels (RGB). We repeat our 1 channel 3 times.
        x = x.repeat(1, 3, 1, 1)
        return self.resnet(x)
"""

    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            
            # Find the AudioCNN class definition
            if "class AudioCNN(nn.Module):" in src and "self.features = nn.Sequential(" in src:
                # We replace the whole cell content with the new model
                # Plus the def conv_block which might be there but we don't need it anymore.
                # Just to be safe, we replace the class AudioCNN block
                start_idx = src.find("class AudioCNN(nn.Module):")
                
                new_src = src[:start_idx] + new_model_code
                
                # Update cell source
                lines = new_src.split("\n")
                if lines:
                    cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
                break

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
        
    print("Thanh cong! Da thay loi AudioCNN cu bang ResNet18 (Transfer Learning) vao Notebook.")

if __name__ == "__main__":
    patch_model_in_notebook()
