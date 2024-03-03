import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import Adam
import torch.nn as nn

# Define your video folder and preprocessed folder
video_folder = "/workspace/CS482_proj/Videos"
preprocessed_folder = "/workspace/CS482_proj/Frames"

# Define your Autoencoder and Dataset classes here...
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Use Sigmoid to get values between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Dataset class to load video frames
class VideoFrameDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.frame_files = [f for f in os.listdir(root) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, idx):
        frame_file = self.frame_files[idx]
        frame_path = os.path.join(self.root, frame_file)
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Convert to tensor
        frame = transforms.ToTensor()(frame)

        # Normalize
        mean = torch.mean(frame)
        std = torch.std(frame)
        frame = transforms.Normalize(mean=[mean.item()]*3, std=[std.item()]*3)(frame)
        frame = frame/255
        return frame

# Load your dataset without normalization
dataset = VideoFrameDataset(root=preprocessed_folder)

data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize the autoencoder and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss() 

# Function to combine embeddings
def combine_embeddings(embeddings):
    # Stack all embeddings into a single tensor
    embeddings_tensor = torch.stack(embeddings)
    # Compute the mean across the embeddings for each frame
    combined_embedding = torch.mean(embeddings_tensor, dim=0)
    return combined_embedding

# Training loop with embeddings
def train_autoencoder(model, data_loader, epochs=10):
    model.train()
    for epoch in range(epochs):
        embeddings = []
        for i, batch in enumerate(data_loader):
            # Move batch to device
            batch = batch.to(device)
            # Forward pass
            encoded, reconstructed = model(batch)
            embeddings.append(encoded.detach())  # Keep on same device as model
            # Compute loss
            loss = criterion(reconstructed, batch)
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Iteration {i+1} completed successfully, Loss: {loss.item():.4f}")
        print(f"Epoch [{epoch+1}/{epochs}] completed successfully")
        # Combine embeddings after each epoch
        combined_embedding = combine_embeddings(embeddings)
        print(f"Combined embedding for epoch {epoch+1}: {combined_embedding}")

# Call the training function
train_autoencoder(model, data_loader, epochs=2)

print("Autoencoder training completed.")