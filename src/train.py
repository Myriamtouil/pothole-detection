import torch
from torch.utils.data import DataLoader
from dataset import PotholeDataset
from model import SimpleCNN

# load dataset
dataset = PotholeDataset("data/train")
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# init model
model = SimpleCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# training loop
for epoch in range(2):
    for images, labels in dataloader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# save model
torch.save(model.state_dict(), "model.pth")
