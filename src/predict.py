import torch
from torchvision import transforms
from PIL import Image
from model import SimpleCNN

def predict(image_path):
    model = SimpleCNN()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, predicted = output.max(1)
    return "Pothole" if predicted.item() == 1 else "No Pothole"

if __name__ == "__main__":
    result = predict("data/sample/test.jpg")
    print("Prediction:", result)
