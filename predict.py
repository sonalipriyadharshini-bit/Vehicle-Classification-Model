import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ---------------- Config ----------------
MODEL_PATH = "vehicle_type_resnet18_final.pth"
CLASS_NAMES = ['Bus', 'Car', 'Motorcycle', 'Truck']  # must match training order (alphabetical)
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Rebuild model architecture ----------------
def load_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run vehicle_classifier.py first to train and save the model."
        )
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, len(CLASS_NAMES))
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# ---------------- Preprocessing (must match val_transform used in training) ----------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

def predict(model, image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    print(f"\nImage: {image_path}")
    print(f"Predicted class: {CLASS_NAMES[pred_idx]}")
    print(f"Confidence: {probs[pred_idx]*100:.2f}%")
    print("\nAll class probabilities:")
    for cls, p in zip(CLASS_NAMES, probs):
        print(f"  {cls}: {p*100:.2f}%")

def main():
    model = load_model()
    print("Model loaded. Ready to classify images.")

    while True:
        image_path = input("\nEnter path to an image to classify (or press Enter to quit): ").strip().strip('"')
        if image_path == "":
            print("Exiting.")
            break
        if not os.path.isfile(image_path):
            print("File not found. Try again.")
            continue
        predict(model, image_path)

if __name__ == "__main__":
    main()