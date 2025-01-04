import torch
from model import VisionTransformerClassifier
from data_module import CIFAR100DataModule
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse




app = FastAPI(title="ViT CIFAR-100 Inference API")


def load_model(checkpoint_path: str = 'checkpoints/ViT-epoch=XX-val_acc=YY.ckpt'):
    model = VisionTransformerClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model

model = load_model()


def transform_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4865, 0.4409),
                             std=(0.2673, 0.2564, 0.2762)),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as e:
        return JSONResponse(content={"error": "Invalid image file"}, status_code=400)

    # Transform image
    input_tensor = transform_image(image)

    # Make prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

    # Get class names
    class_names = data_module.get_class_names()

    return {
        "predicted_class": class_names[predicted_class.item()],
        "confidence": confidence.item()
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)