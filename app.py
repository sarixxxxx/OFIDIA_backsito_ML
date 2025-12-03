import io
from typing import Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image

# =========================================
# CONFIGURACIÓN GENERAL
# =========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ⚠️ ESTE es el archivo que mostraste en el screenshot:
# torch.save(model.state_dict(), "modelo_serpientes.pth")
MODEL_PATH = "modelo_serpientes.pth"

class_names = ["No venenosa", "Coral", "Víbora"]
NUM_CLASSES = len(class_names)

MODEL_LOADED_OK = False

# =========================================
# MODELO: MISMA ESTRUCTURA QUE EN EL NOTEBOOK
# =========================================

class SnakeNet(nn.Module):
    """
    Envuelve una ResNet50 en self.model, porque el state_dict
    que guardaste tiene claves tipo 'model.conv1.weight', etc.
    """
    def __init__(self, num_classes: int = 3):
        super().__init__()

        # Backbone: ResNet50 preentrenada
        self.model = models.resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V1
        )

        # Cabezal final: Sequential con Dropout + Linear
        # (state_dict tiene 'model.fc.1.weight' -> fc[1] es Linear)
        in_f = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_f, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# Instanciar modelo
model: nn.Module = SnakeNet(num_classes=NUM_CLASSES).to(DEVICE)

# =========================================
# CARGAR PESOS DESDE modelo_serpientes.pth
# =========================================

try:
    # Guardaste SOLO state_dict -> es seguro usar weights_only=True
    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    # Aquí las claves deben verse como 'model.conv1.weight', etc.
    # SnakeNet también tiene esas claves, porque tiene self.model = resnet50.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing or unexpected:
        # Solo para ver en consola si quedó algo raro
        print("⚠️ Missing keys:", missing)
        print("⚠️ Unexpected keys:", unexpected)

    model.eval()
    MODEL_LOADED_OK = True
    print("✅ Modelo cargado correctamente desde modelo_serpientes.pth")

except Exception as e:
    MODEL_LOADED_OK = False
    print("❌ Error al cargar el modelo:", e)

# =========================================
# TRANSFORMACIONES (IGUAL QUE EN TEST)
# =========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================
# FUNCIÓN DE PREDICCIÓN
# =========================================

def predict_pil_image(pil_image: Image.Image) -> Tuple[str, float]:
    """
    Recibe una imagen PIL y devuelve:
    - nombre de clase predicha
    - confianza en porcentaje
    """
    if not MODEL_LOADED_OK:
        raise RuntimeError("El modelo no está cargado correctamente.")

    img_t = transform(pil_image)
    batch_t = img_t.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(batch_t)
        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    predicted_class = class_names[pred.item()]
    confidence = conf.item() * 100.0
    return predicted_class, confidence

# =========================================
# FASTAPI
# =========================================

app = FastAPI(
    title="API Clasificación de Serpientes",
    description="API que recibe una imagen y devuelve si es 'No venenosa', 'Coral' o 'Víbora'.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en producción restringe esto
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "message": "API de clasificación de serpientes funcionando 🚀",
        "model_loaded": MODEL_LOADED_OK,
        "model_path": MODEL_PATH,
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint que recibe un archivo de imagen (multipart/form-data)
    y devuelve la clase predicha y la confianza.
    """
    if not MODEL_LOADED_OK:
        raise HTTPException(
            status_code=500,
            detail="El modelo no se cargó correctamente al iniciar la API."
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser una imagen (content-type image/*)."
        )

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        predicted_class, confidence = predict_pil_image(pil_image)

        return JSONResponse({
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": confidence
        })

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la imagen: {e}"
        )
