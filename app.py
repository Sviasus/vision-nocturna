from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import uuid
import os

# Crear app
app = FastAPI()

# Montar carpetas
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/results", StaticFiles(directory="results"), name="results")

# Templates
templates = Jinja2Templates(directory="templates")

# Crear carpeta results si no existe
if not os.path.exists("results"):
    os.makedirs("results")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/procesar/")
async def procesar(request: Request, file: UploadFile = File(...)):
    # Guardar archivo temporal
    contents = await file.read()
    filename = f"results/{uuid.uuid4().hex}_{file.filename}"
    with open(filename, "wb") as f:
        f.write(contents)

    # Leer imagen con OpenCV
    img = cv2.imread(filename)
    if img is None:
        return {"error": "No se pudo leer la imagen"}

    # 1. Original
    original_path = filename

    # 2. Escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_path = f"results/gray_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(gray_path, gray)

    # 3. Ecualización de histograma (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    eq_path = f"results/eq_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(eq_path, gray_eq)

    # 4. Reducción de ruido
    denoise = cv2.GaussianBlur(gray_eq, (5, 5), 1)
    denoise = cv2.medianBlur(denoise, 3)
    denoise_path = f"results/denoise_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(denoise_path, denoise)

    # 5. Bordes + nitidez
    blur = cv2.GaussianBlur(denoise, (9, 9), 10)
    sharp = cv2.addWeighted(denoise, 1.5, blur, -0.5, 0)
    edges = cv2.Canny(denoise, 50, 150)
    fusion = cv2.addWeighted(sharp, 0.8, edges, 0.5, 0)
    fusion_path = f"results/fusion_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(fusion_path, fusion)

    # 6. Visión nocturna (gamma + falso color)
    def corregir_gamma(img, gamma):
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(img, table)

    night_gamma = corregir_gamma(fusion, gamma=2.8)
    night_vision = cv2.applyColorMap(night_gamma, cv2.COLORMAP_SUMMER)
    night_path = f"results/night_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(night_path, night_vision)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "original": "/" + original_path,
        "gray": "/" + gray_path,
        "eq": "/" + eq_path,
        "denoise": "/" + denoise_path,
        "fusion": "/" + fusion_path,
        "night": "/" + night_path
    })
