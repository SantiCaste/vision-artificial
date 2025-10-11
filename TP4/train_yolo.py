from ultralytics import YOLO
import torch


# Modelo base "small" (más preciso que nano)
model = YOLO("yolov8s.pt")

# Frizar el backbone (las capas de extracción de características) y dejar libres las capas del neck y head
for name, param in model.model.model.named_parameters():
    # Las capas del backbone suelen tener "backbone" o ser las primeras 3 bloques
    if "backbone" in name or int(name.split('.')[0]) < 3:  
        param.requires_grad = False

# Confirmar cuántos parámetros se frizan
total_params = sum(p.numel() for p in model.model.parameters())
trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

print(f"Total parámetros: {total_params}")
print(f"Parámetros entrenables: {trainable_params}")

# Entrenar
model.train(
    data="dataset/data.yaml",  # apunta a tu data.yaml
    epochs=5,
    imgsz=256,
    batch=8,
    workers=2,
    device='cpu'       
)

#Tomamos un modelo YOLOv8 preentrenado (yolov8s.pt).
#Este modelo ya conoce muchos objetos generales gracias a su entrenamiento previo (COCO dataset).
#Entrenamos el modelo sobre nuestro dataset de frutas (lo sacamos de roboflow) usando transfer learning.

#No cambiamos el backbone: la red que extrae características sigue siendo la misma.
#Lo que hacemos es ajustar el modelo para que aprenda a reconocer las nuevas clases.

#COCO solo tenia: Manzana, Naranja y Banana
#algunas nuevas que no reconocia COCO: Tomate, Cebolla, Berenjena, Frutilla