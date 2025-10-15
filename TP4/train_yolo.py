from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # Modelo base "small" (más preciso que nano)
    model = YOLO("yolov8s.pt")

    # Congelar el backbone para transfer learning
    # El backbone ya está bien entrenado en COCO, solo ajustamos el head para nuevas clases
    for name, param in model.model.named_parameters():
        # Congelar las primeras 10 capas (backbone)
        if 'model.0.' in name or 'model.1.' in name or 'model.2.' in name or \
           'model.3.' in name or 'model.4.' in name or 'model.5.' in name or \
           'model.6.' in name or 'model.7.' in name or 'model.8.' in name or 'model.9.' in name:
            param.requires_grad = False

    # Confirmar parámetros del modelo
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

    print(f"Total parámetros: {total_params}")
    print(f"Parámetros entrenables: {trainable_params}")

    # Entrenar
    model.train(
        data="dataset/data.yaml",  # apunta a tu data.yaml
        epochs=100,      # Óptimo para tu dataset
        imgsz=256,
        batch=16,        # Ajustado para RTX 3090
        augment=True,
        workers=0,       # Cambiado a 0 para evitar problemas de multiprocessing en Windows
        device='cuda',   # Usando GPU
        patience=30,     # Ajustado proporcionalmente
        project='runs/detect',
        name='frutas_model',
        exist_ok=True  # Sobrescribir si ya existe
    )

    #Tomamos un modelo YOLOv8 preentrenado (yolov8s.pt).
    #Este modelo ya conoce muchos objetos generales gracias a su entrenamiento previo (COCO dataset).
    #Entrenamos el modelo sobre nuestro dataset de frutas (lo sacamos de roboflow) usando transfer learning.

    #No cambiamos el backbone: la red que extrae características sigue siendo la misma.
    #Lo que hacemos es ajustar el modelo para que aprenda a reconocer las nuevas clases.

    #COCO solo tenia: Manzana, Naranja y Banana
    #algunas nuevas que no reconocia COCO: Tomate, Cebolla, Berenjena, Frutilla