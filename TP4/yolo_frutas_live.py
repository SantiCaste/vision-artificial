import cv2
from ultralytics import YOLO

# Cargar modelo entrenado
model = YOLO("runs/detect/frutas_model/weights/best.pt")  # ruta a tu best.pt

# Abrir la cámara
cap = cv2.VideoCapture(0)  # 0 = cámara principal

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detectar objetos
    results = model.predict(frame)

    # Dibujar cajas y etiquetas
    img = results[0].plot()

    # Mostrar ventana
    cv2.imshow("YOLOv8 Frutas", img)

    # Presioná 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
