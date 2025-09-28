import cv2
import numpy as np
import yaml

# Configuración
CAP = 'http://192.168.1.42:8080/video'# 0                 # cámara: 0/1 (webcam) o URL (p. ej. 'http://IP:PORT/video' o 'rtsp://...')
GRID_N = 3              # grilla NxN
S = 400                 # tamaño de la vista frontal
FORCE_YAML_SIZE = False # si True, fuerza el tamaño de 'EuRoC_stereo.yaml'

# Cargar parámetros de cámara
def load_cam_yaml(path):
    with open(path, 'r') as f:
        cam = yaml.safe_load(f).get('Camera', yaml.safe_load(f))
    K = np.array([[cam['fx'], 0, cam['cx']], [0, cam['fy'], cam['cy']], [0, 0, 1]], dtype=np.float32)
    D = np.array([float(cam.get(k, 0)) for k in ('k1','k2','p1','p2','k3')], dtype=np.float32)
    return K, D, (cam['cols'], cam['rows'])

K, D, (W, H) = load_cam_yaml('EuRoC_stereo.yaml')
dst = np.float32([[0,0],[S,0],[S,S],[0,S]])

# Variables globales
mode, clicks, last_points, Hmat = 'viz', [], [], None
cam = cv2.VideoCapture(CAP)
qr = cv2.QRCodeDetector()

def draw_grid_and_front(frame, Hmat):
    disp = frame.copy()
    invH = np.linalg.inv(Hmat)
    
    # Dibujar grilla
    for i in range(GRID_N + 1):
        t = i / GRID_N
        # Líneas verticales y horizontales
        v_pts = np.float32([[[t*S, 0], [t*S, S]]])
        h_pts = np.float32([[[0, t*S], [S, t*S]]])
        v = cv2.perspectiveTransform(v_pts, invH)[0]
        h = cv2.perspectiveTransform(h_pts, invH)[0]
        
        cv2.line(disp, tuple(map(int, np.round(v[0]))), tuple(map(int, np.round(v[1]))), (0, 255, 255), 1)
        cv2.line(disp, tuple(map(int, np.round(h[0]))), tuple(map(int, np.round(h[1]))), (0, 255, 255), 1)
    
    # Vista frontal
    cv2.imshow("front_view", cv2.warpPerspective(frame, Hmat, (S, S)))
    return disp

def on_mouse(event, x, y, flags, param):
    global clicks, mode, Hmat, last_points
    if mode == 'click' and event == cv2.EVENT_LBUTTONDOWN:
        clicks.append([x, y])
        last_points = clicks.copy()
        if len(clicks) == 4:
            try:
                Hmat = cv2.getPerspectiveTransform(np.float32(clicks), dst)
                print("[ok] Homografía calculada desde clicks")
            except:
                print("[error] No se pudo calcular homografía")
            clicks, mode = [], 'viz'

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', on_mouse)

while True:
    ok, frame = cam.read()
    if not ok: break
    
    # Preprocesamiento
    if FORCE_YAML_SIZE and (frame.shape[1], frame.shape[0]) != (W, H):
        frame = cv2.resize(frame, (W, H))
    if np.any(D != 0):
        frame = cv2.undistort(frame, K, D)
    
    disp = frame.copy()
    
    # Modo QR
    if mode == 'qr':
        retval, points = qr.detect(frame)
        if retval and points is not None and len(points) == 1 and points[0].shape[0] == 4:
            try:
                Hmat = cv2.getPerspectiveTransform(points[0].astype(np.float32), dst)
                print("[ok] Homografía calculada desde QR")
            except:
                print("[error] No se pudo calcular homografía desde QR")
        mode = 'viz'  # vuelve a viz después de intentar detectar QR
    
    # Visualización
    if Hmat is not None:
        disp = draw_grid_and_front(frame, Hmat)
    
    # Dibujar puntos
    for i, (x, y) in enumerate(clicks if mode == 'click' else []):
        cv2.circle(disp, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(disp, str(i+1), (x+6, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    for i, (x, y) in enumerate(last_points):
        cv2.circle(disp, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(disp, str(i+1), (int(x)+6, int(y)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Instrucciones
    msgs = {
        'qr': "MODO QR: Presiona cualquier tecla para computar homografia",
        'click': f"MODO CLICK: {len(clicks)}/4 puntos - Presiona cualquier tecla para abortar"
    }
    cv2.putText(disp, msgs.get(mode, f"mode: {mode}  (q=QR, h=click, c=clear H, ESC=salir)"), 
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow('frame', disp)
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27: break
    elif k == ord('q'): mode = 'qr'
    elif k == ord('h'): mode, clicks = 'click', []
    elif k == ord('c'): Hmat, last_points = None, []
    elif mode == 'click' and k != 255: clicks, mode = [], 'viz'

cam.release()
cv2.destroyAllWindows()
