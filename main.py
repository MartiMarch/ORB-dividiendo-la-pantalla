"""
    Algoritmo ORB. La imagen es dividida en n partes y se buscan
    los puntos clave en cada una.
"""
import cv2

def procesarSubimagen(imagen, x1, x2, y1, y2):
    subimagen = imagen[x1:x2, y1:y2]
    puntosClave = orb.detect(subimagen, None)
    puntosClave, descriptores = orb.compute(subimagen, puntosClave)
    subimagen = cv2.drawKeypoints(subimagen, puntosClave, outImage=None, color=(255, 0, 0))
    return subimagen

# Se abre la cámara
camara = cv2.VideoCapture(2)

# Se crea un objeto ORB
orb = cv2.ORB_create(50)

while True:
    # Se consume una imagen
    _, imagen = camara.read()

    # Se crea un punto clave
    puntosClave = orb.detect(imagen, None)

    # Se obtiene los descriptores
    puntosClave, descriptores = orb.compute(imagen, puntosClave)

    # Se dibujan los puntos clave
    imagenConPuntos = cv2.drawKeypoints(imagen, puntosClave, outImage = None, color = (255, 0, 0))

    # Se muestran los puntos clave
    cv2.imshow("ORB pantalla completa", imagenConPuntos)

    # Dividimos la imagen en cuatro, obtenemos sus puntos clave y los dibujamos
    subimagen1 = procesarSubimagen(imagen, 0, int(imagen.shape[0]/2), 0, int(imagen.shape[1]/2))
    subimagen2 = procesarSubimagen(imagen, 0, int(imagen.shape[0]/2), int(imagen.shape[1]/2), int(imagen.shape[1]))
    subimagen3 = procesarSubimagen(imagen, int(imagen.shape[0]/2), int(imagen.shape[0]), 0, int(imagen.shape[1]/2))
    subimagen4 = procesarSubimagen(imagen, int(imagen.shape[0]/2), int(imagen.shape[0]), int(imagen.shape[1]/2), int(imagen.shape[1]))
    
    # Mostramos la subimagenes
    cv2.imshow("ORB subimagen 1", subimagen1)
    cv2.imshow("ORB subimagen 2", subimagen2)
    cv2.imshow("ORB subimagen 3", subimagen3)
    cv2.imshow("ORB subimagen 4", subimagen4)

    # Pulsar 'q' para salir
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break