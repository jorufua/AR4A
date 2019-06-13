#OpenCV module
import cv2
import numpy as np
from keras.preprocessing import image

from keras.models import model_from_json
model = model_from_json(open("facial_expression_model_structure.json", "r").read())
model.load_weights('facial_expression_model_weights.h5') #load weights

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

#Tamano para reducir a miniaturas las fotografias
size = 4

cap = cv2.VideoCapture(0)

img_width, img_height = 200, 200

#count = 0

while(cap.isOpened()):
#while(count < 25):
    
    rval, img = cap.read()
    img = cv2.flip(img, 1, 0)

    #convertimos la imagen a blanco y negro
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #redimensionar la imagen
    mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

    #"buscamos las coordenadas de los rostros (si los hay) y guardamos su posicion"
    faces = face_cascade.detectMultiScale(mini)    
    faces = sorted(faces, key=lambda x: x[3])

    if faces:

	face_i = faces[0]
	(x, y, w, h) = [v * size for v in face_i]
	face = gray[y:y + h, x:x + w]
	face_resize = cv2.resize(face, (img_width, img_height))

        detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
        detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48

        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255
        predictions = model.predict(img_pixels)
       
        #find max indexed array
        max_index = np.argmax(predictions[0])
 
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        emotion = emotions[max_index]
		
	#Dibujamos un rectangulo en las coordenadas del rostro
	cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)    

	#Metemos la foto en el directorio
	#cv2.imwrite("face-" + str(count) + ".jpg", face_resize)

	#Contador del ciclo 
	#count += 1

    #Mostramos la imagen
    cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow('img',img)

    #con la tecla 'q' salimos del programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
	break;

cap.release()
cv2.destroyAllWindows()
