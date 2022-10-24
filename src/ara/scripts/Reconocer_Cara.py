#!/usr/bin/env python
# Prueba1 
# Video 
# Author:RUBY
# Import the necessary libraries
import rospy # Python library for ROS
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
from std_msgs.msg import String

import cv2
import numpy as np
from keras.preprocessing import image
from datetime import datetime, time, timedelta

from keras.models import model_from_json
model = model_from_json(open('/home/ar4/ros_workspace/src/ara/scripts/facial_expression_model_structure.json', "r").read())
model.load_weights('/home/ar4/ros_workspace/src/ara/scripts/facial_expression_model_weights.h5') #load weights

#LOADING HAND CASCADE

hand_cascade = cv2.CascadeClassifier('/home/ar4/ros_workspace/src/ara/scripts/Hand_haar_cascade.xml')
face_cascade = cv2.CascadeClassifier('/home/ar4/ros_workspace/src/ara/scripts/haarcascade_frontalface_alt.xml')

def getaverageemotions(emotionsperminute):

	average = []
	maxvalue = 0

	for i in range(0,7):
		average.append(emotionsperminute.count(i))

	maxvalue = max(average)

	return average.index(maxvalue)
def Reconocer_Cara():
	pub_e = rospy.Publisher('emotion', String, queue_size=10)
	pub_h = rospy.Publisher('hand', String, queue_size=10)
	rospy.init_node('emotion_hand_rec', anonymous=True)
	rate = rospy.Rate(10) # 10hz
	#Tamano para reducir a miniaturas las fotografias
	size = 4

	cap = cv2.VideoCapture(0)
	
	img_width, img_height = 520, 520#200
	
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)#320
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)#240
	
	#print("Introduzca tiempo de la sesion (5,10,15,30,45)")
	#sessiontime = input()
	sessiontime = 5
	hourformat = "%H:%M:%S"
	starttime = datetime.now()
	finishtime = starttime + timedelta(minutes=sessiontime)
	print("Hora de Inicio de sesion:", starttime.strftime(hourformat))
	print("Hora de Termino de sesion:", finishtime.strftime(hourformat))
	
	emotionsperminute = []
	averageemotions = []
	numericalpercentage = 0
	
	timeaverage = starttime + timedelta(minutes=1)
	
	while((datetime.now()).strftime(hourformat) != finishtime.strftime(hourformat)):
	
		width = cap.get(3)  # Ancho video
		height = cap.get(4) # Altura video
	
		rval, img = cap.read()
		blur = cv2.GaussianBlur(img,(5,5),0) # BLURRING IMAGE TO SMOOTHEN EDGES
		img = cv2.flip(img, 1, 0)
		
		#convertimos la imagen a blanco y negro
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		#redimensionar la imagen
		mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))
		
		#"buscamos las coordenadas de los rostros (si los hay) y guardamos su posicion"
		faces = face_cascade.detectMultiScale(mini)
		faces = sorted(faces, key=lambda x: x[3])
		
		retval2,thresh1 = cv2.threshold(gray,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) # THRESHOLDING IMAGE
		hand = hand_cascade.detectMultiScale(thresh1, 1.3, 5) # DETECTING HAND IN THE THRESHOLDE IMAGE
		mask = np.zeros(thresh1.shape, dtype = "uint8") # CREATING MASK	
		
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
			###########################################		############
			emotion_str = emotion #% rospy.get_time()
			rospy.loginfo(emotion_str)
			pub_e.publish(emotion_str)
			emotionsperminute.append(max_index)
	
			if ((datetime.now()).strftime(hourformat) == timeaverage.strftime(hourformat)):
			
				averageemotions.append(getaverageemotions(emotionsperminute))
				emotionsperminute = []
				timeaverage = timeaverage + timedelta(minutes=1)
			
			#Dibujamos un rectangulo en las coordenadas del rostro
			cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)    
			cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
			
			if len(hand) <= 1:
				h_str=""
				for (x,y,w,h) in hand: # MARKING THE DETECTED ROI
					
					cv2.rectangle(img,(x,y),(x+w,y+h), (122,122,0), 2) 
					cv2.rectangle(mask, (x,y),(x+w,y+h),255,-1)
	
					if (x <= int(width/2) and y <= int(height/2)):
						h_str='Left'
						cv2.putText(img, h_str, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, 	(255,255,255), 2)
	
					if (x >= int(width/2) and y <= int(height/2)):
						h_str='Right'
						cv2.putText(img, h_str, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, 	(255,255,255), 2)
				
				img2 = cv2.bitwise_and(thresh1, mask)
				final = cv2.GaussianBlur(img2,(7,7),0)	
				contours, hierarchy = cv2.findContours(final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
				#_, contours, _= cv2.findContours(final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
				cv2.drawContours(img, contours, 0, (255,255,0), 3)
				cv2.drawContours(final, contours, 0, (255,255,0), 3)		###########################################		############
				hand_str = h_str #% rospy.get_time()
				rospy.loginfo(hand_str)
				pub_h.publish(hand_str)
		#Mostramos la imagen
		cv2.imshow('img2',img)

		#rospy.spin()		
		#con la tecla 'q' salimos del programa
		if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
			break
	
	cap.release()
	cv2.destroyAllWindows()
	
	for i in range(0,7):
		if(averageemotions.count(i) != 0):
			numericalpercentage = ((averageemotions.count(i))*100)/sessiontime
			print(emotions[i] , numericalpercentage)
	rospy.spin()
	
if __name__ == '__main__':
  try:
    Reconocer_Cara()
  except rospy.ROSInterruptException:
    pass
