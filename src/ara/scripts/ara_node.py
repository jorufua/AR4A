#!/usr/bin/env python
import rospy # Python library for ROS
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
from std_msgs.msg import String

import cv2
import face_recognition
import numpy as np
from keras.preprocessing import image
from datetime import datetime, time, timedelta

from keras.models import model_from_json
model = model_from_json(open('/home/alexmorfin/ros_workspace/src/ara/scripts/facial_expression_model_structure.json', "r").read())
model.load_weights('/home/alexmorfin/ros_workspace/src/ara/scripts/facial_expression_model_weights.h5') #load weights

#LOADING HAND CASCADE

hand_cascade = cv2.CascadeClassifier('/home/alexmorfin/ros_workspace/src/ara/scripts/Hand_haar_cascade.xml')
face_cascade = cv2.CascadeClassifier('/home/alexmorfin/ros_workspace/src/ara/scripts/haarcascade_frontalface_alt.xml')

ruben_image = face_recognition.load_image_file("/home/alexmorfin/ros_workspace/src/ara/scripts/img/known/724306.jpg")
ruben_face_encoding = face_recognition.face_encodings(ruben_image)[0]

alex_image = face_recognition.load_image_file("/home/alexmorfin/ros_workspace/src/ara/scripts/img/known/185430.jpg")
alex_face_encoding = face_recognition.face_encodings(alex_image)[0]

luca_image = face_recognition.load_image_file("/home/alexmorfin/ros_workspace/src/ara/scripts/img/known/luca.jpeg")
luca_face_encoding = face_recognition.face_encodings(luca_image)[0]


def getaverageemotions(emotionsperminute):

	average = []
	maxvalue = 0

	for i in range(0,7):
		average.append(emotionsperminute.count(i))

	maxvalue = max(average)

	return average.index(maxvalue)
def Reconocer_ID_emociones():
	pub_id = rospy.Publisher('identity', String, queue_size=10)
	pub_e = rospy.Publisher('emotion', String, queue_size=10)
	pub_h = rospy.Publisher('hand', String, queue_size=10)

	rospy.init_node('ara_node', anonymous=True)
	rate = rospy.Rate(10) # 10hz
	# Get a reference to webcam #0 (the default one)
	size = 4
	video_capture = cv2.VideoCapture(0)
	
	img_width, img_height = 520, 520#200
	
	video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)#320
	video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)#240
	
	# Create arrays of known face encodings and their names
	known_face_encodings = [
	    ruben_face_encoding,
	    alex_face_encoding,
            luca_face_encoding]

	known_face_names = [
	    "Ruben",
	    "Alex",
            "Luca"]

# Initialize some variables
	face_locations = []
	face_encodings = []
	face_names = []
	process_this_frame = True
	id_str=[]
	str1=""
####################################################
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
	#while True:
    	# Grab a single frame of video
		width = video_capture.get(3)  # Ancho video
		height = video_capture.get(4) # Altura video
	
		#rval, img = cap.read()
		ret, frame = video_capture.read()

		blur = cv2.GaussianBlur(frame,(5,5),0) # BLURRING IMAGE TO SMOOTHEN EDGES
		frame = cv2.flip(frame, 1, 0)
		#convertimos la imagen a blanco y negro
    		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#redimensionar la imagen
		mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))
		#"buscamos las coordenadas de los rostros (si los hay) y guardamos su posicion"
		faces = face_cascade.detectMultiScale(mini)
		faces = sorted(faces, key=lambda x: x[3])
		retval2,thresh1 = cv2.threshold(gray,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) # THRESHOLDING IMAGE
		hand = hand_cascade.detectMultiScale(thresh1, 1.3, 5) # DETECTING HAND IN THE THRESHOLDE IMAGE
		mask = np.zeros(thresh1.shape, dtype = "uint8") # CREATING MASK	

    # Resize frame of video to 1/4 size for faster face recognition processing
    		#small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
		small_frame = cv2.resize(frame, (img_width, img_height))
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    		rgb_small_frame = small_frame[:, :, ::-1]
		
    # Only process every other frame of video to save time
    		if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        		face_locations = face_recognition.face_locations(rgb_small_frame)
        		face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

		        face_names = []
			
	        	for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            			matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            			name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            			face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            			best_match_index = np.argmin(face_distances)
            			if matches[best_match_index]:
                			name = known_face_names[best_match_index]

            			face_names.append(name)

    		process_this_frame = not process_this_frame


    # Display the results
    		for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        		top *= 4
        		right *= 4
        		bottom *= 4
       			left *= 4

        # Draw a box around the face
        		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        		cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        		font = cv2.FONT_HERSHEY_DUPLEX
        		cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
		
		id_str = face_names#% rospy.get_time()
		id_str = str1.join(id_str)
		rospy.loginfo(id_str)#id_str
		pub_id.publish(id_str)


		if faces:
		
			face_i = faces[0]
			(x, y, w, h) = [v * size for v in face_i]
			face = gray[y:y + h, x:x + w]
			face_resize = cv2.resize(face, (img_width, img_height))
		
			detected_face = frame[int(y):int(y+h), int(x):int(x+w)] #crop detected face
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
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)    
			cv2.putText(frame, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
			
			if len(hand) <= 1:
				h_str=""
				for (x,y,w,h) in hand: # MARKING THE DETECTED ROI
					
					cv2.rectangle(frame,(x,y),(x+w,y+h), (122,122,0), 2) 
					cv2.rectangle(mask, (x,y),(x+w,y+h),255,-1)
	
					if (x <= int(width/2) and y <= int(height/2)):
						h_str='Left'
						cv2.putText(frame, h_str, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, 	(255,255,255), 2)
	
					if (x >= int(width/2) and y <= int(height/2)):
						h_str='Right'
						cv2.putText(frame, h_str, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, 	(255,255,255), 2)
				
				img2 = cv2.bitwise_and(thresh1, mask)
				final = cv2.GaussianBlur(img2,(7,7),0)	
				contours, hierarchy = cv2.findContours(final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
				#_, contours, _= cv2.findContours(final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
				cv2.drawContours(frame, contours, 0, (255,255,0), 3)
				cv2.drawContours(final, contours, 0, (255,255,0), 3)		###########################################		############
				hand_str = h_str #% rospy.get_time()
				rospy.loginfo(hand_str)
				pub_h.publish(hand_str)
		#Mostramos la imagen
		#cv2.imshow('img2',img)
    # Display the resulting image
    		cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    		if cv2.waitKey(1) & 0xFF == ord('q'):
        		break

# Release handle to the webcam
	video_capture.release()
	cv2.destroyAllWindows()
		
	for i in range(0,7):
		if(averageemotions.count(i) != 0):
			numericalpercentage = ((averageemotions.count(i))*100)/sessiontime
			print(emotions[i] , numericalpercentage)
	rospy.spin()

	
if __name__ == '__main__':
  try:
    Reconocer_ID_emociones()
  except rospy.ROSInterruptException:
    pass
