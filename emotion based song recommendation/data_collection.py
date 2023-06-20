import mediapipe as mp #MediaPipe library to detect facial landmarks, hand landmarks, and draw them on the frames
import numpy as np 
import cv2 
 
cap = cv2.VideoCapture(0)

name = input("Enter the name of the data : ")

holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

X = []
data_size = 0

while True:
	lst = []

	_, frm = cap.read()

	frm = cv2.flip(frm, 1)#flip the img horizontally

	res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))#converts the color space of the frm frame from BGR (Blue-Green-Red) to RGB (Red-Green-Blue). 


	if res.face_landmarks:#checks if any face landmarks were detected in the processed frame. If face landmarks are present, it enters the loop.
		for i in res.face_landmarks.landmark:# iterates over each detected face landmark.
                                                        #The landmark attribute of res.face_landmarks provides access to the individual face landmarks.
			lst.append(i.x - res.face_landmarks.landmark[1].x)#difference in the x and y-coordinate of the current face landmark (i.x)
			lst.append(i.y - res.face_landmarks.landmark[1].y)#and the reference x-coordinate of the second face landmark 

		if res.left_hand_landmarks:
			for i in res.left_hand_landmarks.landmark:
				lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
				lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
		else:
			for i in range(42):
				lst.append(0.0)

		if res.right_hand_landmarks:
			for i in res.right_hand_landmarks.landmark:
				lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
				lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
		else:
			for i in range(42):
				lst.append(0.0)


		X.append(lst)#positional differences of the detected landmarks with respect to specific references points
		data_size = data_size+1



	drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
	drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
	drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

	cv2.putText(frm, str(data_size), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

	cv2.imshow("window", frm)

	if cv2.waitKey(1) == 27 or data_size>99:
		cv2.destroyAllWindows()
		cap.release()
		break


np.save(f"{name}.npy", np.array(X))
print(np.array(X).shape)
