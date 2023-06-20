import cv2
import random
import webbrowser#for link to youtube
import numpy as np#used to convert the video frames to NumPy arrays,store the results of the holistic and hands solutions
import mediapipe as mp #MediaPipe library to detect facial landmarks, hand landmarks, and draw them on the frames
                        #collection of machine learning solutions for computer vision and augmented reality.
from keras.models import load_model #For creating the neural network model.


model  = load_model("model.h5")#pre-trained neural network model
label = np.load("labels.npy")



holistic = mp.solutions.holistic#used to detect and track human body keypoints, including the face, hands, and body
hands = mp.solutions.hands
holis = holistic.Holistic()# creates a new Holistic object. The Holistic object is used to detect and track human body keypoints
                            #part of the MediaPipe library
drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

class Node:
    def __init__(self, emotion):
        self.emotion = emotion
        self.recommendations = []

# Create nodes for different emotions
happy_node = Node("happy")
sad_node = Node("sad")
angry_node = Node("angry")

# Create edges between emotions
happy_node.recommendations.extend(["https://www.youtube.com/watch?v=TdrL3QxjyVw&pp=ygUSc3VtbWVydGltZSBzYWRuZXNz",
                                   "https://www.youtube.com/watch?v=TdrL3QxjyVw&pp=ygUSc3VtbWVydGltZSBzYWRuZXNz",
                                   "https://www.youtube.com/watch?v=2Vv-BfVoq4g&pp=ygUsZGFuY2luZyBpbiB0aGUgZGFyayB3aXRoIHlvdSBiZXR3ZWVuIG15IGFybXM%3D"])
sad_node.recommendations.extend(["https://www.youtube.com/watch?v=HZgiAgYXneE&pp=ygUIc2l4IGZlZXQ%3D",
                                 "https://www.youtube.com/watch?v=V1Pl8CzNzCw&pp=ygUUbG9uZWx5IGJpbGxpZSBlaWxpc2g%3D",
                                 "https://www.youtube.com/watch?v=V1Pl8CzNzCw&pp=ygUUbG9uZWx5IGJpbGxpZSBlaWxpc2g%3D"])
angry_node.recommendations.extend(["https://www.youtube.com/watch?v=mWRsgZuwf_8&pp=ygUGZGVtb25z",
                                   "https://www.youtube.com/watch?v=7wtfhZwyrcc&pp=ygUNYmVsaWV2ZXIgc29uZw%3D%3D",
                                   "https://www.youtube.com/watch?v=PDeTO26fRVQ&pp=ygUEaWRmdg%3D%3D"])






for i in range(50):
        lst = []#is used to store the x and y coordinates of the face landmarks

        _, frm = cap.read()#cap variable is a reference to the webcam
                            # read() method returns a tuple of two values:1)
                            #indicates whether or not the frame was read successfully, and the second value is the actual frame data.

        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))#processes the frame and performs holistic analysis, including face detection


        if res.face_landmarks:#checks if any face landmarks were detected in the processed frame. If face landmarks are present, it enters the loop.
                for i in res.face_landmarks.landmark:# iterates over each detected face landmark.
                                                    #The landmark attribute of res.face_landmarks provides access to the individual face landmarks.
                        lst.append(i.x - res.face_landmarks.landmark[1].x)#The landmark attribute of res.face_landmarks provides access to the individual face landmarks.
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

                lst = np.array(lst).reshape(1,-1)

                pred = label[np.argmax(model.predict(lst))]#predicts the emotion according to the pretrained model

                print(pred)
                emotionlist=[]
                emotionlist.append(pred)
                
                cv2.putText(frm, pred, (50,50),cv2.FONT_ITALIC, 1, (255,0,0),2)

                
        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)# function to draw the detected face and hand
                                                                                    #landmarks on the frame, using the FACEMESH_CONTOURS
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        cv2.imshow("window", frm)

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
# Dictionary to map emotions to nodes
emotion_nodes = {
    "happy": happy_node,
    "sad": sad_node,
    "angry": angry_node,
    # Add more emotions and corresponding nodes
}
controller = webbrowser.get()
ang=emotionlist.count("angry")
hap=emotionlist.count("happy")
sed=emotionlist.count("sad")
if(ang>hap)and(ang>sed):
    current_emotion="angry"
elif(hap>ang and hap>sed):
    current_emotion="happy"
elif(sed>hap) and (sed>ang):
    current_emotion="sad"
# Assuming 'current_emotion' contains the real-time emotion value
if current_emotion in emotion_nodes:
    node = emotion_nodes[current_emotion]
    recommended_songs = node.recommendations

    if recommended_songs:
        # Select a random song from the list
        song_to_play = random.choice(recommended_songs)
        print("Playing song:",controller.open(song_to_play))
        # Your code to play the selected song goes here
    else:
        print("No music recommendations for the current emotion.")
else:
    print("Invalid emotion.")




                
cv2.destroyAllWindows()
cap.release()
