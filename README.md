# Hand-Gesture-Recognition-GUI
A Python program that uses computer vision and machine learning to recognize hand gestures from a live video feed. It displays the recognized gesture on the screen, shows the video in black and white, and provides the status of each finger (thumb, index, middle, ring, and little) as extended (1) or not (0).

It identifies the hand landmarks using the MediaPipe library, then feeds these landmarks into a pre-trained neural network model to predict the hand gesture. The recognized gesture is displayed on the screen, and the video feed is shown side-by-side with a black and white version of the video. It is designed to detect 10 basic features, which are: okay, peace, thumbs up, thumbs down, call me, stop, rock, live long, fist, and smile.

MediaPipe: A library developed by Google that provides pre-trained models for hand landmark detection.
TensorFlow: A machine learning framework used to load the pre-trained neural network model for hand gesture recognition.
tkinter: The standard Python interface to the Tk GUI toolkit, used to create the graphical user interface (GUI).

## Components present in the program
Hand Landmark Detection: The program uses the MediaPipe library to detect hand landmarks (key points) from the live video feed. These landmarks are used as input to the gesture recognition model.

Gesture Recognition Model: The program loads a pre-trained neural network model using TensorFlow and uses it to predict the hand gesture based on the detected landmarks.

Video Stream Processing: The application captures the video feed from the webcam using OpenCV, processes each frame, and displays it in the GUI.

GUI (Graphical User Interface): The tkinter library is used to create the GUI for the Hand Gesture Recognition App. The GUI includes buttons to start and stop the recognition process, video streams, and a status bar that displays the detected gesture and finger status.

## How to use this program
1. Install the required libraries: Ensure you have installed the necessary libraries like OpenCV, MediaPipe, TensorFlow, and tkinter. You can install them using pip:
pip install opencv-python mediapipe tensorflow tkinter
2. Load the Pre-trained Models: Place the pre-trained hand gesture recognition model (mp_hand_gesture) and the gesture names file (gesture.names) in the specified locations (as mentioned in the program).
3. Run the Application: Run the Python script containing the HandGestureRecognitionApp class.
4. Start Recognition: Click the "Start" button to start the gesture recognition process. The program will capture the video from the webcam and begin recognizing hand gestures.
5. Stop Recognition: Click the "Stop" button to stop the recognition process. The video stream will freeze, and the gesture recognition will pause.
6. Gesture and Finger Status: The detected hand gesture will be displayed on the video stream. Additionally, the status bar under the video streams will show the current gesture recognized.

The project uses TechVidvan's tutorial as a starting point (https://techvidvan.com/tutorials/hand-gesture-recognition-tensorflow-opencv/).
