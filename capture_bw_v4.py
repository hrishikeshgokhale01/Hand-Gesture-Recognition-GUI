import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model(r"C:\Users\mukun\Desktop\hand-gesture-recognition-code\mp_hand_gesture")

# Load class names
f = open(r"C:\Users\mukun\Desktop\hand-gesture-recognition-code\gesture.names", 'r')
classNames = f.read().split('\n')
f.close()

class HandGestureRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Recognition")

        self.start_button = ttk.Button(self.root, text="Start", command=self.start_recognition)
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.stop_button = ttk.Button(self.root, text="Stop", command=self.stop_recognition)
        self.stop_button.pack(side=tk.LEFT, padx=10)
        self.stop_button.configure(state="disabled")  # Disable stop button initially

        self.is_recognizing = False

        # Create a frame to hold video and finger options
        self.frame = ttk.Frame(self.root)
        self.frame.pack()

        self.video_label = ttk.Label(self.frame)
        self.video_label.pack(side=tk.LEFT, padx=10)

        self.bw_video_label = ttk.Label(self.frame)
        self.bw_video_label.pack(side=tk.LEFT, padx=10)

        # Status Bar
        self.status_bar = ttk.Label(self.root, text="Gesture: None", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        self.status_bar.config(background='lightgray')  # Add a background color to make it more prominent

        # About Menu
        menubar = tk.Menu(self.root)
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about_dialog)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.root.config(menu=menubar)

    def start_recognition(self):
        if not self.is_recognizing:
            self.is_recognizing = True
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.capture_video_with_bw_conversion()

    def stop_recognition(self):
        self.is_recognizing = False
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

    def show_about_dialog(self):
        messagebox.showinfo("About", "Hand Gesture Recognition App\n\nDeveloped using Python and tkinter.\nVersion: 1.0\n\n© 2023 YourName")

    def capture_video_with_bw_conversion(self):
        cap = cv2.VideoCapture(0)

        while self.is_recognizing:
            _, frame = cap.read()
            x, y, c = frame.shape
            frame = cv2.flip(frame, 1)
            framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get hand landmark prediction
            result = hands.process(framergb)

            gesture_detected = 'None'

            if result.multi_hand_landmarks:
                for handslms in result.multi_hand_landmarks:
                    landmarks = []
                    for lm in handslms.landmark:
                        lmx = int(lm.x * x)
                        lmy = int(lm.y * y)
                        landmarks.append([lmx, lmy])

                    # Drawing landmarks on frames
                    mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                    # Predict gesture using the loaded model
                    prediction = model.predict([landmarks])
                    classID = np.argmax(prediction)
                    gesture_detected = classNames[classID]

            # Show the prediction on the frame
            cv2.putText(frame, gesture_detected, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)

            # Convert the frame to black and white
            frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, frame_bw = cv2.threshold(frame_bw, 128, 255, cv2.THRESH_BINARY)

            # Convert the frame to PIL format for displaying in tkinter
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=image)

            self.video_label.config(image=photo)
            self.video_label.image = photo

            # Convert the black and white frame to PIL format for displaying in tkinter
            image_bw = Image.fromarray(frame_bw)
            photo_bw = ImageTk.PhotoImage(image=image_bw)

            self.bw_video_label.config(image=photo_bw)
            self.bw_video_label.image = photo_bw

            # Update the status bar with the current gesture detected
            self.status_bar.config(text=f"Gesture: {gesture_detected}")

            self.root.update()  # Update the GUI

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = HandGestureRecognitionApp(root)
    root.mainloop()
