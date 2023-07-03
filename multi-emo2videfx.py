import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.utils import img_to_array
from IPython.display import clear_output
import argparse
import imutils
import cv2
from keras.models import load_model
import numpy as np
import video_modifications as vm
import tensorflow as tf

# Parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# Load models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# List of emotions and setting the frame interval
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral"]

class EmotionRecognizer:
    def __init__(self, neutral_accepted, interval_frames, fps):
        self.fps = fps
        self.neutral_accepted = neutral_accepted
        self.interval_frames = interval_frames
        self.neutral_index = EMOTIONS.index('neutral') if 'neutral' in EMOTIONS else None
    

    def recognize_emotions(self, frames):
        # Initialize emotion sums and frame count
        emotion_sums = np.zeros(len(EMOTIONS) if self.neutral_accepted else len(EMOTIONS) - 1)
        frame_count = 0

        # Initialize highest_average_emotion and emotion_percentage
        highest_average_emotion = None
        emotion_percentage = 0.0

        for frame in frames:
            # Frame preprocessing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.HOUGH_MULTI_SCALE)

            if len(faces) > 0:
                for (fX, fY, fW, fH) in faces:   
                    roi = cv2.resize(gray[fY:fY + fH, fX:fX + fW], (64, 64)).astype("float") / 255.0
                    roi = np.expand_dims(img_to_array(roi), axis=0)
                    
                    preds = emotion_classifier.predict(roi, verbose=0)[0]
                    if not self.neutral_accepted and self.neutral_index is not None:
                        preds = np.delete(preds, self.neutral_index)
                    emotion_sums += preds

                frame_count += 1
                if frame_count % self.interval_frames == 0:
                    # Calculate the average predictions and the highest average emotion
                    average_emotions = emotion_sums / frame_count
                    highest_emotion_index = np.argmax(average_emotions)
                    highest_average_emotion = EMOTIONS[highest_emotion_index]
                    
                    # Calculate the percentage of the highest average emotion
                    emotion_percentage = 100 * average_emotions[highest_emotion_index] / np.sum(average_emotions)
                    
                    print(f"Average emotion at {frame_count / self.fps} seconds: {highest_average_emotion} ({emotion_percentage:.2f}%)")

                    # Reset counters
                    emotion_sums = np.zeros(len(EMOTIONS))
                    frame_count = 0

        return highest_average_emotion, emotion_percentage

class ModifyVideo:
    def __init__(self, neutral_accepted=True, interval_seconds=1):
        self.neutral_accepted = neutral_accepted
        self.interval_seconds = interval_seconds

    def process_video(self, emotion_video_path, effect_video_path, output_path):
        # Capture both videos
        cap_emotion = cv2.VideoCapture(emotion_video_path)
        cap_effect = cv2.VideoCapture(effect_video_path)
        
        fps = cap_emotion.get(cv2.CAP_PROP_FPS)
        self.interval_frames = int(fps * self.interval_seconds)
        self.emotion_recognizer = EmotionRecognizer(neutral_accepted=self.neutral_accepted, interval_frames=self.interval_frames, fps=fps)
        frame_count = 0
        interval_count = 0
        dominant_emotion = None
        emotion_intensity = 0.0
        frames = []

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        print(f"Processing video...")

        while cap_emotion.isOpened() and cap_effect.isOpened():
            ret_emotion, frame_emotion = cap_emotion.read()
            ret_effect, frame_effect = cap_effect.read()
            if not ret_emotion or not ret_effect:
                break

            frame_count += 1
            interval_count += 1
            frames.append(frame_emotion)
            print(frame_count, end='\r')

            if interval_count == self.interval_frames:
                interval_count = 0
                dominant_emotion, emotion_intensity = self.emotion_recognizer.recognize_emotions(frames)
                frames = []

            if dominant_emotion == 'happy':
                frame_effect = vm.threshold_image(frame_effect, emotion_intensity)
            elif dominant_emotion == 'sad':
                frame_effect = vm.edge_detection(frame_effect, emotion_intensity)
            elif dominant_emotion == 'angry':
                frame_effect = vm.desaturate_and_blur(frame_effect, emotion_intensity)
            elif dominant_emotion == 'neutral':
                frame_effect = vm.invert_colors(frame_effect, emotion_intensity)
            elif dominant_emotion == 'surprised':
                frame_effect = vm.rotate_image(frame_effect, emotion_intensity)
            elif dominant_emotion == 'scared':
                frame_effect = vm.add_noise(frame_effect, emotion_intensity)
            elif dominant_emotion == 'disgust':
                frame_effect = vm.invert_colors(frame_effect)

            # Add overlay
            # Convert emotion_intensity to string
            emotion_intensity_str = str(dominant_emotion)+" : "+str(emotion_intensity)
            frame_effect = vm.add_text_overlay(frame_effect, emotion_intensity_str)

            if out is None:
                out = cv2.VideoWriter(output_path, fourcc, fps, (frame_effect.shape[1], frame_effect.shape[0]))
            out.write(frame_effect)

        cap_emotion.release()
        cap_effect.release()
        if out is not None:
            out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a video based on detected emotions.')
    parser.add_argument('--emotion_video_path', type=str, help='Path to the video file that holds the emotions.', required=True)
    parser.add_argument('--effect_video_path', type=str, help='Path to the video file that you want to modify.', required=True)
    parser.add_argument('--output', type=str, help='Path to the output file.', required=True)
    parser.add_argument('--interval', type=float, help='Amount of seconds to calculate effect for.', required=True)
    parser.add_argument('--neutral_accepted', action='store_true', help='Accept neutral as an emotion. If not provided, defaults to False.')

    args = parser.parse_args()

    modifier = ModifyVideo(neutral_accepted=args.neutral_accepted, interval_seconds=args.interval)
    modifier.process_video(args.emotion_video_path, args.effect_video_path, args.output)

