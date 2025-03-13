import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
import shutil

# Directories
input_dir = 'input'
output_dir = 'output'
copy_dir = 'copy'
pose_dir = os.path.join(output_dir, 'B')
landmarks_dir = os.path.join(output_dir, 'C')

# Create directories if not exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(copy_dir, exist_ok=True)
os.makedirs(pose_dir, exist_ok=True)
os.makedirs(landmarks_dir, exist_ok=True)

# Initialize MediaPipe models
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.5)

def process_image(image_path, output_pose_path, output_landmarks_image_path, copy_image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read: {image_path}")
        return False

    # Resize image to 512x512
    resized_image = cv2.resize(image, (512, 512))

    # Detect face landmarks
    results_face = face_mesh.process(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))

    if results_face.multi_face_landmarks:
        # If face landmarks are detected, save resized image to copy directory
        cv2.imwrite(copy_image_path, resized_image)

        # Draw landmarks on a blank image
        landmarks_image = np.zeros((512, 512, 3), dtype=np.uint8)
        face_landmarks = results_face.multi_face_landmarks[0]

        for landmark in face_landmarks.landmark:
            x = int(landmark.x * 512)
            y = int(landmark.y * 512)
            cv2.circle(landmarks_image, (x, y), 2, (255, 255, 255), -1)

        # Save landmarks image
        cv2.imwrite(output_landmarks_image_path, landmarks_image)

        # Detect pose landmarks
        results_pose = pose.process(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        pose_image = np.zeros_like(resized_image)

        if results_pose.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                pose_image,
                results_pose.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
            )

            # Save pose image
            cv2.imwrite(output_pose_path, pose_image)

        return True
    else:
        print(f"No face detected in {image_path}")
        return False

# Process all image files in the input directory
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for filename in tqdm(image_files, desc="Processing images", unit="image"):
    input_path = os.path.join(input_dir, filename)
    output_pose_path = os.path.join(pose_dir, filename)
    output_landmarks_image_path = os.path.join(landmarks_dir, filename)
    copy_image_path = os.path.join(copy_dir, filename)

    # Process the image
    if process_image(input_path, output_pose_path, output_landmarks_image_path, copy_image_path):
        print(f"Processed and saved: {filename}")
    else:
        print(f"Skipped: {filename}")

print("Finished")
