import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm


input_dir = 'input_images'
output_dir = 'output_poses'
os.makedirs(output_dir, exist_ok=True)

# MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def process_image(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"can not read：{image_path}")
        return

    # Using MediaPipe detect
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    pose_image = np.zeros_like(image)

    # if detected pose key point
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            pose_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
        )
    cv2.imwrite(output_path, pose_image)

image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]


for filename in tqdm(image_files, desc="Processing images", unit="image"):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)
    process_image(input_path, output_path)

print("Finished")
