import os
import mediapipe as mp
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2

# Suppress unnecessary logs
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def track_hands_with_sam(input_path, output_path, sam_model_key='vit_h'):
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Initialize SAM model
    try:
        sam_builder = sam_model_registry[sam_model_key]
        sam = sam_builder()  # Instantiate the SAM model
        mask_generator = SamAutomaticMaskGenerator(sam)
    except KeyError:
        print(f"Error: Invalid SAM model key '{sam_model_key}'. Available keys: {list(sam_model_registry.keys())}")
        return

    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Failed to open video from path: {input_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare the output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Calculate bounding box
                h, w, _ = frame.shape
                x_min = max(0, int(min([lm.x for lm in hand_landmarks.landmark]) * w))
                x_max = min(w, int(max([lm.x for lm in hand_landmarks.landmark]) * w))
                y_min = max(0, int(min([lm.y for lm in hand_landmarks.landmark]) * h))
                y_max = min(h, int(max([lm.y for lm in hand_landmarks.landmark]) * h))

                if x_max <= x_min or y_max <= y_min:
                    continue  # Skip invalid bounding boxes

                hand_roi = frame[y_min:y_max, x_min:x_max]

                # Generate mask
                masks = mask_generator.generate(hand_roi)
                if masks:
                    mask = masks[0]["segmentation"]
                    mask = (mask * 255).astype(np.uint8)
                    mask = cv2.resize(mask, (hand_roi.shape[1], hand_roi.shape[0]))
                    mask = np.stack([mask] * 3, axis=-1)

                    # Apply mask to the region
                    frame[y_min:y_max, x_min:x_max] = cv2.addWeighted(hand_roi, 0.7, mask, 0.3, 0)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Output video saved at: {output_path}")

if __name__ == "__main__":
    # Example usage
    input_video_path = "test.mp4"  # Path to the input video
    output_video_path = "output_video_with_masks.mp4"  # Path to save the output video
    sam_model_key = "vit_b"  # SAM model key (e.g., 'vit_b', 'vit_l', 'vit_h')

    print("Starting video processing...")
    track_hands_with_sam(input_video_path, output_video_path, sam_model_key)
    print("Video processing complete!")
    print(f"Output video saved at: {output_video_path}")

