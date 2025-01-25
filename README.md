# Hand Tracking and Mask Generation Pipeline

This repository provides a pipeline to detect and track hands in a video using MediaPipe and generate segmentation masks for each frame using SAM 2.

## Setup Instructions

### Prerequisites
Ensure you have the following installed:
- Python 3.8 or later
- `pip` (Python package installer)

### Dependencies
The project relies on the following Python libraries:
- `opencv-python`
- `mediapipe`
- `numpy`
- `segment-anything`(sam2)

### Environment Set Up
conda create -n sam2 python=3.12
conda activate sam2
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Follow the instructions for setting up SAM 2 
Install mediapipe with pip install -q mediapipe
And cv2 with pip install opencv-python

Install the dependencies using:
```bash
pip install opencv-python mediapipe numpy segment-anything
```

### Download SAM Model
You need to download the SAM model weights. Follow these steps:
1. Visit the [Segment Anything repository](https://github.com/facebookresearch/segment-anything) to locate the model weights.
2. Download the desired model file (e.g., `sam_vit_h.pth`).
3. Place the downloaded model in a directory of your choice and note the path.

## How to Run the Pipeline

### Step 1: Input Video Preparation
Ensure you have an input video file in a supported format (e.g., `.mp4`). Place the video file in the project directory or specify its path.

### Step 2: Running the Script
1. Save the provided Python script as `hand_tracking_pipeline.py`.
2. Run the script with the following command:
   ```bash
   python hand_tracking_pipeline.py --input_path <input_video_path> --output_path <output_video_path> --sam_model_path <path_to_sam_model>
   ```

#### Arguments
- `--input_path`: Path to the input video file.
- `--output_path`: Path to save the output video with segmentation masks.
- `--sam_model_path`: Path to the SAM model weights file.

#### Example Command
```bash
python hand_tracking_pipeline.py --input_path input_video.mp4 --output_path output_video_with_masks.mp4 --sam_model_path sam_vit_h.pth
```

### Step 3: Output
After the script finishes, the output video will be saved at the specified `--output_path`. This video will contain the original frames with segmentation masks overlaid on detected hands.


## Additional Notes
- This project uses MediaPipe for hand detection and SAM 2 for segmentation mask generation.
- Tested on Python 3.9 with SAM models from the official [Segment Anything repository](https://github.com/facebookresearch/segment-anything).
