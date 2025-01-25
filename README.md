# Hand Segmentation Pipeline

This project performs hand segmentation on a video using the SAM2 model, MediaPipe, and OpenCV. It processes a video file to segment hands, saves the segmented frames, and generates a final segmented video.

---

## Features
- **Hand Detection**: Uses MediaPipe to identify wrist coordinates.
- **Segmentation**: Utilizes Facebook's SAM2 model for accurate hand segmentation.
- **Video Processing**: Extracts frames, applies segmentation, and combines frames into a final video.

---
## Demo

### 1. Original Video (`test.mp4`)
The input video file used for processing. Watch it below:



https://github.com/user-attachments/assets/fb6cc044-e251-4cab-857d-4d30768987ce



### 2. Segmented Video (`segmented_video.mp4`)
The final output video showing the segmented hands. Watch it below:


https://github.com/user-attachments/assets/a1f6e9b1-638c-4166-9598-161b26e5349a



### 3. Wrist Detection Image (`output_image.jpg`)
An image highlighting the detected wrist before segmentation:

![output_image](https://github.com/user-attachments/assets/ee0c0398-2059-45fe-bfb5-e11810814857)

---

## Requirements
1. Python 3.8 or later
2. A compatible NVIDIA GPU (for CUDA) or CPU
3. FFmpeg installed and available in your system PATH

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo-url.git
cd your-repo-folder
```

### 2. Install Dependencies
Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # For Linux/macOS
env\Scripts\activate     # For Windows
```

Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Ensure FFmpeg is Installed
FFmpeg is used for frame extraction and video creation. Install it as follows:

#### On Linux:
```bash
sudo apt update
sudo apt install ffmpeg
```

#### On macOS (with Homebrew):
```bash
brew install ffmpeg
```

#### On Windows:
1. Download FFmpeg from [ffmpeg.org](https://drive.google.com/drive/folders/15fTVaGijOnzADNftl5Fo6G-n8P6ydUcs?usp=sharing).
2. Add FFmpeg to your system PATH.

---

### 4. Load Model weights :
1. Since github wont let me upload files > 50 MB I have created a google drive for weights please download it and save it in the checkpoints folder :)
2. Link : https://drive.google.com/drive/folders/15fTVaGijOnzADNftl5Fo6G-n8P6ydUcs?usp=sharing
---
## Running the Script

### Input:
- A video file (e.g., `test.mp4`) to process.

### Output:
- Processed frames in the `output/frames` folder.
- Segmented frames in the `output/segmented_frames` folder.
- A final segmented video in `output/segmented_video.mp4`.
- An output image showing wrist detection: `output_image.jpg`.

### Steps to Run:
1. Place your video file in the project directory or note its path.
2. Run the script with the video file path as an argument:
   ```bash
   python script.py "path/to/video.mp4"
   ```

### Example:
```bash
python script.py "/path/to/test.mp4"
```
---

## Project Structure
```plaintext
├── checkpoints/
│   ├── sam2.1_hiera_large.pt      # SAM2 model weights
│   ├── sam2.1_hiera_l.yaml        # SAM2 model configuration
├── output/
│   ├── frames/                    # Extracted video frames
│   ├── segmented_frames/          # Segmented frames
│   ├── segmented_video.mp4        # Final segmented video
│   ├── output_image.jpg           # Wrist detection image
├── script.py                      # Main script for processing
├── requirements.txt               # Required dependencies
├── README.md                      # Project documentation
```

---

## Notes
- Ensure the `checkpoints` directory contains the SAM2 weights and configuration files:
  - `sam2.1_hiera_large.pt`
  - `sam2.1_hiera_l.yaml`
- The final video is saved in the `output` directory as `segmented_video.mp4`.
- The wrist detection image is saved as `output_image.jpg`.

---

## Troubleshooting
1. **Hydra Config Not Found**:
   Ensure the `checkpoints` folder is in the same directory as `script.py` and contains the correct model files.

2. **FFmpeg Not Found**:
   Verify FFmpeg is installed and available in your system PATH.

3. **CUDA Errors**:
   Ensure you have a compatible GPU with the correct version of CUDA installed.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- [Facebook SAM2](https://github.com/facebookresearch/sam2)
- [MediaPipe](https://google.github.io/mediapipe/)
- [OpenCV](https://opencv.org/)

