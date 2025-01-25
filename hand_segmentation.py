import os
import subprocess
import sys
import cv2
import numpy as np
import mediapipe as mp
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

class HandSegmentationPipeline:
    def __init__(self, video_file, output_dir="output"):
        """
        Initialize the pipeline for hand segmentation.

        Args:
            video_file (str): Path to the input video file.
            output_dir (str): Directory to save the outputs.
        """
        self.video_file = video_file

        # Automatically detect the `checkpoints` directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_dir = os.path.join(base_dir, "checkpoints")

        # Construct paths for weights and config
        self.weights_path = os.path.join(checkpoint_dir, "sam2.1_hiera_large.pt")
        self.model_config = os.path.join(checkpoint_dir, "sam2.1_hiera_l.yaml")

        # Output directories
        self.output_dir = os.path.abspath(output_dir)
        self.frames_dir = os.path.join(self.output_dir, "frames")
        self.segmented_dir = os.path.join(self.output_dir, "segmented_frames")
        self.output_video_path = os.path.join(self.output_dir, "segmented_video.mp4")

        # Device configuration
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.predictor = None
        self.mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

        # Create required directories
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.segmented_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_video_with_ffmpeg(self, frames_dir, output_video_path, framerate=30):
        """
        Generate a video from frames using FFmpeg.

        Args:
            frames_dir (str): Path to the folder containing segmented frames.
            output_video_path (str): Path to save the output video.
            framerate (int): Frame rate for the output video.
        """
        try:
            print("Generating video using FFmpeg...")
            command = [
                "ffmpeg",
                "-framerate", str(framerate),
                "-i", f"{frames_dir}/%05d.jpg",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                output_video_path
            ]
            subprocess.run(command, check=True)
            print(f"Video generated successfully: {output_video_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error generating video with FFmpeg: {e}")

    def extract_frames_ffmpeg(self):
        """
        Extract frames from the video using ffmpeg and save them in the frames directory.
        """
        print(f"Extracting frames from video: {self.video_file} using ffmpeg...")
        command = [
            "ffmpeg",
            "-i", self.video_file,
            "-q:v", "2",
            "-start_number", "0",
            f"{self.frames_dir}/%05d.jpg"
        ]
        try:
            subprocess.run(command, check=True)
            print("Frames extracted successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error while extracting frames with ffmpeg: {e}")
            raise

    def initialize_predictor(self):
        """
        Initialize the SAM2 predictor.
        """
        print("Initializing SAM2 predictor...")
        self.predictor = build_sam2_video_predictor(self.model_config, self.weights_path, device=self.device)

    def get_wrist_coordinates(self, image_path):
        """
        Extract wrist coordinates using MediaPipe from the input image.

        Args:
            image_path (str): Path to the image.

        Returns:
            List[Tuple[int, int]]: List of wrist coordinates (x, y).
        """
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(image_rgb)

        wrist_points = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                wrist_landmark = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
                wrist_x = int(wrist_landmark.x * image.shape[1])
                wrist_y = int(wrist_landmark.y * image.shape[0])
                wrist_points.append((wrist_x, wrist_y))

        return wrist_points

    def save_masked_image_on_original(self, mask, original_image_path, output_path, alpha=0.5):
        """
        Overlay the mask on the original image and save the result.

        Args:
            mask (numpy.ndarray): 2D segmentation mask.
            original_image_path (str): Path to the original image.
            output_path (str): Path to save the output image.
            alpha (float): Blending factor for overlay transparency (0.0 to 1.0).
        """
        original_image = Image.open(original_image_path).convert("RGBA")
        if mask.ndim > 2:
            mask = np.squeeze(mask, axis=0)
        mask_image = (mask * 255).astype(np.uint8)
        mask_image_pil = Image.fromarray(mask_image, mode="L").convert("RGBA")
        blended_image = Image.blend(original_image, mask_image_pil, alpha=alpha)
        blended_image.convert("RGB").save(output_path)
        print(f"Masked image saved at: {output_path}")

    def segment_video(self):
        """
        Perform hand segmentation on the extracted frames and save the results as a segmented video.
        """
        self.extract_frames_ffmpeg()
        self.initialize_predictor()

        frame_names = sorted(
            [f for f in os.listdir(self.frames_dir) if f.lower().endswith(".jpg")],
            key=lambda p: int(os.path.splitext(p)[0]),
        )
        inference_state = self.predictor.init_state(video_path=self.frames_dir)
        self.predictor.reset_state(inference_state)

        initial_frame_idx = 0
        object_id = 1

        initial_image_path = os.path.join(self.frames_dir, frame_names[initial_frame_idx])
        wrist_points = self.get_wrist_coordinates(initial_image_path)

        if not wrist_points:
            print("No wrist points detected in the initial frame. Aborting segmentation.")
            return

        points = np.array(wrist_points, dtype=np.float32)
        labels = np.array([1] * len(points), np.int32)
        self.predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=initial_frame_idx,
            obj_id=object_id,
            points=points,
            labels=labels,
        )

        segmented_frames = []
        for frame_idx, out_obj_ids, mask_logits in self.predictor.propagate_in_video(inference_state):
            original_image_path = os.path.join(self.frames_dir, frame_names[frame_idx])
            output_path = os.path.join(self.segmented_dir, frame_names[frame_idx])
            mask = (mask_logits[0] > 0.0).cpu().numpy()
            self.save_masked_image_on_original(mask, original_image_path, output_path)
            segmented_frame = cv2.imread(output_path)
            segmented_frames.append(segmented_frame)

        if segmented_frames:
            print("Saving segmented frames completed. Generating video...")
            self.generate_video_with_ffmpeg(self.segmented_dir, self.output_video_path)
            print(f"Segmented video saved at: {self.output_video_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_video_file>")
        sys.exit(1)

    # Get video file path from command-line arguments
    video_file_path = sys.argv[1]

    # Initialize and run the pipeline
    pipeline = HandSegmentationPipeline(video_file_path)
    pipeline.segment_video()
