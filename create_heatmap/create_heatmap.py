import cv2
import numpy as np
import os

input_video = "path/to/video.mp4"
output_folder = "path/to/output" 

output_video = os.path.join(output_folder, "squirrel_heatmap.mp4")
heatmap_image = os.path.join(output_folder, "squirrel_heatmap.png")

# Open video
cap = cv2.VideoCapture(input_video)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Heatmap accumulator
heatmap_acc = np.zeros((height, width), dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale to detect dark squirrel on background
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1]  # adjust 200 if needed

    # Update heatmap
    heatmap_acc += mask.astype(np.float32)

    # Normalize heatmap for current frame
    heatmap_norm = cv2.normalize(heatmap_acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    # Overlay heatmap on original frame
    overlay = cv2.addWeighted(frame, 0.5, heatmap_color, 0.5, 0)

    # Write frame to video
    out.write(overlay)

cap.release()
out.release()

# Save final heatmap as PNG
final_heatmap = cv2.normalize(heatmap_acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
final_heatmap_color = cv2.applyColorMap(final_heatmap, cv2.COLORMAP_JET)

ext = os.path.splitext(heatmap_image)[1]
_, encoded_img = cv2.imencode(ext, final_heatmap_color)
with open(heatmap_image, mode="wb") as f:
    encoded_img.tofile(f)