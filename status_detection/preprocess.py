# Preprocessing module for squirrel state detection
# This module applies various image processing techniques to enhance the input frames for better mask creation and distance
import cv2
import numpy as np

# Parameters for preprocessing (tune if needed)
def preprocess_frame(frame, debug=False):
    """
    Attempts to achieve effects similar to a screenshot:
        - Automatic gamma correction
        - Contrast enhancement (CLAHE)
        - Light denoising
        - Optional resizing (interpolation)
    """
    img = frame.copy()

    # auto gamma correction based on mean brightness
    img_float = img.astype(np.float32) / 255.0
    # global mean brightness (Luma) 
    gray = cv2.cvtColor(img_float, cv2.COLOR_BGR2GRAY)
    mean_luma = np.mean(gray)
    mean_luma = max(mean_luma, 1e-4)  

    # Gamma so that mean brightness maps to ~0.5 (mid-gray)
    gamma = np.log(0.5) / np.log(mean_luma)
    gamma = np.clip(gamma, 0.5, 1.5)   # nicht zu extrem

    img_gamma = np.power(img_float, gamma)
    img_gamma = np.clip(img_gamma * 255.0, 0, 255).astype(np.uint8)

    # Contrast enhancement using CLAHE (adaptive histogram equalization)
    lab = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_clahe = clahe.apply(L)
    lab_clahe = cv2.merge((L_clahe, A, B))
    img_contrast = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Denoising (fastNlMeansDenoisingColored is good for cartoon-like images)
    img_denoised = cv2.fastNlMeansDenoisingColored(
        img_contrast, None,
        h=5, hColor=5,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # Optional resizing (e.g., 70% of original size) - can help with noise and speed, but may lose detail
    scale = 0.7  # 70 % der Originalgröße
    new_w = int(img_denoised.shape[1] * scale)
    new_h = int(img_denoised.shape[0] * scale)
    img_resized = cv2.resize(
        img_denoised,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA  
    )

    # Debug: show intermediate steps
    if debug:
        cv2.imshow("Original frame", frame)
        cv2.imshow("Gamma corrected", img_gamma)
        cv2.imshow("CLAHE contrast", img_contrast)
        cv2.imshow("Denoised + resized", img_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img_resized
