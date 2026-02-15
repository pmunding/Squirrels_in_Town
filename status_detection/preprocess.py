import cv2
import numpy as np

def preprocess_frame(frame, debug=False):
    """
    Versucht, ähnliche Effekte wie ein Screenshot zu erzielen:
    - automatische Gamma-Korrektur
    - Kontrastverbesserung (CLAHE)
    - leichtes Denoising
    - optionales Resizing (Interpolation)
    """
    img = frame.copy()

    # ---------- (1) automatische Gamma-Korrektur ----------
    # Normiere auf [0, 1]
    img_float = img.astype(np.float32) / 255.0
    # grobe Helligkeit schätzen
    gray = cv2.cvtColor(img_float, cv2.COLOR_BGR2GRAY)
    mean_luma = np.mean(gray)
    mean_luma = max(mean_luma, 1e-4)  # Division by zero vermeiden

    # Ziel: mittlere Helligkeit Richtung 0.5 ziehen
    gamma = np.log(0.5) / np.log(mean_luma)
    gamma = np.clip(gamma, 0.5, 1.5)   # nicht zu extrem

    img_gamma = np.power(img_float, gamma)
    img_gamma = np.clip(img_gamma * 255.0, 0, 255).astype(np.uint8)

    # ---------- (2) Kontrastverbesserung via CLAHE ----------
    lab = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_clahe = clahe.apply(L)
    lab_clahe = cv2.merge((L_clahe, A, B))
    img_contrast = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # ---------- (3) leichtes Denoising ----------
    # ähnlich wie "OS Denoising"
    img_denoised = cv2.fastNlMeansDenoisingColored(
        img_contrast, None,
        h=5, hColor=5,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # ---------- (4) optionales Resizing (z.B. halbe Auflösung) ----------
    # Wenn du willst, kannst du das abschalten oder anpassen
    scale = 0.7  # 70 % der Originalgröße
    new_w = int(img_denoised.shape[1] * scale)
    new_h = int(img_denoised.shape[0] * scale)
    img_resized = cv2.resize(
        img_denoised,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA  # schöne Interpolation, wenig Alias
    )

    if debug:
        cv2.imshow("Original frame", frame)
        cv2.imshow("Gamma corrected", img_gamma)
        cv2.imshow("CLAHE contrast", img_contrast)
        cv2.imshow("Denoised + resized", img_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img_resized