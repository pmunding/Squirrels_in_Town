import cv2
import numpy as np

def detect_entry(image):  
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Could not load image from path: {image}")
    else:
        img = image.copy()

    output = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9), 0)

    shape = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35, 5
    )

    # Hough Circle Detection
    circles = cv2.HoughCircles(
        shape,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,      
        param1=100,         
        param2=40,       
        minRadius=50,       
        maxRadius=180
    )

    if circles is None:
        print("No circle found.")
        return None

    circles = np.uint16(np.around(circles))
    x, y, r = circles[0][0]

    # Draw and get region sizes
    output, circle_sizes = get_entry_area(output, x, y, r)

    cv2.imshow("Detected Entry", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return {
        "center": (x, y),
        "entrance": circle_sizes["entrance"],
        "half": circle_sizes["half"],
        "full": circle_sizes["full"]
    }

# function to create circles for the masks
def get_entry_area(image, x, y, r):
    
    entrance_r = r
    half_r = r + 180
    full_r = r + 390

    cv2.circle(image, (x, y), entrance_r, (0,255,0), 3)
    cv2.circle(image, (x, y), half_r, (255,0,0), 3)
    cv2.circle(image, (x, y), full_r, (0,0,255), 3)

    return image, {
        "entrance": entrance_r, 
        "half": half_r, 
        "full": full_r
    }