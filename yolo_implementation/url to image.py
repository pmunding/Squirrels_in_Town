import pandas as pd
import requests
import os
from tqdm import tqdm

# Load CSV
csv_path = "C:/Users/SIVA SHANMUKHA/OneDrive/Desktop/Treps_02_SIT/observations-659473.csv"   # update path if needed
df = pd.read_csv(csv_path)

# Create output folder
output_dir = "C:/Users/SIVA SHANMUKHA/OneDrive/Desktop/Treps_02_SIT/YOLO/squirrel_images"
os.makedirs(output_dir, exist_ok=True)

# Loop through rows
for idx, row in tqdm(df.iterrows(), total=len(df)):
    url = row["image_url"]

    if pd.isna(url):
        continue

    # Clean file name
    img_name = f"img_{idx}.jpg"
    img_path = os.path.join(output_dir, img_name)

    # Skip if already downloaded
    if os.path.exists(img_path):
        continue

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(img_path, "wb") as f:
                f.write(response.content)
    except Exception as e:
        print(f"Failed to download {url}")
