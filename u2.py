import zipfile, os
zip_path   = "images.zip"  
extract_to = "dataset1"

with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(extract_to)

dataset_path = "dataset1"
image_paths = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_paths.append(os.path.join(root, file))

print("Total images found:", len(image_paths))
valid = 0
corrupted = []

for path in image_paths:
    img = cv2.imread(path)

    if img is None:
        corrupted.append(path)
    else:
        valid += 1
print("Valid Images:", valid)
print("Corrupted Images:", len(corrupted))