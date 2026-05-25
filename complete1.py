
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog, local_binary_pattern
from skimage import exposure

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
for idx, path in enumerate(image_paths[:3]):
    print(f"\nProcessing Image {idx+1}: {path}")

    image = cv2.imread(path)

    if image is None:
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(image, (256, 256))
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    center = (128, 128)
    matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated = cv2.warpAffine(resized, matrix, (256, 256))

    flipped = cv2.flip(resized, 1)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    edges = cv2.Canny(gray, 100, 200)

    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    orb_image = cv2.drawKeypoints(
        resized,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    hog_features, hog_image = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True
    )

    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')

    plt.figure(figsize=(18, 14))

    plt.subplot(3, 4, 1)
    plt.imshow(resized)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(3, 4, 2)
    plt.imshow(gray, cmap='gray')
    plt.title("Grayscale")
    plt.axis("off")

    plt.subplot(3, 4, 3)
    plt.imshow(rotated)
    plt.title("Rotated")
    plt.axis("off")

    plt.subplot(3, 4, 4)
    plt.imshow(flipped)
    plt.title("Flipped")
    plt.axis("off")

    plt.subplot(3, 4, 5)
    plt.imshow(blurred, cmap='gray')
    plt.title("Blurred")
    plt.axis("off")

    plt.subplot(3, 4, 6)
    plt.imshow(threshold, cmap='gray')
    plt.title("Threshold")
    plt.axis("off")

    plt.subplot(3, 4, 7)
    plt.imshow(edges, cmap='gray')
    plt.title("Canny Edges")
    plt.axis("off")

    plt.subplot(3, 4, 8)
    plt.imshow(orb_image)
    plt.title("ORB Keypoints")
    plt.axis("off")

    plt.subplot(3, 4, 9)
    plt.imshow(hog_image, cmap='gray')
    plt.title("HOG Features")
    plt.axis("off")

    plt.subplot(3, 4, 10)
    plt.imshow(lbp, cmap='gray')
    plt.title("LBP Texture")
    plt.axis("off")

    plt.subplot(3, 4, 11)
    plt.hist(gray.ravel(), bins=256)
    plt.title("Color Histogram")

    plt.tight_layout()
    plt.show()
def extract_features(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Corrupted image")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False
    )

    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
    lbp_hist = lbp_hist.astype(float)
    lbp_hist /= (lbp_hist.sum() + 1e-7)

    color_features = []

    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        color_features.extend(hist)

    return np.hstack([hog_features, lbp_hist, color_features])
X = []
y = []
corrupted = []

for path in image_paths:
    try:
        features = extract_features(path)

        label = os.path.basename(os.path.dirname(path))

        X.append(features)
        y.append(label)

    except:
        corrupted.append(path)

X = np.array(X)
y = np.array(y)

print("Valid Images:", len(X))
print("Corrupted Skipped:", len(corrupted))
print("Detected Classes:", np.unique(y))
encoder = LabelEncoder()
y = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
models = {
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    results[name] = acc

    print(name, "Accuracy:", acc)
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

predictions = best_model.predict(X_test)

print("Best Model:", best_model_name)
print(classification_report(y_test, predictions))
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6,5))
plt.imshow(cm, cmap='Blues')

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()
plt.figure(figsize=(8,5))
plt.plot(list(results.keys()), list(results.values()), marker='o')
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.show()
 