import zipfile, os, cv2, numpy as np, matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
from skimage import exposure
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

zip_path   = "images.zip"  
extract_to = "dataset1"

with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(extract_to)

dataset_path = extract_to
image_paths  = [os.path.join(r, f) for r, _, files in os.walk(dataset_path)
                for f in files if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))]
print("Total images:", len(image_paths))

for idx, path in enumerate(image_paths[:3]):
    img  = cv2.imread(path)
    if img is None: continue
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res  = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)

    ops = {
        "Original":      res,
        "Grayscale":     gray,
        "Rotated":       cv2.warpAffine(res, cv2.getRotationMatrix2D((128,128), 45, 1.0), (256,256)),
        "Flipped":       cv2.flip(res, 1),
        "Blurred":       cv2.GaussianBlur(gray, (5,5), 0),
        "Threshold":     cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1],
        "Canny Edges":   cv2.Canny(gray, 100, 200),
        "ORB Keypoints": cv2.drawKeypoints(res, cv2.ORB_create().detectAndCompute(gray, None)[0],
                                           None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS),
        "HOG":           exposure.rescale_intensity(hog(gray,9,(8,8),(2,2),visualize=True)[1], in_range=(0,10)),
        "LBP":           local_binary_pattern(gray, P=8, R=1, method='uniform'),
    }

    plt.figure(figsize=(18, 10))
    for i, (title, data) in enumerate(ops.items(), 1):
        plt.subplot(3, 4, i)
        plt.imshow(data, cmap='gray' if data.ndim == 2 else None)
        plt.title(title); plt.axis("off")
    plt.subplot(3, 4, 11); plt.hist(gray.ravel(), bins=256); plt.title("Histogram")
    plt.tight_layout(); plt.show()

def extract_features(path):
    img  = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img  = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    hog_f = hog(gray, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2))
    lbp   = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_h, _ = np.histogram(lbp.ravel(), bins=10, range=(0,10))
    lbp_h = lbp_h.astype(float) / (lbp_h.sum() + 1e-7)
    color = np.hstack([cv2.normalize(cv2.calcHist([img],[i],None,[32],[0,256]), None).flatten()
                       for i in range(3)])
    return np.hstack([hog_f, lbp_h, color])

X, y, bad = [], [], []
for path in image_paths:
    try:
        name  = os.path.basename(path).lower()
        label = 'without_mask' if 'without' in name else 'with_mask'
        X.append(extract_features(path))
        y.append(label)
    except:
        bad.append(path)

X = np.array(X)
y = LabelEncoder().fit_transform(y)
print(f"Valid: {len(X)}  |  Corrupted: {len(bad)}")
print(f"Classes: {np.unique(y)}")  # should print [0 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

models = {"SVM": SVC(), "Random Forest": RandomForestClassifier(),
          "KNN": KNeighborsClassifier(), "Logistic Regression": LogisticRegression(max_iter=1000)}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    results[name] = accuracy_score(y_test, model.predict(X_test))
    print(f"{name}: {results[name]:.4f}")

best_name  = max(results, key=results.get)
best_model = models[best_name]
preds      = best_model.predict(X_test)
print(f"\nBest Model: {best_name}")
print(classification_report(y_test, preds, target_names=['with_mask', 'without_mask']))

cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(6,5))
plt.imshow(cm, cmap='Blues')
[[plt.text(j,i,cm[i,j],ha='center',va='center') for j in range(len(cm))] for i in range(len(cm))]
plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.xticks([0,1], ['with_mask','without_mask']); plt.yticks([0,1], ['with_mask','without_mask'])
plt.colorbar(); plt.show()

plt.figure(figsize=(8,5))
plt.plot(list(results.keys()), list(results.values()), marker='o')
plt.title("Model Accuracy Comparison"); plt.xlabel("Models"); plt.ylabel("Accuracy")
plt.ylim(0, 1); plt.grid(True); plt.show()