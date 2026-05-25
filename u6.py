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
