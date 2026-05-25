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