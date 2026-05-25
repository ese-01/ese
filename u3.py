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