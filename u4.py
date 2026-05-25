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