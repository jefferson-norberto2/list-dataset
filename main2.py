from dataset_curator import check_image_similarity

# 1. High-level evaluation with threshold decision
img1_path = "Datasets/LISTV2/images/test/src/5265_pristine_1800.png"
img2_path = "Datasets/LISTV2/images/test/src/5260_pristine_1800.png"
result = check_image_similarity(img1_path, img2_path)
if result.is_similar:
    print(f"Images are similar! SSIM: {result.score:.4f}")
else:
    print(f"Images aren't similar. SSIM: {result.score:.4f} < {result.threshold}")

