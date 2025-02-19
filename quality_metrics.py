import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_psnr(original, watermarked):
    """Calculate Peak Signal-to-Noise Ratio"""
    # Convert PIL images to numpy arrays
    original_array = np.array(original)
    watermarked_array = np.array(watermarked)

    # Ensure same dimensions
    if original_array.shape != watermarked_array.shape:
        watermarked_array = cv2.resize(
            watermarked_array, (original_array.shape[1], original_array.shape[0])
        )

    return peak_signal_noise_ratio(original_array, watermarked_array)


def calculate_ssim(original, watermarked):
    """Calculate Structural Similarity Index"""
    # Convert PIL images to numpy arrays
    original_array = np.array(original)
    watermarked_array = np.array(watermarked)

    # Ensure same dimensions
    if original_array.shape != watermarked_array.shape:
        watermarked_array = cv2.resize(
            watermarked_array, (original_array.shape[1], original_array.shape[0])
        )

    return structural_similarity(original_array, watermarked_array, channel_axis=2)


def analyze_resistance(watermarked_image):
    """Test watermark resistance against common attacks"""
    img_array = np.array(watermarked_image)
    results = {}

    # Test compression resistance
    _, compressed = cv2.imencode(".jpg", img_array, [cv2.IMWRITE_JPEG_QUALITY, 60])
    decompressed = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
    results["compression"] = peak_signal_noise_ratio(img_array, decompressed)

    # Test noise resistance
    noisy = img_array + np.random.normal(0, 10, img_array.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    results["noise"] = peak_signal_noise_ratio(img_array, noisy)

    # Test cropping resistance
    h, w = img_array.shape[:2]
    cropped = img_array[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    cropped_full = np.zeros_like(img_array)
    cropped_full[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = cropped
    results["cropping"] = peak_signal_noise_ratio(img_array, cropped_full)

    return results
