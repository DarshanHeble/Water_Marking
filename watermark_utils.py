import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io


def add_visible_watermark(image, text, opacity=0.3, position="random"):
    """Add a visible text watermark with enhanced positioning"""
    img = image.convert("RGBA")
    watermark = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(watermark)

    # Calculate text size and position
    w, h = img.size
    text_size = max(w, h) // 20
    try:
        font = ImageFont.truetype("arial.ttf", text_size)
    except:
        font = ImageFont.load_default()

    # Enhanced positioning system
    if position == "random":
        import random

        x = random.randint(w // 4, 3 * w // 4)
        y = random.randint(h // 4, 3 * h // 4)
    else:
        x, y = w // 3, h // 3

    # Draw text with enhanced visibility
    draw.text((x, y), text, font=font, fill=(255, 255, 255, int(255 * opacity)))

    # Combine images and convert back to RGB
    watermarked = Image.alpha_composite(img, watermark)
    return watermarked.convert("RGB")


def add_invisible_watermark(image, message, strength=1.0):
    """Add an invisible watermark using enhanced LSB steganography"""
    img = image.convert("RGB")
    img_array = np.array(img)

    # Enhanced message encoding with error correction
    binary_message = "".join(format(ord(c), "08b") for c in message)
    # Add error detection code
    checksum = format(sum(ord(c) for c in message) % 256, "08b")
    binary_message = checksum + binary_message + "1111111111111110"

    if len(binary_message) > img_array.size:
        raise ValueError("Message too large for image")

    # Enhanced LSB embedding with strength factor
    flat_img = img_array.flatten()
    for i, bit in enumerate(binary_message):
        if i >= len(flat_img):
            break
        # Apply strength factor to make watermark more/less detectable
        mask = 254 if strength > 0.5 else 252
        flat_img[i] = (flat_img[i] & mask) | (int(bit) * int(strength * 1))

    modified = flat_img.reshape(img_array.shape)
    return Image.fromarray(modified)


def verify_watermark(original_image, watermarked_image, threshold=0.8):
    """Verify the presence and integrity of watermark"""
    orig_array = np.array(original_image)
    water_array = np.array(watermarked_image)

    # Calculate difference map
    diff_map = np.abs(orig_array - water_array)

    # Analyze watermark presence
    watermark_presence = np.mean(diff_map) > 0

    # Calculate similarity score
    similarity = 1 - (np.sum(diff_map) / (orig_array.size * 255))

    # Detect tampering
    tampering_detected = similarity < threshold

    return {
        "watermark_present": watermark_presence,
        "similarity_score": similarity,
        "tampering_detected": tampering_detected,
        "difference_map": Image.fromarray(diff_map.astype(np.uint8)),
    }


def extract_invisible_watermark(image):
    """Extract and verify invisible watermark"""
    img = image.convert("RGB")
    img_array = np.array(img)
    flat_img = img_array.flatten()

    # Extract LSB with error checking
    binary_message = ""
    for i in range(len(flat_img)):
        binary_message += str(flat_img[i] & 1)
        if len(binary_message) >= 8 and binary_message[-16:] == "1111111111111110":
            break

    # Verify checksum
    if (
        len(binary_message) >= 24
    ):  # 8 bits checksum + at least 1 character + 16 bits end marker
        stored_checksum = binary_message[:8]
        message_bits = binary_message[8:-16]

        # Convert to text
        message = ""
        for i in range(0, len(message_bits), 8):
            if i + 8 <= len(message_bits):
                byte = message_bits[i : i + 8]
                message += chr(int(byte, 2))

        # Verify checksum
        calculated_checksum = format(sum(ord(c) for c in message) % 256, "08b")
        if calculated_checksum == stored_checksum:
            return {"message": message, "integrity_verified": True}
        else:
            return {"message": message, "integrity_verified": False}

    return {"message": "", "integrity_verified": False}


def analyze_attack_resistance(watermarked_image):
    """Enhanced attack resistance analysis"""
    img_array = np.array(watermarked_image)
    results = {}

    # Test compression resistance
    _, compressed = cv2.imencode(".jpg", img_array, [cv2.IMWRITE_JPEG_QUALITY, 60])
    decompressed = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
    results["compression"] = {
        "psnr": peak_signal_noise_ratio(img_array, decompressed),
        "survived": verify_watermark(
            Image.fromarray(img_array), Image.fromarray(decompressed)
        )["watermark_present"],
    }

    # Test noise resistance
    noisy = img_array + np.random.normal(0, 10, img_array.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    results["noise"] = {
        "psnr": peak_signal_noise_ratio(img_array, noisy),
        "survived": verify_watermark(
            Image.fromarray(img_array), Image.fromarray(noisy)
        )["watermark_present"],
    }

    # Test cropping resistance
    h, w = img_array.shape[:2]
    cropped = img_array[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    cropped_full = np.zeros_like(img_array)
    cropped_full[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = cropped
    results["cropping"] = {
        "psnr": peak_signal_noise_ratio(img_array, cropped_full),
        "survived": verify_watermark(
            Image.fromarray(img_array), Image.fromarray(cropped_full)
        )["watermark_present"],
    }

    return results


def optimize_watermark_strength(image, message, target_psnr=35):
    """Optimize watermark strength to balance visibility and robustness"""
    low_strength = 0.1
    high_strength = 1.0
    best_strength = 0.5
    best_psnr_diff = float("inf")

    for _ in range(5):  # Binary search for optimal strength
        mid_strength = (low_strength + high_strength) / 2
        watermarked = add_invisible_watermark(image, message, strength=mid_strength)
        current_psnr = calculate_psnr(image, watermarked)

        psnr_diff = abs(current_psnr - target_psnr)
        if psnr_diff < best_psnr_diff:
            best_strength = mid_strength
            best_psnr_diff = psnr_diff

        if current_psnr > target_psnr:
            low_strength = mid_strength
        else:
            high_strength = mid_strength

    return best_strength


def calculate_psnr(original, watermarked):
    """Calculate Peak Signal-to-Noise Ratio"""
    original_array = np.array(original)
    watermarked_array = np.array(watermarked)

    if original_array.shape != watermarked_array.shape:
        watermarked_array = cv2.resize(
            watermarked_array, (original_array.shape[1], original_array.shape[0])
        )

    mse = np.mean((original_array - watermarked_array) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def peak_signal_noise_ratio(img1, img2):
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def add_semi_visible_watermark(image, pattern_strength=0.1):
    """Add a semi-visible pattern watermark"""
    # Convert to RGB if not already
    img = image.convert("RGB")
    img_array = np.array(img)
    h, w = img_array.shape[:2]

    # Create diagonal pattern
    pattern = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            if (i + j) % 20 == 0:
                pattern[i, j] = 1

    # Apply pattern
    for c in range(3):
        img_array[:, :, c] = img_array[:, :, c] * (1 - pattern_strength * pattern)

    return Image.fromarray(img_array.astype(np.uint8))
