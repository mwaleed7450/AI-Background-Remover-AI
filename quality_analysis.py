"""
Image Quality Analysis
Detects blur, noise, resolution, lighting, and compression quality,
then returns a combined overall quality score.
"""

import cv2
import numpy as np


def analyze_resolution(image: np.ndarray) -> dict:
    h, w = image.shape[:2]
    total_px = h * w
    if total_px >= 1920 * 1080:
        rating = "Good"
        score = 100
    elif total_px >= 1280 * 720:
        rating = "Medium"
        score = 70
    else:
        rating = "Poor"
        score = 40
    return {"width": w, "height": h, "rating": rating, "score": score}


def analyze_blur(gray: np.ndarray) -> dict:
    # Variance of Laplacian - low variance = blurry
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance >= 300:
        rating = "Low"      # low blur = sharp image
        score = 100
    elif variance >= 100:
        rating = "Medium"
        score = 65
    else:
        rating = "High"
        score = 30
    return {"variance": round(float(variance), 2), "rating": rating, "score": score}


def analyze_noise(gray: np.ndarray) -> dict:
    # Estimate noise via difference between image and a denoised version
    denoised = cv2.medianBlur(gray, 5)
    diff = cv2.absdiff(gray, denoised)
    noise_level = float(np.mean(diff))
    if noise_level <= 2.0:
        rating = "Low"
        score = 100
    elif noise_level <= 6.0:
        rating = "Medium"
        score = 65
    else:
        rating = "High"
        score = 30
    return {"noise_level": round(noise_level, 2), "rating": rating, "score": score}


def analyze_lighting(image: np.ndarray) -> dict:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = float(np.mean(hsv[:, :, 2]))
    if 90 <= brightness <= 200:
        rating = "Good"
        score = 100
    elif 60 <= brightness < 90 or 200 < brightness <= 230:
        rating = "Medium"
        score = 65
    else:
        rating = "Poor"
        score = 35
    return {"avg_brightness": round(brightness, 2), "rating": rating, "score": score}


def analyze_compression(gray: np.ndarray, block_size: int = 8) -> dict:
    # Rough blockiness estimate as a proxy for JPEG compression artifacts
    h, w = gray.shape
    h_crop = h - (h % block_size)
    w_crop = w - (w % block_size)
    cropped = gray[:h_crop, :w_crop].astype(np.float32)

    horiz_diff = np.abs(np.diff(cropped, axis=1))
    block_edges = horiz_diff[:, block_size - 1::block_size]
    non_block_edges = np.delete(horiz_diff, np.arange(block_size - 1, horiz_diff.shape[1], block_size), axis=1)

    blockiness = float(np.mean(block_edges)) - float(np.mean(non_block_edges))
    if blockiness <= 1.0:
        rating = "Good"
        score = 100
    elif blockiness <= 3.0:
        rating = "Medium"
        score = 65
    else:
        rating = "Poor"
        score = 35
    return {"blockiness": round(blockiness, 2), "rating": rating, "score": score}


def analyze_image_quality(image_path: str) -> dict:
    """
    Main entry point. Takes a path to an image and returns a full quality report.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    resolution = analyze_resolution(image)
    blur = analyze_blur(gray)
    noise = analyze_noise(gray)
    lighting = analyze_lighting(image)
    compression = analyze_compression(gray)

    overall = round(
        (resolution["score"] * 0.25 +
         blur["score"] * 0.25 +
         noise["score"] * 0.2 +
         lighting["score"] * 0.2 +
         compression["score"] * 0.1),
        1
    )

    return {
        "resolution": resolution,
        "blur": blur,
        "noise": noise,
        "lighting": lighting,
        "compression": compression,
        "overall_quality_percent": overall,
    }