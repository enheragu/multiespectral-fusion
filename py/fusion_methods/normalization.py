

import numpy as np

def normalize(image):
    fused_image = np.asarray(image, dtype=np.float32)
    min_value = float(np.min(fused_image))
    max_value = float(np.max(fused_image))
    dynamic_range = max_value - min_value

    # Avoid NaN/Inf propagation on near-constant images.
    if dynamic_range <= 1e-8:
        return np.zeros_like(fused_image, dtype=np.uint8)

    fused_image = (fused_image - min_value) / dynamic_range
    fused_image = np.clip(fused_image * 255.0, 0.0, 255.0).astype(np.uint8)
    return fused_image