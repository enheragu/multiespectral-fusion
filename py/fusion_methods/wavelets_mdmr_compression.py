
#!/usr/bin/env python3
# encoding: utf-8
"""
    Creates different approachs of mixing RGB with Thermal images:
    Wavelet Transform:
    MDMR:
"""

import os
import time

import cv2 as cv
import numpy as np
import pywt

from sklearn.decomposition import PCA


# NOTE (web port): the original did `from curvelets.numpy import SimpleUDCT` at
# module level. In Pyodide that package is installed lazily (and may be absent),
# so a module-level import would also break the wavelet methods. This shim defers
# the import to the first curvelet call; call sites use `SimpleUDCT(...)` as-is.
def SimpleUDCT(*args, **kwargs):
    from curvelets.numpy import SimpleUDCT as _SimpleUDCT
    return _SimpleUDCT(*args, **kwargs)

# Small hack so packages can be found
if __name__ == "__main__":
    import sys
    sys.path.append('./src')


from utils import log, bcolors
from Dataset.decorators import time_execution_measure, save_image_if_path, save_npmat_if_path
from Dataset.fusion_methods.normalization import normalize

def processWaveletCoeffsMax(visible_image, thermal_image, coeffs):
    # Fuse coefficients
    # Detail sub-bads capture mostly: bordes, texturas y cambios locales
    cAprox = [c[0] for c in coeffs]
    cDetail = [c[1] for c in coeffs]

    cDetail_fused = []
    for i in range(3):  # Fuse for 3 output channels
        # Averaged with thermal channel coeffs? Max?
        # Average tends to lose texture and borders, abs max better preserve these features
        cH = np.where(np.abs(cDetail[i][0]) > np.abs(cDetail[3][0]), cDetail[i][0], cDetail[3][0])
        cV = np.where(np.abs(cDetail[i][1]) > np.abs(cDetail[3][1]), cDetail[i][1], cDetail[3][1])
        cD = np.where(np.abs(cDetail[i][2]) > np.abs(cDetail[3][2]), cDetail[i][2], cDetail[3][2])
        cDetail_fused.append((cH, cV, cD))

    ## Aproximation sub-bands contain most of the information. Thats why a better fusion is applied
    # Las bandas de aproximación contienen la mayor parte de la información estructural y espectral 
    # relevante de la imagen.
    
    # For aproximation coef if pixel in thermal image is high, more weight to thermal coeff
    weights = thermal_image / (thermal_image.max() + 1e-8)
    weights_resized = cv.resize(weights, (cAprox[3].shape[1], cAprox[3].shape[0]), interpolation=cv.INTER_LINEAR)
    cA_fused = []
    for i in range(3):
        cA_fused.append(weights_resized * cAprox[3] + (1 - weights_resized) * cAprox[i])

    fused_image = []
    for i in range(3):
        new_coeffs = (cA_fused[i], cDetail_fused[i])
        fused_channel = pywt.idwt2(new_coeffs, 'haar')
        fused_channel = fused_channel[:visible_image.shape[0], :visible_image.shape[1]] # Ensure original shape
        fused_image.append(fused_channel)
        
    fused_image = np.stack(fused_image, axis=-1) 
    
    # Ensure range 0-255 and uint8 encoding
    fused_image = normalize(fused_image)
    return fused_image

@save_npmat_if_path
def combine_rgbt_wavelet_max(visible_image, thermal_image):
    rgbt = np.dstack((visible_image, thermal_image))

    # Wavelet decoposition in approximation sub-bands (low freq.) and detail sub-bands (high freq.).
    # Apply Discret Wavelet Transform to each channel
    coeffs = [pywt.dwt2(rgbt[:,:,i], 'haar') for i in range(4)]

    fused_image = processWaveletCoeffsMax(visible_image, thermal_image, coeffs)
    return fused_image


@save_npmat_if_path
def combine_rgbt_wavelet(visible_image, thermal_image):
    rgbt = np.dstack((visible_image, thermal_image))

    # Wavelet decoposition in approximation sub-bands (low freq.) and detail sub-bands (high freq.).
    # Apply Discret Wavelet Transform to each channel
    coeffs = [pywt.dwt2(rgbt[:,:,i], 'haar') for i in range(4)]
    cAprox = [c[0] for c in coeffs]
    cDetail = [c[1] for c in coeffs]

    # Fuse coefficients
    # Detail sub-bads capture mostly: bordes, texturas y cambios locales
    cDetail_fused = []
    cA_fused = []
    for i in range(3):  # Fuse for 3 output channels
        # Averaged with thermal channel coeffs? Max?
        # Average tends to lose texture and borders, abs max better preserve these features
        cH = (cDetail[i][0] + cDetail[3][0])/2
        cV = (cDetail[i][1] + cDetail[3][1])/2
        cD = (cDetail[i][2] + cDetail[3][2])/2
        cDetail_fused.append((cH, cV, cD))
        cA_fused.append((cAprox[3]+cAprox[i])/2)

    ## Aproximation sub-bands contain most of the information. Thats why a better fusion is applied
    # Las bandas de aproximación contienen la mayor parte de la información estructural y espectral 
    # relevante de la imagen.
    
    fused_image = []
    for i in range(3):
        new_coeffs = (cA_fused[i], cDetail_fused[i])
        fused_channel = pywt.idwt2(new_coeffs, 'haar')
        fused_channel = fused_channel[:visible_image.shape[0], :visible_image.shape[1]] # Ensure original shape
        fused_image.append(fused_channel)
        
    fused_image = np.stack(fused_image, axis=-1) 
    
    # Ensure range 0-255 and uint8 encoding
    fused_image = normalize(fused_image)

    fused_image_max = processWaveletCoeffsMax(visible_image, thermal_image, coeffs)
    return fused_image, fused_image_max, 'wavelet', 'wavelet_max'

def _fuse_curvelet_scales(coeff_v, coeff_th, mode):
    """Fuse one channel's curvelet coeffs against the thermal coeffs.
    mode 'avg': average every sub-band. mode 'max': average the approximation
    (j==0) and abs-max the detail sub-bands. Bit-identical to the originals."""
    c_fused = []
    for j in range(len(coeff_v)):  # levels (nscales)
        scale_fused = []
        for k in range(len(coeff_v[j])):  # coeffs in each level
            a = np.array(coeff_v[j][k])
            b = np.array(coeff_th[j][k])
            if mode == 'avg' or j == 0:
                fused_coeff = (a + b) / 2
            else:
                mask = np.abs(a) >= np.abs(b)
                fused_coeff = np.where(mask, a, b)
            scale_fused.append(fused_coeff)
        c_fused.append(scale_fused)
    return c_fused


# WEB-PORT MEMORY NOTE: the originals built `coeffs=[forward(ch) for ch in 4]`,
# holding all four channels' (redundant, complex) curvelet coeffs simultaneously
# — ~5 GB peak at 1280×1024, which OOM-crashes the browser's WASM heap. These
# rewrites transform the thermal channel once, then process each RGB channel in
# turn (forward → fuse → backward → free), so only two channels' coeffs are live
# at a time. Verified bit-identical (maxdiff=0) to the original outputs.
def _curvelet_fuse_per_channel(visible_image, thermal_image, mode):
    rgbt = np.dstack((visible_image, thermal_image))
    curvelet_transform = SimpleUDCT(
        shape=thermal_image.shape, nscales=4, nbands_per_direction=4, alpha=0.3, winthresh=1e-5
    )
    coeff_th = curvelet_transform.forward(rgbt[:, :, 3])
    fused_channels = []
    for i in range(3):
        coeff_v = curvelet_transform.forward(rgbt[:, :, i])
        c_fused = _fuse_curvelet_scales(coeff_v, coeff_th, mode)
        fused_channels.append(curvelet_transform.backward(c_fused))
        del coeff_v, c_fused
    fused_image = np.stack(fused_channels, axis=-1)
    return normalize(fused_image)


def processCurveletCoeffsMax(coeffs, curvelet_transform):
    # Retained for API compatibility (callers that already hold all-channel coeffs).
    fused_coeffs = [_fuse_curvelet_scales(coeffs[i], coeffs[3], 'max') for i in range(3)]
    fused_channels = [curvelet_transform.backward(coeff) for coeff in fused_coeffs]
    fused_image = np.stack(fused_channels, axis=-1)
    fused_image = normalize(fused_image)
    return fused_image

@save_npmat_if_path
def combine_rgbt_curvelet_max(visible_image, thermal_image):
    return _curvelet_fuse_per_channel(visible_image, thermal_image, 'max')

@save_npmat_if_path
def combine_rgbt_curvelet(visible_image, thermal_image):
    fused_image = _curvelet_fuse_per_channel(visible_image, thermal_image, 'avg')
    fused_image_max = _curvelet_fuse_per_channel(visible_image, thermal_image, 'max')
    return fused_image, fused_image_max, 'curvelet', 'curvelet_max'




if __name__ == '__main__':
    import matplotlib.pyplot as plt

    lwir_image_path = '/home/arvc/eeha/kaist-cvpr15/images/set00/V000/lwir/I01689.jpg'
    visible_image_path = lwir_image_path.replace('/lwir/', '/visible/')
    
    visible_image = cv.imread(visible_image_path)
    lwir_image = cv.imread(lwir_image_path, cv.IMREAD_GRAYSCALE)

    # decorated_function = time_execution_measure(combine_hsvt_wavelet)
    # hsvt_wavelet_image, hsvt_wavelet_time_execution = decorated_function(visible_image, lwir_image)
    
    decorated_function = time_execution_measure(combine_rgbt_wavelet)
    rgbt_wavelet_image, rgbt_wavelet_time_execution = decorated_function(visible_image, lwir_image)
    
    # decorated_function = time_execution_measure(combine_hsv_curvelet)
    # hsv_curvelet_image, hsv_curvelet_time_execution = decorated_function(visible_image, lwir_image)
    
    decorated_function = time_execution_measure(combine_rgbt_curvelet)
    rgbt_curvelet_image, rgb_curvelet_time_execution = decorated_function(visible_image, lwir_image)
    
    data = [
        [visible_image, f'Original image (Visible)'],
        # [hsvt_wavelet_image, f'HSV+Wavelet ({hsvt_wavelet_time_execution:.2f} s)'],
        # [hsv_curvelet_image, f'HSV+Curvelet ({hsv_curvelet_time_execution:.2f} s)'],
        [lwir_image, f'Original image (LWIR)'],
        [rgbt_wavelet_image, f'RGB+Wavelet ({rgbt_wavelet_time_execution:.2f} s)'],
        [rgbt_curvelet_image, f'RGB+Curvelet ({rgb_curvelet_time_execution:.2f} s)']
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 12))
    for ax, (img, title) in zip(axes.flatten(), data):
        ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')
    
    # plt.tight_layout()
    plt.show()
