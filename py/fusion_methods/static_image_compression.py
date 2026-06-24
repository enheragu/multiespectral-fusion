#!/usr/bin/env python3
# encoding: utf-8
"""
    Creates different approachs of mixing RGB with Thermal images
    hsvt -> combines v + t and reescalate that channel
    rgbt -> averages each chanel with thermal data (r+t/2)
    4ch -> Stores ndarray with all foru channels as [b,g,r,t]
    vths -> Intensiti from visible, Thermal and HS compressed in one channel (4 bits of each channel).
        It seems that most of the relevant information is on Intensity channels, and pretty little in the color part. 
        So color is compressed in one channel.
        4 bits shifted to the right (same as 16 division, but faster)
        8 7 6 5 4 3 2 1 -> Keeps from 8 to 5 bits from both images
        8h 7h 6h 5h 8v 7v 6v 5v -> shifts one of the channels back and adds them
    vt -> Removes color information having one channel for Intensity from visual image, Thermal channel and the average of both in the third channel.
"""
import numpy as np
import cv2 as cv

from utils import log, bcolors
from Dataset.decorators import time_execution_measure, save_image_if_path, save_npmat_if_path

def rescaleChannel(channel, max_value, new_max):
    channel = new_max * (channel / max_value)
    channel = channel.astype(np.uint8)
    return channel

def channelAverage(channel1, channel2):
    channel1 = channel1.astype(np.float64)
    channel2 = channel2.astype(np.float64)
    return rescaleChannel(channel=channel1+channel2, max_value=255+255, new_max=255)

def channelProduct(channel1, channel2):
    channel1 = channel1.astype(np.float64)
    channel2 = channel2.astype(np.float64)
    return rescaleChannel(channel=channel1*channel2, max_value=255*255, new_max=255)


@save_image_if_path   
def combine_hsvt(visible_image, thermal_image):
    h,s,v = cv.split(cv.cvtColor(visible_image, cv.COLOR_BGR2HSV))
    th_channel = thermal_image

    intensity = channelAverage(v, th_channel)

    hsvt_image = cv.merge([h, s, intensity])
    hsvt_image = cv.cvtColor(hsvt_image, cv.COLOR_HSV2BGR)
    
    return hsvt_image
    

@save_image_if_path
def combine_rgbt(visible_image, thermal_image):
    b,g,r = cv.split(visible_image)
    th_channel = thermal_image
    th_channel = th_channel.astype(np.float64)

    # Apply fusion per channel and keep the transformed outputs.
    b = channelAverage(b, th_channel)
    g = channelAverage(g, th_channel)
    r = channelAverage(r, th_channel)

    rgbt_image = cv.merge([b,g,r])
    
    return rgbt_image

@save_image_if_path
def combine_rgbt_v2(visible_image, thermal_image):
    b,g,r = cv.split(visible_image)
    th_channel = thermal_image
    th_channel = th_channel.astype(np.float64)

    # Apply multiplicative fusion per channel and keep the transformed outputs.
    b = channelProduct(b, th_channel)
    g = channelProduct(g, th_channel)
    r = channelProduct(r, th_channel)

    rgbt_image = cv.merge([b,g,r])
    
    return rgbt_image

@save_image_if_path
def combine_rgbtalpha(visible_image, thermal_image, alpha_min=0.10, alpha_max=0.7, gamma=1.4):
    """Thermal-guided alpha blending.

    The thermal image is converted into a per-pixel alpha map in [alpha_min, alpha_max],
    then used to blend visible (BGR) with thermal replicated to 3 channels.
    """
    visible_f = visible_image.astype(np.float32)
    thermal_f = thermal_image.astype(np.float32)

    thermal_norm = np.clip(thermal_f / 255.0, 0.0, 1.0)
    alpha_map = np.power(thermal_norm, max(float(gamma), 1e-6))

    alpha_min = float(np.clip(alpha_min, 0.0, 1.0))
    alpha_max = float(np.clip(alpha_max, 0.0, 1.0))
    if alpha_max < alpha_min:
        alpha_min, alpha_max = alpha_max, alpha_min
    alpha_map = alpha_min + (alpha_max - alpha_min) * alpha_map

    alpha_3ch = alpha_map[..., None]
    thermal_3ch = np.repeat(thermal_f[..., None], 3, axis=2)
    fused = (1.0 - alpha_3ch) * visible_f + alpha_3ch * thermal_3ch

    return np.clip(fused, 0, 255).astype(np.uint8)

@save_image_if_path
def combine_vths(visible_image, thermal_image):
    h,s,v = cv.split(cv.cvtColor(visible_image, cv.COLOR_BGR2HSV))
    th_channel = thermal_image

    # Pack 4 MSBs from H and 4 MSBs from S into one byte: HHHHSSSS.
    h_nibble = (h >> 4).astype(np.uint8)
    s_nibble = (s >> 4).astype(np.uint8)
    hs = ((h_nibble << 4) | s_nibble).astype(np.uint8)

    vths_image = cv.merge([v,th_channel,hs])
    
    return vths_image


@save_image_if_path
def combine_vths_v2(visible_image, thermal_image):
    h,s,v = cv.split(cv.cvtColor(visible_image, cv.COLOR_BGR2HSV_FULL))
    th_channel = thermal_image

    # Quantize H and S to 4 bits and pack as HHHHSSSS.
    h_nibble = (h // 16).astype(np.uint8)
    s_nibble = (s // 16).astype(np.uint8)
    hs = ((h_nibble << 4) | s_nibble).astype(np.uint8)

    vths_image = cv.merge([v,th_channel,hs])
    
    return vths_image


@save_image_if_path
def combine_vths_v3(visible_image, thermal_image):
    h,s,v = cv.split(cv.cvtColor(visible_image, cv.COLOR_BGR2HSV_FULL))
    th_channel = thermal_image
         
    hs = channelAverage(h, s)

    vths_image = cv.merge([v,th_channel,hs])
    
    return vths_image


@save_image_if_path
def combine_vt(visible_image, thermal_image):
    h,s,v = cv.split(cv.cvtColor(visible_image, cv.COLOR_BGR2HSV))
    th_channel = thermal_image
    
    both = channelAverage(v, th_channel)
    vt_image = cv.merge([v,th_channel,both])
    
    return vt_image


@save_npmat_if_path
def combine_4ch(visible_image, thermal_image):
    b,g,r = cv.split(visible_image)
    th_channel = thermal_image

    ch4_image = cv.merge([b,g,r,th_channel])

    return ch4_image


"""
    Creates 4ch structure with just visible image
"""
@save_npmat_if_path
def combine_4ch_visible(visible_image, thermal_image):
    b,g,r = cv.split(visible_image)
    th_channel = np.zeros_like(thermal_image)

    ch4_image = cv.merge([b,g,r,th_channel])

    return ch4_image

"""
    Creates 4ch structure with just thermal image
"""
@save_npmat_if_path
def combine_4ch_lwir(visible_image, thermal_image):
    b,g,r = cv.split(np.zeros_like(visible_image))
    th_channel = thermal_image

    ch4_image = cv.merge([b,g,r,th_channel])

    return ch4_image


@save_npmat_if_path
def combine_lwir_npy(visible_image, thermal_image):
    th_channel = thermal_image
    th_image = cv.merge([th_channel,th_channel,th_channel]).astype(np.uint8)
    return th_channel


@save_npmat_if_path
def combine_vt_2ch(visible_image, thermal_image):
    h,s,v = cv.split(cv.cvtColor(visible_image, cv.COLOR_BGR2HSV))
    th_channel = thermal_image
    vt_image = cv.merge([v,th_channel]).astype(np.uint8)
    return vt_image