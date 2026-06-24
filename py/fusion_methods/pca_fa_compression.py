#!/usr/bin/env python3
# encoding: utf-8
"""
    Creates different approachs of mixing RGB with Thermal images
    PCA -> Principal component analysis (based on covariation between channels)
    FA -> Factorial analysis (based on correlation between channels)
"""

import os
import numpy as np
import cv2 as cv

import scipy.stats
import pickle
from sklearn.decomposition import PCA, FactorAnalysis

from utils import log, bcolors
from Dataset.decorators import time_execution_measure, save_image_if_path, save_npmat_if_path
from Dataset.fusion_methods.normalization import normalize


def draw_text(img, text,
          font=cv.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=1,
          font_thickness=1,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size

"""
    Common matrix analsisi for PCA and FA for different use cases
"""
def MatrixAnalisis(data_vector, mat, img_shape, components, standarice = True):
    if standarice:
        mean = np.mean(data_vector, axis=0)
        std = np.std(data_vector, axis=0)
        data_vector_std = (data_vector - mean) / std
    else:
        data_vector_std = data_vector

    mat = np.cov(data_vector_std, ddof = 1, rowvar = False)
    # eigenvalues, eigenvectors = np.linalg.eig(mat)
    eigenvalues, eigenvectors = np.linalg.eigh(mat) # -> faster but only can be used for symmetric matrix
    # eith returns eigenvalues in ascendin order already
    
    signs = np.sign(eigenvectors[0, :]) # extract sign of each first component of each column
    # Multiply matriz with them to make all first component positive
    eigenvectors = eigenvectors * signs

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Transformar los datos originales a los nuevos componentes
    k = components # select the number of principal components
    
    # use sorted_eigenvalues to ensure the explained variances correspond to the eigenvectors
    explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)
    principal_components = sorted_eigenvectors[:, :k]

    transformed_data = np.dot(data_vector_std, principal_components)
    
    # Re-escale intensity
    if standarice:
        transformed_data = transformed_data * std[:k] + mean[:k]
    
    ## Store eigenvalue and vectors to file to later study
    # autov_path = path.split("/")[:-1]
    # autov_path = "/".join(autov_path) + "/00_eigenvalue_vector.yaml"

    # import yaml
    # from yaml.loader import SafeLoader
    # import os

    # data = {}
    # if os.path.exists(autov_path) and os.path.isfile(autov_path):
    #     with open(autov_path, 'r') as file:
    #         data = yaml.safe_load(file)
    #         data = {} if data is None else data
    
    # data[path] = {"img": str(path), "eigenvectors": (eigenvectors.transpose().tolist()), "eigenvalues": (eigenvalues.tolist())}
    # with open(autov_path, 'w') as file:
    #     yaml.safe_dump(data, file)

    # total_explained_variance = sum(explained_variance[:k])
    # log(f"[RGBThermalMix::combine_pca] Explained variance for {path} with {k} principal components is: {total_explained_variance}")
    # print(f"{sorted_eigenvectors[:,:k] = }")
    

    image_vector = [transformed_data[:, i].reshape(img_shape) for i in range(components)]
    image = cv.merge(image_vector)
    
    # # if test:
    # condition_01 = sorted_eigenvalues[0] / sorted_eigenvalues[1]
    # condition_12 = sorted_eigenvalues[1] / sorted_eigenvalues[2]
    # condition_23 = sorted_eigenvalues[2] / sorted_eigenvalues[3]
    # w, h = draw_text(image, f"eigenvalues[0]={np.around(eigenvalues[0],6)}", pos=(10, 10))
    # w, h = draw_text(image, f"eigenvalues[1]={np.around(eigenvalues[1],6)}", pos=(10, 20 + h))
    # w, h = draw_text(image, f"eigenvalues[2]={np.around(eigenvalues[2],6)}", pos=(10, 40 + h))
    # w, h = draw_text(image, f"eigenvalues[3]={np.around(eigenvalues[3],6)}", pos=(10, 60 + h))
    # # Eigenvectors are in the columns, transpose it first
    # w, h = draw_text(image, f"eigenvectors[0]={list(np.around(np.array(eigenvectors.transpose()[0]),6))}", pos=(10, 80 + h))
    # w, h = draw_text(image, f"eigenvectors[1]={list(np.around(np.array(eigenvectors.transpose()[1]),6))}", pos=(10, 100 + h))
    # w, h = draw_text(image, f"eigenvectors[2]={list(np.around(np.array(eigenvectors.transpose()[2]),6))}", pos=(10, 120 + h))
    # w, h = draw_text(image, f"eigenvectors[3]={list(np.around(np.array(eigenvectors.transpose()[3]),6))}", pos=(10, 140 + h))
    # w, h = draw_text(image, f"{explained_variance[0] = }", pos=(10, 160 + h))
    # w, h = draw_text(image, f"{explained_variance[1] = }", pos=(10, 180 + h))
    # w, h = draw_text(image, f"{explained_variance[2] = }", pos=(10, 200 + h))
    # w, h = draw_text(image, f"{explained_variance[3] = }", pos=(10, 220 + h))
    
    return image


def combine_rgbt_pca_toXch(visible_image, thermal_image, output_channels = 3):

    # NOISE FILTERING WITH BETTER EDGE PRESERVATION
    visible_imgage_filered = visible_image #cv.bilateralFilter(visible_image, 9, 50, 50) 
    thermal_image_filtered = thermal_image #cv.bilateralFilter(thermal_image, 9, 50, 50) 

    b,g,r = cv.split(visible_imgage_filered)
    th = thermal_image_filtered
    img_shape = th.shape

    data_vector = np.array([f.flatten() for f in [b,g,r,th]]).transpose()   
    cov_mat = np.cov(data_vector, ddof = 1, rowvar = False) 
    
    image = MatrixAnalisis(data_vector, cov_mat, img_shape, components = output_channels, standarice = True)
    image = image.astype(np.uint8) # Recast to image format type after all operations
    
    return image


def combine_rgbt_fa_toXch(visible_image, thermal_image, output_channels = 3):
    
    # NOISE FILTERING WITH BETTER EDGE PRESERVATION
    visible_imgage_filered = visible_image #cv.bilateralFilter(visible_image, 9, 50, 50) 
    thermal_image_filtered = thermal_image #cv.bilateralFilter(thermal_image, 9, 50, 50) 

    b,g,r = cv.split(visible_imgage_filered)
    th = thermal_image_filtered
    img_shape = th.shape

    data_vector = np.array([f.flatten() for f in [b,g,r,th]]).transpose()

    # cov_mat = np.corrcoef(data_vector_std) # -> Pearson correlation. Careful!!
    cov_mat, p_matrix = scipy.stats.spearmanr(data_vector, axis=0) # -> Spearman. axis whether columns (0) or rows (1) represent the features

    image = MatrixAnalisis(data_vector, cov_mat, img_shape, components = output_channels, standarice = False)
    image = image.astype(np.uint8) # Recast to image format type after all operations

    return image


@save_npmat_if_path
def combine_rgbt_pca_to3ch(visible_image, thermal_image):
    # EEHA - PRevious version
    # image = combine_rgbt_pca_toXch(visible_image, thermal_image, 3)
    # return image

    b,g,r = cv.split(visible_image)
    th = thermal_image

    img = cv.merge([b,g,r,th])
    h, w, ch = img.shape

    # Need 2d Array
    img_reshaped = img.reshape(-1, 4)

    pca = PCA(n_components=3)
    img_pca = pca.fit_transform(img_reshaped)

    img_compressed = img_pca.reshape(h, w, 3)
    img_compressed = normalize(img_compressed)
    return img_compressed



def factor_analysis_patch(img_patch, n_components=3):
    h, w, ch = img_patch.shape
    reshaped = img_patch.reshape(-1, ch)
    fa = FactorAnalysis(n_components=n_components)
    transformed = fa.fit_transform(reshaped)
    
    ## Patch normalization can cause diferences between patches
    # transformed = (transformed - transformed.min()) / (transformed.max() - transformed.min())
    # transformed = (transformed * 255).astype(np.uint8)
    return transformed.reshape(h, w, n_components)

def process_image_in_patches(visible_image, thermal_image, patch_size=128, n_components=3):
    b, g, r = cv.split(visible_image)
    th = thermal_image
    img = cv.merge([b, g, r, th])
    h, w, ch = img.shape

    output = np.zeros((h, w, n_components), dtype=np.float32)

    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            y_end = min(y + patch_size, h)
            x_end = min(x + patch_size, w)
            patch = img[y:y_end, x:x_end, :]
            out_patch = factor_analysis_patch(patch, n_components)
            output[y:y_end, x:x_end, :] = out_patch

    # Global image normalization
    output = normalize(output)
    return output

@save_npmat_if_path
def combine_rgbt_fa_to3ch(visible_image, thermal_image):
    # EEHA - Older version
    # image = combine_rgbt_fa_toXch(visible_image, thermal_image, 3)
    # return image   

    ## EEHA - Interesting but slow! (with patches in single thread is even slower...)
    b,g,r = cv.split(visible_image)
    th = thermal_image

    img = cv.merge([b,g,r,th])
    h, w, ch = img.shape

    img_reshaped = img.reshape(-1, 4)

    fa = FactorAnalysis(n_components=3)
    img_fa = fa.fit_transform(img_reshaped)

    img_compressed = img_fa.reshape(h, w, 3)
    img_compressed = normalize(img_compressed)
    
    # img_compressed = process_image_in_patches(visible_image, thermal_image)
    return img_compressed

@save_npmat_if_path
def combine_rgbt_pca_to1ch(visible_image, thermal_image):
    image = combine_rgbt_pca_toXch(visible_image, thermal_image, 1)
    image = cv.merge([image,image,image])
    return image


@save_npmat_if_path
def combine_rgbt_fa_to1ch(visible_image, thermal_image):
    image = combine_rgbt_fa_toXch(visible_image, thermal_image, 1)
    image = cv.merge([image,image,image])
    return image    
 

# From https://ieeexplore.ieee.org/document/10095874
# Alpha version:  Ifus=αPCA⋅I1+(1−αPCA)⋅I2
@save_npmat_if_path
def combine_rgbt_alpha_pca_to3ch(visible_image, thermal_image):
    img_rgb = visible_image
    img_th = thermal_image.astype(np.float32)
    img_rgb_gray = cv.cvtColor(visible_image, cv.COLOR_BGR2GRAY).astype(np.float32)

    V1 = np.var(img_th)
    V2 = np.var(img_rgb_gray)
    cov_mat = np.cov(img_th.flatten(), img_rgb_gray.flatten())[0, 1]

    discriminant = np.sqrt((V1 - V2)**2 + 4*cov_mat**2)
    lambda1 = (V1 + V2 + discriminant) / 2
    lambda2 = (V1 + V2 - discriminant) / 2

    if cov_mat != 0:
        eigen_vec_v1 = 1
        eigen_vec_v2 = (lambda1 - V1) / cov_mat
    else:  # Si C=0, eigen vectors
        eigen_vec_v1 = 1
        eigen_vec_v2 = 0
    eigen_vec = np.array([eigen_vec_v1, eigen_vec_v2])

    norm = np.sqrt(eigen_vec_v1**2 + eigen_vec_v2**2)
    alpha = 1 / norm

    fused_gray = alpha * img_th + (1-alpha) * img_rgb_gray
    epsilon = 1e-6 # Avoid zero division
    ratio = fused_gray/(img_rgb_gray + epsilon)
    ratio_3ch = np.repeat(ratio[:, :, np.newaxis], 3, axis=2)
    F_color = ratio_3ch * img_rgb
    fused_rgb = np.clip(F_color, 0, 255).astype(np.uint8)
    
    return fused_rgb






@save_npmat_if_path
def combine_hsvt_pca_to3ch(visible_image, thermal_image):
    # NOISE FILTERING WITH BETTER EDGE PRESERVATION
    visible_imgage_filered = visible_image #cv.bilateralFilter(visible_image, 9, 50, 50) 
    thermal_image_filtered = thermal_image #cv.bilateralFilter(thermal_image, 9, 50, 50) 

    h,s,v = cv.split(cv.cvtColor(visible_image, cv.COLOR_BGR2HSV))
    th = thermal_image_filtered
    img_shape = th.shape

    data_vector = np.array([f.flatten() for f in [h,s,v,th]]).transpose()   
    cov_mat = np.cov(data_vector, ddof = 1, rowvar = False) 
    
    image = MatrixAnalisis(data_vector, cov_mat, img_shape, components = 3, standarice = True)

    return image


"""
    Gets PCA eigenvectors and eigenvalues for given dataset
"""
pca = None                  # Avoid multiple load of data. Load once and reuse
fa_eigen_vectors = None

def preprocess_datset_matrix():
    from Dataset.constants import kaist_yolo_dataset_path
    from Dataset.constants import images_folder_name, lwir_folder_name, visible_folder_name
    flattened_dataset_output_path = f'{kaist_yolo_dataset_path}/.flattened_dataset_cache.npz'

    if os.path.exists(flattened_dataset_output_path):
        return np.load(flattened_dataset_output_path)['data_matrix']

    image_dict = {} # Store image and paths without repeating

    # Get all unique LWIR and RGB image pairs
    # Iterate each set in dataset path, get LWIR and RGB image from /images/...
    for folder in os.listdir(kaist_yolo_dataset_path):
        if not os.path.isdir(os.path.join(kaist_yolo_dataset_path,folder)):
            continue
        rgb_image_path = os.path.join(kaist_yolo_dataset_path,folder,visible_folder_name,images_folder_name)
        lwir_image_path = os.path.join(kaist_yolo_dataset_path,folder,lwir_folder_name,images_folder_name)

        for image in os.listdir(rgb_image_path):
            if not image.endswith('.png'):
                continue
            
            image_dict[image] = [rgb_image_path,lwir_image_path]

    # Accumulate matrix and flatten to compute PCA

    image_data = []
    for img, (rgb_path, lwir_path) in image_dict.items():
        visible_image = cv.imread(os.path.join(rgb_path,img), cv.IMREAD_COLOR)
        thermal_image = cv.imread(os.path.join(lwir_path,img), cv.IMREAD_GRAYSCALE)
        
        b, g, r = cv.split(visible_image)
        th = thermal_image
        
        # Aplanar las imágenes
        data_vector = np.array([f.flatten() for f in [b, g, r, th]]).transpose()
        image_data.append(data_vector)
        
    data_matrix = np.vstack(image_data)
    
    np.savez_compressed(flattened_dataset_output_path, data_matrix = data_matrix)    
    return data_matrix

"""
    Gets PCA eigenvectors and eigenvalues for given dataset
"""
def preprocess_rgbt_pca_full(n_components = 3):
    from Dataset.constants import kaist_yolo_dataset_path
    pca_output_path = f'{kaist_yolo_dataset_path}/.pca_cache.pkl'

    log(f"Compute general PCA eigenvectors and values to later compress images.")
    data_matrix = preprocess_datset_matrix()

    data_matrix_std = (data_matrix - data_matrix.mean(axis=0)) / data_matrix.std(axis=0)
    pca = PCA(n_components=n_components)
    pca.fit_transform(data_matrix)

    with open(pca_output_path, 'wb') as file:
        pickle.dump({'pca':pca}, file)

"""
    Gets FA eigenvectors and eigenvalues for given dataset
"""
def preprocess_rgbt_fa_full():
    from Dataset.constants import kaist_yolo_dataset_path
    fa_output_path = f'{kaist_yolo_dataset_path}/.fa_cache.pkl'

    log(f"Compute general FA eigenvectors and values to later compress images.")
    data_matrix = preprocess_datset_matrix()
    pass


@save_npmat_if_path
def combine_rgbt_pca_full(visible_image, thermal_image, n_components = 3):
    if pca is None:
        with open(pca_output_path, 'rb') as file:
            pca = pickle.load(file)['pca']

    visible_imgage_filered = visible_image #cv.bilateralFilter(visible_image, 9, 50, 50) 
    thermal_image_filtered = thermal_image #cv.bilateralFilter(thermal_image, 9, 50, 50) 

    b,g,r = cv.split(visible_imgage_filered)
    th = thermal_image_filtered
    img_shape = th.shape

    data_vector = np.array([f.flatten() for f in [b,g,r,th]]).transpose()   
    data_vector_std = (data_vector - data_vector.mean()) / data_vector.std() # standarize 
    
    transformed_image = pca.transform(data_vector_std)
    reconstructed_image_std = pca.inverse_transform(transformed_image)
    reconstructed_image = reconstructed_image_std.reshape(visible_image.shape[0], visible_image.shape[1], 4)
    
    image_vector = [reconstructed_image[:, i].reshape(img_shape) for i in range(n_components)]
    image = cv.merge(image_vector)

    return image


@save_npmat_if_path
def combine_rgbt_fa_full(visible_image, thermal_image):
    global fa_eigen_vectors
    visible_imgage_filered = visible_image #cv.bilateralFilter(visible_image, 9, 50, 50) 
    thermal_image_filtered = thermal_image #cv.bilateralFilter(thermal_image, 9, 50, 50) 

    b,g,r = cv.split(visible_imgage_filered)
    th = thermal_image_filtered
    img_shape = th.shape

    if not fa_eigen_vectors:
        fa_eigen_vectors = np.load(fa_output_path)
    data_vector = np.array([f.flatten() for f in [b,g,r,th]]).transpose()   
    image = np.matmul(data_vector, fa_eigen_vectors[:,:4])

    return image