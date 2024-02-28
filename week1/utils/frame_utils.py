import cv2
import numpy as np
import math
import statistics


# def shadow_remove(img):
#     rgb_planes = cv2.split(img)
#     result_norm_planes = []
#     for plane in rgb_planes:
#         dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
#         bg_img = cv2.medianBlur(dilated_img, 21)
#         diff_img = 255 - cv2.absdiff(plane, bg_img)
#         norm_img = cv2.normalize(
#             diff_img,
#             None,
#             alpha=0,
#             beta=255,
#             norm_type=cv2.NORM_MINMAX,
#             dtype=cv2.CV_8UC1,
#         )
#         result_norm_planes.append(norm_img)
#     shadowremov = cv2.merge(result_norm_planes)
#     return shadowremov


def shadow_remove(img):

    img = np.float64(img)
    blue, green, red = cv2.split(img)

    # Avoid division by zero
    blue[blue == 0] = 1
    green[green == 0] = 1
    red[red == 0] = 1

    # Calculate div for normalization
    div = np.multiply(np.multiply(blue, green), red) ** (1.0 / 3)

    # Log chromaticity of each channel
    a = np.log1p((blue / div) - 1)
    b = np.log1p((green / div) - 1)
    c = np.log1p((red / div) - 1)

    # Stack for RGB channels
    rho = np.concatenate((np.atleast_3d(c), np.atleast_3d(b), np.atleast_3d(a)), axis=2)

    # Eigenvectors for transformation
    U = np.array(
        [
            [1 / math.sqrt(2), -1 / math.sqrt(2), 0],
            [1 / math.sqrt(6), 1 / math.sqrt(6), -2 / math.sqrt(6)],
        ]
    )
    X = np.dot(rho, U.T)

    # Initialize e_t for angle calculations
    e_t = np.zeros((2, 181))
    for j in range(181):
        e_t[0][j] = math.cos(j * math.pi / 180.0)
        e_t[1][j] = math.sin(j * math.pi / 180.0)

    Y = np.dot(X, e_t)
    nel = img.shape[0] * img.shape[1]

    # Bandwidth calculation
    bw = np.zeros((1, 181))
    for i in range(181):
        bw[0][i] = (3.5 * np.std(Y[:, :, i])) * ((nel) ** (-1.0 / 3))

    # Entropy calculation
    entropy = []
    for i in range(181):
        temp = []
        comp1 = np.mean(Y[:, :, i]) - 3 * np.std(Y[:, :, i])
        comp2 = np.mean(Y[:, :, i]) + 3 * np.std(Y[:, :, i])
        for j in range(Y.shape[0]):
            for k in range(Y.shape[1]):
                if Y[j, k, i] > comp1 and Y[j, k, i] < comp2:
                    temp.append(Y[j, k, i])
        nbins = round((max(temp) - min(temp)) / bw[0][i])
        (hist, _) = np.histogram(temp, bins=nbins)
        hist = list(filter(lambda var1: var1 != 0, hist))
        hist1 = np.array([float(var) for var in hist]) / sum(hist)
        entropy.append(-1 * sum(np.multiply(hist1, np.log2(hist1))))

    # Find optimal angle
    angle = entropy.index(min(entropy))

    # Transformations for shadow removal
    e_t = np.array(
        [math.cos(angle * math.pi / 180.0), math.sin(angle * math.pi / 180.0)]
    )
    e = np.array(
        [-1 * math.sin(angle * math.pi / 180.0), math.cos(angle * math.pi / 180.0)]
    )

    I1D = np.exp(np.dot(X, e_t))

    # Apply transformations and adjustments
    p_th = np.dot(e_t.T, e_t)
    X_th = X * p_th
    mX = np.dot(X, e.T)
    mX_th = np.dot(X_th, e.T)

    mX = np.atleast_3d(mX)
    mX_th = np.atleast_3d(mX_th)

    theta = (math.pi * float(angle)) / 180.0
    theta = np.array(
        [[math.cos(theta), math.sin(theta)], [-1 * math.sin(theta), math.cos(theta)]]
    )
    alpha = np.atleast_2d(theta[0, :])
    beta = np.atleast_2d(theta[1, :])

    # Top 1% extraction and adjustments
    mX1 = mX.reshape(mX.shape[0] * mX.shape[1])
    mX1sort = np.argsort(mX1)[::-1]
    top_1_percent_index = int(0.01 * mX.shape[0] * mX.shape[1])
    mX_top = mX1[mX1sort[:top_1_percent_index]]
    mX_th_top = mX_th.reshape(mX_th.shape[0] * mX_th.shape[1])[
        mX1sort[:top_1_percent_index]
    ]
    X_E = (statistics.median(mX_top) - statistics.median(mX_th_top)) * beta.T
    X_E = X_E.T

    for i in range(X_th.shape[0]):
        for j in range(X_th.shape[1]):
            X_th[i, j, :] += X_E

    rho_ti = np.dot(X_th, U)
    c_ti = np.exp(rho_ti)
    sum_ti = np.sum(c_ti, axis=2).reshape(c_ti.shape[0], c_ti.shape[1], 1)
    r_ti = c_ti / sum_ti

    r_ti2 = np.clip(255 * r_ti, 0, 255).astype(np.uint8)

    return r_ti2


def adjust_brightness(input_frame, target_brightness, use_clahe=False):

    # Convert to grayscale to calculate current brightness
    gray = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
    current_brightness = np.mean(gray)

    # Calculate adjustment factor
    adjustment_factor = target_brightness / current_brightness

    if use_clahe:
        # Convert to LAB color space
        lab = cv2.cvtColor(input_frame, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to the L-channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # Merge the CLAHE enhanced L-channel back with the A and B channels
        limg = cv2.merge((cl, a, b))

        # Convert back to BGR color space
        adjusted_frame = cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)

        # Adjust overall brightness if needed
        adjusted_frame = cv2.convertScaleAbs(
            adjusted_frame, alpha=adjustment_factor, beta=0
        )
    else:
        # Directly adjust brightness and contrast
        adjusted_frame = cv2.convertScaleAbs(
            input_frame, alpha=adjustment_factor, beta=0
        )

    return adjusted_frame


def remove_frame_noise(image):
    """
    Remove salt-and-pepper noise from an image using median filtering and morphological closing.

    Parameters:
    - image: Input image with salt-and-pepper noise.

    Returns:
    - denoised_image: The denoised image after applying median filtering and morphological closing.
    """
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Apply median filter to remove salt-and-pepper noise
    median_filtered = cv2.medianBlur(
        binary_image, 5
    )  # You can adjust the kernel size as needed

    # Define the kernel for the morphological operation
    kernel = np.ones((5, 5), np.uint8)  # Adjust kernel size as needed

    # Apply morphological closing to close small holes inside the foreground objects
    morphological_closing = cv2.morphologyEx(median_filtered, cv2.MORPH_CLOSE, kernel)

    return morphological_closing
