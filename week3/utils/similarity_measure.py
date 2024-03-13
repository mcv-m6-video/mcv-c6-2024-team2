import numpy as np
from skimage.metrics import structural_similarity as compare_ssim


# Sum of Absolute Differences (SAD)
def calculate_SAD(block_a, block_b):
    return np.sum(np.abs(block_a - block_b))


# Sum of Squared Differences (SSD)
def calculate_SSD(block_a, block_b):
    return np.sum((block_a - block_b) ** 2)


# Normalized Cross-Correlation (NCC)
def calculate_NCC(block_a, block_b):
    mean_a = np.mean(block_a)
    mean_b = np.mean(block_b)
    numerator = np.sum((block_a - mean_a) * (block_b - mean_b))
    denominator = np.sqrt(
        np.sum((block_a - mean_a) ** 2) * np.sum((block_b - mean_b) ** 2)
    )
    return numerator / denominator if denominator != 0 else 0


# Mean Squared Error (MSE)
def calculate_MSE(block_a, block_b):
    err = np.sum((block_a.astype("float") - block_b.astype("float")) ** 2)
    err /= float(block_a.shape[0] * block_a.shape[1])
    return err


#  Structural Similarity Index (SSIM)
def calculate_ssim(block_a, block_b):
    ssim, _ = compare_ssim(block_a, block_b, full=True)
    return ssim


# Hamming Distance
def calculate_hamming(block_a, block_b):
    """
    Calculate the hamming distance between two blocks
    Binary representation of the blocks is used for the calculation.

    Args:
    block_a: np.array
    block_b: np.array

    Returns:
    hamming_distance: int
    """
    hamming_distance = np.sum(block_a != block_b)
    return hamming_distance


def get_similarity_func(similarity_measure):
    if similarity_measure == "SAD":
        return calculate_SAD
    elif similarity_measure == "SSD":
        return calculate_SSD
    elif similarity_measure == "NCC":
        return calculate_NCC
    elif similarity_measure == "MSE":
        return calculate_MSE
    elif similarity_measure == "SSIM":
        return calculate_ssim
    elif similarity_measure == "HAMMING":
        return calculate_hamming
    else:
        raise ValueError("Invalid similarity measure")
