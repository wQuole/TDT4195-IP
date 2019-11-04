import matplotlib.pyplot as plt
import os
import numpy as np

image_output_dir = "image_processed"
os.makedirs(image_output_dir, exist_ok=True)


def save_im(imname, im, cmap=None):
    impath = os.path.join(image_output_dir, imname)
    plt.imsave(impath, im, cmap=cmap)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.212, 0.7152, 0.0722])

def greyscale(im):
    """ Converts an RGB image to greyscale
    
    Args:
        im ([type]): [np.array of shape [H, W, 3]]
    
    Returns:
        im ([type]): [np.array of shape [H, W]]
    """
    return rgb2gray(im)

def inverse(im):
    """ Finds the inverse of the greyscale image
    
    Args:
        im ([type]): [np.array of shape [H, W]]
    
    Returns:
        im ([type]): [np.array of shape [H, W]]
    """
    return 255 - im


if __name__ == "__main__":
    im = plt.imread("images/lake.jpg")
    gray_im = greyscale(im)
    inverse_im = inverse(gray_im)
    save_im("lake_greyscale.jpg", gray_im, cmap="gray")
    save_im("lake_inverse.jpg", inverse_im, cmap="gray")
