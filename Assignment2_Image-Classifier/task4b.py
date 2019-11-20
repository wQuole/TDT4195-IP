import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
import utils

### START YOUR CODE HERE ### (You can change anything inside this block)
def convolve_im(im: np.array,
                kernel: np.array,
                verbose=True):
    """ Convolves the image (im) with the spatial kernel (kernel),
        and returns the resulting image.

        "verbose" can be used for visualizing different parts of the
        convolution.

        Note: kernel can be of different shape than im.

    Args:
        im: np.array of shape [H, W]
        kernel: np.array of shape [K, K]
        verbose: bool
    Returns:
        im: np.array of shape [H, W]
    """
    ### START YOUR CODE HERE ### (You can change anything inside this block)

    # Find center of image and kernel
    im_center_h = im.shape[0] // 2
    im_center_w = im.shape[1] // 2

    kernel_center_h = kernel.shape[0] // 2
    kernel_center_w = kernel.shape[1] // 2

    # Append kernel into zero-padded kernel with HxW shape.
    if (im.shape != kernel.shape):
        spatial_kernel = np.zeros(shape=(im.shape))
        spatial_kernel[im_center_h - kernel_center_h - 1:im_center_h + kernel_center_h,
        im_center_w - kernel_center_w - 1:im_center_w + kernel_center_w] = kernel

        # Zero shift kernel to be applicable to do fft.
        kernel = spatial_kernel

    # Transform image and kernel into frequency domain.
    fft_kernel = np.fft.fft2(kernel)
    fft_im = np.fft.fft2(im)

    # Execute the convolution.
    convoluted = fft_im * fft_kernel

    # Shift convolution back to spatial domain.
    convoluted = np.fft.ifft2(convoluted)
    convoluted = np.fft.fftshift(convoluted)

    # Use the real part of the ifft, and make sure values are inside the [0,1] range.
    conv_result = np.real(convoluted)
    conv_result = np.add(conv_result, abs(conv_result.min()))
    conv_result = np.multiply(conv_result, 1 / conv_result.max())

    if verbose:
        # Use plt.subplot to place two or more images beside eachother
        plt.figure(figsize=(20, 4))
        # plt.subplot(num_rows, num_cols, position (1-indexed))
        plt.subplot(1, 5, 1)
        plt.imshow(im, cmap="gray")
        plt.subplot(1, 5, 5)
        plt.imshow(conv_result, cmap="gray")
    ### END YOUR CODE HERE ###
    return conv_result




if __name__ == "__main__":
    verbose = True  # change if you want

    # Changing this code should not be needed
    im = skimage.data.camera()
    im = utils.uint8_to_float(im)

    # DO NOT CHANGE
    gaussian_kernel = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1],
    ]) / 256
    image_gaussian = convolve_im(im, gaussian_kernel, verbose)

    # DO NOT CHANGE
    sobel_horizontal = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    image_sobelx = convolve_im(im, sobel_horizontal, verbose)

    if verbose:
        plt.show()

    utils.save_im("camera_gaussian.png", image_gaussian)
    utils.save_im("camera_sobelx.png", image_sobelx)
