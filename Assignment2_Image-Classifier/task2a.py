import skimage
import numpy as np
import utils


def MaxPool2d(im: np.array,
              kernel_size: int):
    """ A function that max pools an image with size kernel size.
    Assume that the stride is equal to the kernel size, and that the kernel size is even.

    Args:
        im: [np.array of shape [H, W, 3]]
        kernel_size: integer
    Returns:
        im: [np.array of shape [H/kernel_size, W/kernel_size, 3]].
    """
    stride = kernel_size
    ### START YOUR CODE HERE ### (You can change anything inside this block) 
    stride = kernel_size
    H_in, W_in, C_in = im.shape

    # calculate the output dimensions after being maxpooled
    H_out = H_in // kernel_size
    W_out = W_in // kernel_size
    # C_out = C_in

    # new zero-padded matrix to receive updates for maxpooling
    mp_image = np.zeros((H_out, W_out, C_in))
    H_half, W_half, C_half = mp_image.shape

    for j in range(H_half):
        for i in range(W_half):
            for k in range(C_half):
                mp_image[j, i, k] = np.max(im[
                                           (j * stride):((j + 1) * stride),
                                           (i * stride):((i + 1) * stride),
                                           k])
    return mp_image
    ### END YOUR CODE HERE ### 


if __name__ == "__main__":

    # DO NOT CHANGE
    im = skimage.data.chelsea()
    im = utils.uint8_to_float(im)
    max_pooled_image = MaxPool2d(im, 4)

    utils.save_im("chelsea.png", im)
    utils.save_im("chelsea_maxpooled.png", max_pooled_image)

    im = utils.create_checkerboard()
    im = utils.uint8_to_float(im)
    utils.save_im("checkerboard.png", im)
    max_pooled_image = MaxPool2d(im, 2)
    utils.save_im("checkerboard_maxpooled.png", max_pooled_image)