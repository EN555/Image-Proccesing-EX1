"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, linalg

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2
import cv2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 316160464


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    try:
        path = cv2.imread(filename)
    except:
        return None
    # divide to two cases
    if representation == 1:
        img = cv2.cvtColor(path, cv2.COLOR_BGR2GRAY)  # convert to grey scale
    else:
        img = cv2.cvtColor(path, cv2.COLOR_BGR2RGB).astype(np.float32)  # convert to RGB
    img = img / 255.0  # normalize the image and represent him as float
    return img


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img = imReadAndConvert(filename, representation)
    if representation == 1:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.show()

def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:  # ndarray mean n - dimensional array
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    transform = np.array([[0.299, 0.587, 0.114],
                          [0.596, -0.275, -0.321],
                          [0.212, -0.523, 0.311]])
    res = np.dot(imgRGB, transform.T)
    return res

def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    transform = np.array([[0.299, 0.587, 0.114],
                          [0.596, -0.275, -0.321],
                          [0.212, -0.523, 0.311]])
    inverse_mat = linalg.inv(transform)     # inverse the transform matrix
    res = np.dot(imgYIQ, inverse_mat.T)
    return res

def RGB_gray_scale(imgOrig: np.ndarray)-> (bool, np.ndarray):
    """"
    :param imgOrig: numpy array of image
    :return tuple if boolean that represent if
    the image RGB or not and if it's RGB it's
    convert him to YIQ and return the Y.
    check if the numpy array is RGB or it's
    gray scale
    """
    isRGB = False
    img_to_work_with = imgOrig
    if imgOrig.shape[-1] == 3 and len(imgOrig.shape) == 3:     # if it's RGB image
        isRGB = True
    if isRGB:
        img_to_work_with = transformRGB2YIQ(imgOrig)    # convert him to YIQ
        img_to_work_with = img_to_work_with[:, :, 0]    # take only the Y
    return isRGB, img_to_work_with

def Y_to_RGB(RGBimg: np.ndarray,y_update: np.array)-> np.ndarray:
    """"
    :param RGBimg: image that represent as RGB
    :param y_update: np array of update Y
    :return RGB model with update Y
    convert an RGB image to YIQ image as
    Y update in other function
    """
    YIQ = transformRGB2YIQ(RGBimg)  # first convert to RGB
    YIQ[:, :, 0] = y_update  # insert the Y
    new_img = transformYIQ2RGB(YIQ)  # convert again to YIQ
    return new_img

def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    # check if it's RGB image and return the correct color space
    isRGB, conv_img = RGB_gray_scale(imgOrig)

    # create histogram for the original image
    conv_img = np.around(conv_img * 255).astype(np.int)
    img_flatten = conv_img.flatten()  # return 1-D array of all the pixels
    histOrg, bins_edges = np.histogram(img_flatten, 256, [0, 255])  # hist is the sum of each pixels from the image

    # create the new image
    cumsum_im = np.cumsum(histOrg)
    cumsum_im = cumsum_im / cumsum_im.max()  # histOrg.max() is the maximum , we got the percentage
    look_up_table = np.zeros(256)
    for loc in range(len(cumsum_im)):
        new_color = 255 * cumsum_im[loc]
        look_up_table[loc] = new_color  # represent the intensity in this pixel
    # insert to the image
    new_im = np.zeros_like(conv_img, dtype=np.float)
    for old_color, new_color in enumerate(look_up_table):
        new_im[conv_img == old_color] = new_color

    # create the histEQ
    new_img_faltten = new_im.flatten()  # return 1-D array of all the pixels
    new_img_faltten = np.around(new_img_faltten)
    histEQ, bins_edges = np.histogram(new_img_faltten, 256, [0, 255])

    if isRGB:   # check how to represent the image
        new_im = Y_to_RGB(imgOrig, new_im/255.0)
    return new_im, histOrg, histEQ

def error(imgOrig:np.array, imgCurr: np.array)-> float:
    """"
    :param imgOrig: array of image
    :param imgCurr: array of the up image
    :return number that represent the size of the error
    the function calculate the error between two pictures using
    MSE model for error calculation
    """
    sub_img = np.subtract(imgOrig, imgCurr)
    sum_to_square_img = np.sum(np.square(sub_img))
    size = imgOrig.size
    res = np.sqrt(sum_to_square_img)/size
    return res

def divide_histogram(n: int, hist: np.ndarray)->np.ndarray:
    """"
    the function create dividation of range of pixels according
    to the number 0f pixels, the dividation can work with image
    that have specific range of intensity like image with colors
    of gray and black
    """
    cumsum = np.cumsum(hist)
    borders = np.searchsorted(cumsum, np.array(range(1, n)) * (cumsum.max()//n))    # find the indexes where the numbers need to be
    borders = np.insert(borders, 0, 0)  # insert 0 to the start of the list
    borders =np.insert(borders, n, 256)  # insert 255 to the end of the list
    return borders

def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    isRGB, conv_img = RGB_gray_scale(imOrig)  # check if it's RGB image or gray scale
    if np.amax(conv_img) <= 1:  # so the picture is normalized
        conv_img = conv_img * 255
    conv_img = np.round(conv_img).astype(np.int)  # check that all the values represent as integers

    # create histogram from the image
    hist, bins = np.histogram(conv_img, 256, [0, 255])
    # divide the range [0,255] to nQuant+1
    z = divide_histogram(nQuant, hist)    # represent the division of all the range
    q = np.zeros(nQuant)   # represent the average of every cell

   # initial two lists that the function need to return
    error_list = list()
    quant_img = list()
    size = conv_img.shape
    for i in range(nIter):  # run n times on the boundries
        img_up = np.zeros(size)
        for divide in range(len(q)):    # divide represent the
            # do average on the range in the z
            curr_cell = np.arange(z[divide], z[divide+1])
            avg = int(np.average(curr_cell, weights=hist[z[divide]:z[divide+1]]))  # give weight for every pixel
            q[divide] = avg
            # update the picture
            cond = np.logical_and(conv_img >= z[divide], conv_img < z[divide+1])
            img_up[cond] = q[divide]

        # update the new boundries
        for i in range(1, len(z)-1):
            z[i] = (q[i-1] + q[i])/2

        # check the MSE and keep him
        MSE = error(conv_img, img_up)
        error_list.append(MSE)

        # return to the image
        new_im = None
        if isRGB:
            img_up = Y_to_RGB(imOrig, img_up/255.0)
        quant_img.append(img_up)

        # check if to terminate the iterations
        if len(error_list) >= 2:
            if np.abs(error_list[-1] - error_list[-2]) < 0.00000001:  # represent if it's converge
                break
    return quant_img, error_list

if __name__ == '__main__':
    pass