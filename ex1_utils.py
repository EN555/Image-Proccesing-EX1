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
    # divide to two cases
    if representation == 1:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  # convert to grey scale
    else:
        img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)  # convert to RGB
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
    plt.axis("off")  # remove the axis from the image
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
    isRGB = False
    img_to_work_with = imgOrig
    if imgOrig.shape[-1] == 3:
        isRGB = True
    if isRGB:
        img_to_work_with = (transformRGB2YIQ(imgOrig))[:,:,0]
    return isRGB, img_to_work_with

def Y_to_RGB(RGBimg: np.ndarray,y_update: np.array)-> np.ndarray:
    YIQ = transformRGB2YIQ(RGBimg)
    YIQ[:, :, 0] = y_update
    new_img = transformYIQ2RGB(YIQ)
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
    img_flatten = conv_img.flatten()  # return 1-D array of all the pixels
    img_flatten = img_flatten * 255
    img_flatten = np.around(img_flatten)  # make sure that all the numbers are integers
    histOrg, bins_edges = np.histogram(img_flatten, 256, [0, 255])  # hist is the sum of each pixels from the image

    # create the new image
    new_image = np.cumsum(histOrg)
    new_image = new_image / new_image.max()  # histOrg.max() is the maximum , we got the percentage
    look_up_table = np.zeros(256)
    for loc in range(len(new_image)):
        new_color = int(np.floor(255 * new_image[loc]))
        look_up_table[loc] = new_color  # represent the intensity in this pixel
    # insert to the image
    new_im = np.zeros_like(conv_img, dtype=np.float)
    for old_color, new_color in enumerate(look_up_table):
        new_im[conv_img*255 == old_color] = new_color
    if isRGB:
        new_im = Y_to_RGB(imgOrig, new_im)
    # create the histEQ
    new_img_faltten = new_im.flatten()
    new_img_faltten = np.around(new_img_faltten)
    histEQ, bins_edges = np.histogram(new_img_faltten, 256, [0, 255])
    return new_im, histOrg, histEQ

def error(imgOrig:np.array, imgCurr: np.array)-> float:
    sub_img = np.subtract(imgOrig, imgCurr)
    sum_to_square_img = np.sum(np.square(sub_img))
    size = imgOrig.size
    res = np.sqrt(sum_to_square_img)/size
    return res

def divide_histogram(n: int)->np.ndarray:
    # divide to n partition (n+1 bondries) start at 0 and finish at 255
    boundries_list = np.zeros(n+1, dtype=int)
    jumps = int(255/n)
    for i in range(1,n):
        boundries_list[i] = boundries_list[i-1] + jumps
    boundries_list[n] = 255
    return boundries_list

def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    isRGB, conv_img = RGB_gray_scale(imOrig)
    if np.amax(conv_img) <= 1:  # so the picture is normalized
        conv_img = conv_img * 255
    conv_img = conv_img.astype('uint8')
    # create histogram from the image
    hist, bins = np.histogram(conv_img, 256, [0, 255])
    # divide the range [0,255] to nQuant+1
    z = divide_histogram(nQuant)    # represent the divdiation of all the range
    q = np.zeros(nQuant)   # represent the average of every cell
    # need to return
    error_list = list()
    quant_img = list()
    size = conv_img.shape
    for i in range(nIter):  # run n times on the boundries
        img_up = np.zeros(size)
        for divide in range(len(q)):
            # do average on the range in the z
            if divide == len(q)-1:  # if it's the last cell
                right_bound = z[divide + 1] + 1  # we use at arange so we need 256
            else:
                right_bound = z[divide + 1]
            curr_cell = np.arange(z[divide], right_bound)
            q[divide] = np.average(curr_cell, weights=hist[z[divide]: right_bound])  # give weight for every pixel
            # update the picture
            cond = np.logical_and(conv_img >= z[divide], conv_img < right_bound)
            img_up[cond] = q[divide]

        # update the new boundries
        for i in range(1, len(z)-1):
            z[i] = (q[i-1] + q[i])/2

        # check the MSE and keep him
        MSE = error(conv_img/255.0, img_up/255.0)
        print(MSE)
        error_list.append(MSE)

        # return to the picture
        new_im = None
        if isRGB:
            img_up = Y_to_RGB(imOrig, img_up/255.0)
        quant_img.append(img_up)

        # check if to terminate the iterations
        if len(error_list) >= 2:
            if np.abs(error_list[-1] - error_list[-2]) < 0.000000001: # represent if it's converge
                break
    plt.plot(error_list)
    plt.show()
    return quant_img, error_list

if __name__ == '__main__':
    # transformRGB2YIQ(cv2.imread("dark.jpg"))
    # plt.imshow(transformRGB2YIQ(imReadAndConvert("beach.jpg",2)))
    # plt.show()
    # new_im, histOrg, histEQ = hsitogramEqualize(cv2.imread("dark.jpg"))
    img = imReadAndConvert("dark.jpg", 1)
    img_lst, err_lst = quantizeImage(img, 3, 20)
