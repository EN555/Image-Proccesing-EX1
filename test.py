import cv2
import numpy as np
import matplotlib.pyplot as plt
from ex1_utils import *

def check_hist_EQ(file_name: np.ndarray, representation: int):
    """"
    check pictures with low contrast
    if the function improve the picture
    """
    im = imReadAndConvert(file_name, representation)
    new_im, histOrg, histEQ = hsitogramEqualize(im)
    plt.plot(np.cumsum(histOrg))
    plt.show()
    plt.plot(histEQ)    # represent the histogram equalization
    plt.show()
    plt.plot(np.cumsum(histEQ))    # represent the cumulative sum of histogram equalization
                                   # check if it's closet to linear graph
    plt.show()
    # show the result
    if representation == 1:
        plt.imshow(new_im, cmap='gray')
    else:
        plt.imshow(new_im)
    plt.show()

def check_quantisize_im(file_name: np.ndarray, representation: int, nQuant: int, nIter: int ):
    im = imReadAndConvert(file_name, representation)

    quant_img, error_list = quantizeImage(im, nQuant, nIter)
    for i in quant_img:
        if representation == 1:
            plt.imshow(i, cmap='gray')
        else:
            plt.imshow(i)
        plt.show()



if __name__ == '__main__':

    # check the histE function on low contrast pictures
    # check_hist_EQ("test2.png", 1)
    # check_hist_EQ("test3.jpg", 1)
    #im = imReadAndConvert("test2.png", 1)
    # im = np.around(im*255).astype(np.int)
    # print(im)
    # res = np.cumsum(np.histogram(im, 256, [0,255])[0])
    # plt.plot(res)
    # plt.show()
    # print(im.size)
    check_quantisize_im("test2.png",1, 10, 2)
    # im = np.round(im*255)
    # hist = np.histogram(im, 255, [0,255])[0]
    # res = divide_histogram(3,hist)
    # print(res)
