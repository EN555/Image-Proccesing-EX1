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
from ex1_utils import LOAD_GRAY_SCALE
import cv2
import numpy as np

def nothing(x: int):
    pass

def gamma_correction(img: np.ndarray, gamma: float) ->np.ndarray:
    help_arr = np.full(img.shape, gamma)
    new_img = np.power(img , help_arr)
    return new_img

def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    # divide between the type of the image, 1 as GrayScale model image either it's RGB image model
    if rep ==1:
        curr_img= cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        curr_img= cv2.imread(img_path)

    cv2.namedWindow('Gamma_Correction', flags=cv2.WINDOW_NORMAL)
    cv2.imshow('Gamma_Correction', curr_img)
    cv2.createTrackbar('Gamma','Gamma_Correction',0,200,nothing)    # create the bar that response on the gamma size
    while True:
        h = cv2.getTrackbarPos('Gamma', 'Gamma_Correction')
        h=h/100   # the differences need to be in 0.1 intervals
        new_image = gamma_correction(curr_img/255.0, h) # need to normalized the image for the gamma correction
        cv2.imshow('Gamma_Correction', new_image)
        key = cv2.waitKey(1000)  # Wait until user press some key
        if key == 27:  # Esc
            break
        if cv2.getWindowProperty("Gamma_Correction", cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyWindow('Gamma_Correction')



def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)



if __name__ == '__main__':
    main()