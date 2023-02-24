"""

Usage example:
python harris_corner.py --window_size 3 --alpha 0.04 --corner_threshold 10000 hw3_images/butterfly.jpg

"""

import cv2
import numpy as np
import sys
import getopt
import operator
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpim

def getMatrix(matrix, curr_x, curr_y, v_boundry,  h_boundary):
    return matrix[curr_x - v_boundry:curr_x + h_boundary,curr_y - v_boundry: curr_y + h_boundary]

def findGradMagnitude(img):
    Iy, Ix = np.gradient(img)
    sum_of_sqares = np.square(Iy) + np.square(Ix)
    gradient_magnitude = np.sqrt(sum_of_sqares)
    plt.imshow(gradient_magnitude)

def findCorners(img, window_size, k, thresh):
    """
    Finds and returns list of corners and new image with corners drawn
    :param img: The original image
    :param window_size: The size (side length) of the sliding window
    :param k: Harris corner constant. Usually 0.04 - 0.06
    :param thresh: The threshold above which a corner is counted
    :return:
    """
    #Find x and y derivatives
    dy, dx = np.gradient(img)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    height = img.shape[0]
    width = img.shape[1]

    cornerList = []
    newImg = img.copy()
    color_img = cv2.cvtColor(newImg, cv2.COLOR_GRAY2RGB)
    offset = math.floor(window_size/2)

    
    # Loop through image and find our corners
    # and do non-maximum supression
    # this can be also implemented without loop
    
    print("Finding Corners...")
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            X_component = getMatrix(Ixx, y,x, offset, offset+1)
            Y_component = getMatrix(Iyy, y,x, offset, offset+1)
            XY_component = getMatrix(Ixy, y,x, offset, offset+1)

            Mxx = np.sum(X_component)
            Myy = np.sum(Y_component)
            Mxy = np.sum(XY_component)

            determinant = (Mxx * Myy) - (Mxy**2)
            trace = Mxx + Myy
            response = determinant - k * (trace**2)

            if response > thresh:
                cornerList.append([x,y,response]);
                color_img.itemset((y,x, 0), 0)
                color_img.itemset((y,x, 1), 0)
                color_img.itemset((y,x, 2), 255)
    

    return color_img, cornerList

def main():
    """
    Main parses argument list and runs findCorners() on the image
    :return: None
    """
    args, img_name = getopt.getopt(sys.argv[1:], '', ['window_size=', 'alpha=', 'corner_threshold='])
    args = dict(args)
    print(args)
    window_size = args.get('--window_size')
    k = args.get('--alpha')
    thresh = args.get('--corner_threshold')

    print("Image Name: " + str(img_name[0]))
    print("Window Size: " + str(window_size))
    print("K alpha: " + str(k))
    print("Corner Response Threshold:" + thresh)

    img = cv2.imread(img_name[0])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    finalImg, cornerList = findCorners(img, int(window_size), float(k), int(thresh))
        
    points = np.array(cornerList)
    plot = plt.figure(1)
    plt.imshow(img, cmap="gray")
    plt.plot(points[:,0],points[:,1], 'b.')
    plt.show()
         
    if finalImg is not None:
            cv2.imwrite("finalimage1.png", finalImg)

    # findGradMagnitude(img)


if __name__ == "__main__":
    main()