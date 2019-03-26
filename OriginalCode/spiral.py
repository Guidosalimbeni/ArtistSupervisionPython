# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 20:40:06 2018

@author: OWNER
"""

import cv2
import numpy as np

im = cv2.imread('D:\\loadedimages\\002.png')

blackImgae = np.zeros((im.shape[0], im.shape[1], 1), np.uint8)

im = blackImgae


# Golden Rectangle
# Draws a golden reactangle, its decomposition into squares
# and the spiral connecting the corners
# Jerry L. Martin, December 2014

phi = (1 + 5 ** 0.5) / 2 # The golden ratio. phi = 1.61803...

# x = 0 and y= 0 are the top left corner , x=x and y=y are the bottom right corner
y = im.shape[0]
x = im.shape[1]
lenght1rect = int(x * (1/phi))
hight1rect = y
# second rectangle
lenght2rect = x - lenght1rect
height2rect = int(y*(1/phi))
# third rectangle
lenght3rect = lenght2rect # confusion here but it works fxed later TODO clean the code
hight3rect = y - height2rect
leftCorn3rectX = int(int(x*(1 /phi)) + (lenght3rect * ( 1 - (1/phi))))
leftCorn3rectY = int(y*(1/phi))
# fourth rectangle
leftCorn4rectX = lenght1rect
leftCorn4rectY = (x + int(y*(1/phi)))
rigthLowerCorn4rectX = lenght1rect + int((1-(1/phi)) * int(lenght2rect))
rigthLowerCorn4rectY = height2rect + int((y - height2rect) * (1 -(1/phi)))
#be careful I drew the little rect here the 4 rect is the bigger
#correct hight and width of the 4 rect
lenght4rect = int((1 -(1/phi)) * lenght3rect)
hight4rect = int(((1/phi)) * hight3rect)
# fifth rectangle
leftCorn5rectX = lenght1rect
leftCorn5rectY = height2rect
rigthLowerCorn5rectX = lenght1rect + int(lenght4rect * (1/phi))
rigthLowerCorn5rectY = height2rect + int((y - height2rect) * (1 -(1/phi)))
lenght5rect = int((1/phi) * lenght4rect)
hight5rect = y - height2rect - hight4rect
# sixth rectangle
lenght6rect = int((1 - (1/phi)) * lenght4rect)
hight6rect = int((1/phi) * hight5rect)
leftCorn6rectX = lenght1rect + lenght5rect
leftCorn6rectY = height2rect
rigthLowerCorn6rectX = lenght1rect + lenght4rect
rigthLowerCorn6rectY = height2rect + hight6rect
# seventh rectangle
lenght7rect = int((1/phi) * lenght6rect)
hight7rect = int((1-(1/phi)) * hight5rect)
leftCorn7rectX = lenght1rect + lenght5rect + int((1 -(1/phi))*  lenght6rect )
leftCorn7rectY = height2rect + hight6rect
rigthLowerCorn7rectX = lenght1rect + lenght4rect
rigthLowerCorn7rectY = height2rect + hight5rect
# eighth rectangle
lenght8rect = int((1 - (1/phi)) * lenght6rect)
hight8rect = int((1/phi) * hight7rect)
leftCorn8rectX = lenght1rect + lenght5rect
leftCorn8rectY = height2rect + hight6rect + int((1 -(1/phi))*  hight7rect )
rigthLowerCorn8rectX = lenght1rect + lenght5rect + lenght8rect
rigthLowerCorn8rectY = height2rect + hight6rect + hight7rect

drawRectangle = False
if drawRectangle:
# draw rectangle #TODO put if statement in the function
    # first rectangle on the left
    cv2.rectangle(im,(0,0), (int(x*(1 /phi)), y), (255, 255, 255), 1)
    cv2.rectangle(im,( int(x*(1 /phi)), 0  ), (x, int(y*(1/phi))), (255, 255, 255), 1 )
    cv2.rectangle(im,( leftCorn3rectX , leftCorn3rectY  ), (x, y), (255, 255, 255), 1 )
    cv2.rectangle(im,( leftCorn4rectX , leftCorn4rectY  ), (rigthLowerCorn4rectX, rigthLowerCorn4rectY), (255, 255, 255), 1)
    cv2.rectangle(im,( leftCorn5rectX , leftCorn5rectY  ), (rigthLowerCorn5rectX, rigthLowerCorn5rectY), (255, 255, 255), 1)
    cv2.rectangle(im,( leftCorn6rectX , leftCorn6rectY  ), (rigthLowerCorn6rectX, rigthLowerCorn6rectY), (255, 255, 255), 1)
    cv2.rectangle(im,( leftCorn7rectX , leftCorn7rectY  ), (rigthLowerCorn7rectX, rigthLowerCorn7rectY), (255, 255, 255), 1)
    cv2.rectangle(im,( leftCorn8rectX , leftCorn8rectY  ), (rigthLowerCorn8rectX, rigthLowerCorn8rectY), (255, 255, 255), 1)
    # main rectangle
    cv2.rectangle(im,(0,0), (x, y), (255, 255, 255), 2)
    
drawEllipses: True
if drawRectangle:
    #draw ellipse
    #first ellipse
    cv2.ellipse(im,  (int(x *(1/phi)), y) , (int(lenght1rect), int(hight1rect))  , 0,  180, 270, (255, 255, 255), 1)
    # 2 ellipse
    cv2.ellipse(im,  (lenght1rect, height2rect) , (lenght2rect, height2rect)  , 0,  270, 360, (255, 255, 255), 1)
    # 3 ellipse
    cv2.ellipse(im,  ((lenght1rect+lenght4rect), height2rect) , (int(lenght2rect*(1/phi)), hight3rect)  , 0,  0, 90, (255, 255, 255), 1)
    # 4 ellipse
    cv2.ellipse(im,  ((lenght1rect+lenght4rect), (height2rect + hight5rect) ), (lenght4rect, hight4rect)  , 0,  90, 180, (255, 255, 255), 1)
    # 5 ellipse
    cv2.ellipse(im,  ((lenght1rect+lenght5rect), (height2rect + hight5rect) ), (lenght5rect, hight5rect)  , 0,  180, 270, (255, 255, 255), 1)
    # 6 ellipse
    cv2.ellipse(im,  ((lenght1rect+lenght5rect), (height2rect + hight6rect) ), (lenght6rect, hight6rect)  , 0,  270, 360, (255, 255, 255), 1)
    # 7 ellipse
    cv2.ellipse(im,  ((lenght1rect+lenght5rect+lenght8rect), (height2rect + hight6rect) ), (lenght7rect, hight7rect)  , 0,  0, 90, (255, 255, 255), 1)
    # 8 ellipse
    cv2.ellipse(im,  ((lenght1rect+lenght5rect+lenght8rect), (height2rect + hight6rect + int((1-(1/phi)) * hight7rect)  ) ), (lenght8rect, hight8rect)  , 0,  90, 180, (255, 255, 255), 1)

im2 = cv2.flip(im, -1 )
im3 = cv2.flip(im2, 1 )
im4 = cv2.flip(im, 1)


cv2.imshow('img', im)
cv2.imshow('img2', im2)
cv2.imshow('img3', im3)
cv2.imshow('img4', im4)
cv2.waitKey()
cv2.destroyAllWindows()




































'''
#this describes the direction. Every iteration turns 90 degrees
vectors = [(1,1), (-1,1), (-1,-1), (1,-1)]
angles = [270, 0, 90, 180]

box_colors = ["white"]
#uncomment if you like it colorful
#box_colors = ["red", "green", "blue"]

squares = [{"origin":(0,0), "length":1},
           {"origin":(1,1), "length":1, "arc_origin":(0,1)},
           {"origin":(0,2), "length":2, "arc_origin":(0,0)},
           {"origin":(-2,0), "length":3, "arc_origin":(1,0)},
           {"origin":(1,-3), "length":5, "arc_origin":(1,2)},
           {"origin":(6,2), "length":8, "arc_origin":(-2,2)}]

for i, sq in enumerate(squares):
    if True:
        #draw the square
        #doc: http://matplotlib.org/api/artist_api.html#matplotlib.patches.Rectangle
        plt.gca().add_patch(Rectangle(
            sq["origin"], 
            sq["length"]*vectors[i%len(vectors)][0],
            sq["length"]*vectors[i%len(vectors)][1], 
            facecolor=box_colors[i%len(box_colors)]
            ))
    if "arc_origin" in sq:
        #draw the arc
        #doc: http://matplotlib.org/api/artist_api.html#matplotlib.patches.Arc
        plt.gca().add_patch(Arc(
            sq["arc_origin"], #origin for the arc
            sq["length"]*2, #width
            sq["length"]*2, #length
            angle = angles[i%len(angles)], #rotation angle
            theta1 = 0, #start angle
            theta2 = 90, #end angle
            lw = 2
            ))

plt.axis([-3.5, 11, -3.5, 11])          
plt.show()

print(7%4)

# better to use the ellipse and rectangluar rule approximation !! 

'''
