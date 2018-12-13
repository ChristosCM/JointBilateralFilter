import numpy as np
import cv2
import sys
import math
import time

#read the images
imgNoFl = cv2.imread("test3a.jpg")
imgWthFl = cv2.imread("test3b.jpg")

def gaussian(sigma):
    pi = math.pi
    e = math.e
    return lambda x: (1 / (sigma * (2 * pi) ** (1 / 2))) * e ** -(x ** 2 / (2 * sigma ** 2))

def distance(point):  
    #calculating the distance between the point and the center (which is 0,0))  
    return ((point[0])**2 + (point[1])**2)**(1/2)

#function that creates the distance mask that is going to be the same for whichever pixel
def distancemask(sigmaDis,size,gdist,diameter):
    distances = np.zeros((diameter,diameter,3))

    for x in range (-size,size+1):
        for y in range (-size,size+1):
            distance_raw = abs(distance((x,y))) # calculate the distance
            distances[size+x][size+y] = gdist(distance_raw) #take the gaussian of the distance
    return distances

 

# i will be x-axis and j will be y-axis
def jointfilter (flash,noflash,sigmaCol, sigmaDis,diameter):
    #getting the half size to make it easier creating and iterating through the masks
    size = diameter//2
    #getting the dimensinos of the image (its going to be the same for both images)
    dimensions = imgNoFl.shape
    height = dimensions[0]
    width = dimensions[1]
    #creating the array of the new image || np.uint8 is the data type for the image
    newimg = np.zeros((height,width,3),np.uint8)
    #gdist and gcol call the function gaussian with the different sigma (gaussian returns lambda)
    gdist = gaussian(sigmaDis)
    gcol = gaussian(sigmaCol)
    distanceMask = distancemask(sigmaDis,size,gdist,diameter) #getting the distance mask from the function above (with the given sigma)
    print ("Starting")
    #parsing and going through the neighbours
    start = time.time()
    #generating the 2 masks, intensityDifMask hold the information for the intensity difference (of the flash photo) and single intensity holds the information for the intensity of the noflash image.
    
    for x in range(height): 
        for y in range(width): #these 2 loops iterate through every pixel in the image

            intensityDifMask = np.zeros((diameter,diameter,3)) 
            singleIntensity = np.zeros((diameter,diameter,3))#generate the different masks, they have to be in the loop because sometimes they need to contain zeros (e.g. when the neighbourhood goes out of bounds)

            for i in range (max(0,x-size),min(height,x+1+size)): # this loop iterates through the neighbourhood given the diameter and checks if its out of bounds with max and min, same as below
                for j in range (max(0,y-size), min(width,y+1+size)):
                        # populate the 2 matrices (masks) by putting the inensity difference and the no flash image intensity 
                        intensityDifMask[i-x+size][j-y+size] = gcol(np.int64(flash[x][y])-np.int64(flash[i][j]))
                        singleIntensity[i-x+size][j-y+size] = noflash[i][j]
                        
            #combined mask is the product of the intensity difference mask and the distance mask  (g1*g2)
            combinedMask = np.multiply(intensityDifMask,distanceMask)
            #kp is the normalisation term
            kp = sum(sum(combinedMask))
            # total sum is Σ in the equation
            totalSum = sum(sum(np.multiply(combinedMask,singleIntensity)))
            # final is just 1/k(p) * Σ that is the pixel of the new image
            final = np.divide(totalSum, kp)
            newimg[x][y] = (final)


    end = time.time()
    print (end-start)
    cv2.imshow("Joint Bilateral Filter", newimg)
    cv2.imwrite("result.jpg",newimg)
    cv2.waitKey()
    cv2.destroyAllWindows()

#parameters: (flashImage, No Flash Image, sigma for Colour , sigma for Distance, diameter of neighbours)
jointfilter(imgWthFl,imgNoFl,10,18,13)