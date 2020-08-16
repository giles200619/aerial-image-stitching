# -*- coding: utf-8 -*-
"""
@author: giles
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from sklearn.neighbors import NearestNeighbors
import random
import glob


def framecapture(path): 
    vidObj = cv2.VideoCapture(path) 
      # Used as counter variable 
    count = 0
      # checks whether frames were extracted 
    success = 1
    while success: 
        success, image = vidObj.read() 
        cv2.imwrite("D:\\frame\\frame%d.jpg" % count, image) 
        count += 1
    return count-1

def preprocess(im):
    #sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    #
    dst = cv2.filter2D(im, -1, kernel=kernel)
    return dst    

def get_Ab(x1,x2):# x1*H=x2
    #u1 = x1[0][0] v1 = x1[0][1]  u1'=x2[0][0] v1'=x2[0][1]
    A =  np.array([[x1[0][0],x1[0][1],1,0,0,0,-x1[0][0]*x2[0][0],-x1[0][1]*x2[0][0]],
               [0,0,0,x1[0][0],x1[0][1],1,-x1[0][0]*x2[0][1],-x1[0][1]*x2[0][1]],
               [x1[1][0],x1[1][1],1,0,0,0,-x1[1][0]*x2[1][0],-x1[1][1]*x2[1][0]],
               [0,0,0,x1[1][0],x1[1][1],1,-x1[1][0]*x2[1][1],-x1[1][1]*x2[1][1]],
               [x1[2][0],x1[2][1],1,0,0,0,-x1[2][0]*x2[2][0],-x1[2][1]*x2[2][0]],
               [0,0,0,x1[2][0],x1[2][1],1,-x1[2][0]*x2[2][1],-x1[2][1]*x2[2][1]],
               [x1[3][0],x1[3][1],1,0,0,0,-x1[3][0]*x2[3][0],-x1[3][1]*x2[3][0]],
               [0,0,0,x1[3][0],x1[3][1],1,-x1[3][0]*x2[3][1],-x1[3][1]*x2[3][1]]])
    
    b =   np.array([[x2[0][0]],[x2[0][1]],[x2[1][0]],[x2[1][1]],[x2[2][0]],[x2[2][1]],[x2[3][0]],[x2[3][1]]])  
    
    return A,b

def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
    # To do
    ##
    #check_list = []
    inlier_list = []
    best_A = np.zeros((8,1))
    for iteration in range(ransac_iter):   
        temp_list = []
        temp_list=random.sample(range(x1.shape[0]), 4)
        temp_list = sorted(temp_list)

        selected_x1 = np.zeros((4,2))
        selected_x2 = np.zeros((4,2))        
        for i in range(4):
            selected_x1[i,:]=x1[temp_list[i],:]
            selected_x2[i,:]=x2[temp_list[i],:]
         #to solveAh=b,define A
        A,b = get_Ab(selected_x1,selected_x2)
        #H = ((A.T*A).I)*(A.T)*b
        H = np.linalg.inv(np.matmul(A.transpose(), A))
        H = np.matmul(H, A.transpose())
        H = np.matmul(H, b)
        H = np.append(H,[1])
        H= np.reshape(H,(3,3))
        inlier = 0
        for j in range(0,x1.shape[0]):
            uv1 = np.array([[x1[j][0]],[x1[j][1]],[1]])
            uv1 = np.matmul(H, uv1)
            uv1[0][0]= uv1[0][0]/uv1[2][0]
            uv1[1][0]= uv1[1][0]/uv1[2][0]
            
            if np.sqrt(math.pow((uv1[0][0]-x2[j][0]),2)+math.pow((uv1[1][0]-x2[j][1]),2)) < ransac_thr:
                inlier+=1
        inlier_list.append(inlier)
        if inlier == max(inlier_list):
            best_A = H
        
    A = best_A
    print(max(inlier_list))
    return A

def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()
    
def get_features(dst1,dst2):
    
    detector = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = detector.detectAndCompute(dst1,None)
    kp2, des2 = detector.detectAndCompute(dst2,None)
      
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(des2)
    distances,matches = neigh.kneighbors(des1,n_neighbors=2, return_distance=True)
    
    x1_des1 = []
    x2_des1 = []
    for i in range(0,matches.shape[0]): # loop through all the matches for des1
        if(distances[i][0]/distances[i][1]<0.5):
            x1_des1.append(kp1[i])
            x2_des1.append(kp2[matches[i][0]])
    #find nearestN for each point in des2?         
    
    
    x1 = np.zeros((len(x1_des1),2))
    x2 = np.zeros((len(x2_des1),2))
    
    for i in range(0,len(x1_des1)):
        (x1[i][0],x1[i][1]) = x1_des1[i].pt
        (x2[i][0],x2[i][1]) = x2_des1[i].pt
    
    return x1,x2
    
def get_new_map_with_direction(x1,x2,img):
    
    vectors_x=[]
    vectors_y=[]
    for i in range(0,x1.shape[0]):
        vectors_x.append(x1[i][0]-x2[i][0])
        vectors_y.append(x1[i][1]-x2[i][1])
        
    if vectors_x[0]>0:
        #int(math.ceil(max(vectors_x)))
        empty=np.zeros((img.shape[0],int(math.ceil(max(vectors_x)))))
        new_map=np.hstack((img,empty))
    else: 
        empty=np.zeros((img.shape[0],abs(int(math.ceil(min(vectors_x))))))
        new_map=np.hstack((empty,img))
    
    if vectors_y[0]>0:
        #int(math.ceil(max(vectors_y))) v = max(vectors_y)
        empty = np.zeros((int(math.ceil(max(vectors_y))),new_map.shape[1]))
        new_map=np.vstack((new_map,empty))
    else: 
        empty = np.zeros((abs(int(math.ceil(max(vectors_y)))),new_map.shape[1]))
        new_map=np.vstack((empty,new_map))
    
    return new_map



def stitch(img1,img2):
    
    img1 = img1.astype(np.uint8)

    dst1 = preprocess(img1)
    dst2 = preprocess(img2)
    x1,x2=get_features(dst1,dst2) 
    
    new_map = get_new_map_with_direction(x1,x2,img1)
    img1 = new_map.astype(np.uint8)
    dst1 = preprocess(img1)
    dst2 = preprocess(img2)
    x1,x2=get_features(dst1,dst2) 
    #visualize_find_match(img1, img2, x1, x2)
    H = align_image_using_feature(x1, x2, 5, 5000)      
    #H_I = np.linalg.inv(H)
    for i in range(0,new_map.shape[0]):
        for j in range(0,new_map.shape[1]):
            if new_map[i][j] == 0:
                uv1 = np.array([[j],[i],[1]])
                uv2 = np.matmul(H, uv1)
                uv2[0][0]= math.floor(uv2[0][0]/uv2[2][0])
                uv2[1][0]= math.floor(uv2[1][0]/uv2[2][0])
                if uv2[0][0]<img2.shape[1] and uv2[1][0]<img2.shape[0]:
                #img_warped[i][j] = img[int(uv2[0][0])][int(uv2[1][0])]
                    new_map[i][j] = img2[int(uv2[1][0])][int(uv2[0][0])]
                    
    return new_map

if __name__=='__main__':
    #%matplotlib qt   
    
    path = './data/*'
    img_list = glob.glob(path)
    result = cv2.imread(img_list[0],cv2.IMREAD_GRAYSCALE)          
    for i in range(1,len(img_list)):
        result = stitch(result,cv2.imread(img_list[i],cv2.IMREAD_GRAYSCALE))    
        
    plt.imshow(result,'gray')
    plt.axis('off')