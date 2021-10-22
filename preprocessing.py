#coding utf-8
import cv2 as cv
import numpy as np
import os
import tkinter
from tkinter.filedialog import askopenfilename
def d_OpenDir():
    os.system('explorer C:\\Users\\Administrator\\Desktop\\EE301\\lzt')
    print('文件夹已打开')
def gamma_adjust(im, gamma=1.0):
    """伽马矫正"""
    return (np.power(im.astype(np.float32)/255, 1/gamma)*255).astype(np.uint8)
def rotation_normalization(src):
    total_x=0
    total_y=0
    u_sq=0
    v_sq=0
    N=0
    rows, cols = src.shape
    for x in range(rows):  
        for y in range(cols):
            if src[x,y]==0:
                total_x+=x
                total_y+=y
                N=N+1  
    u_bar=total_x/N
    v_bar=total_y/N
    img_zeros = np.zeros((rows,cols), np.uint8)
    img_zeros.fill(255)
    print(u_bar)
    print(v_bar)
    for x in range(rows):   
        for y in range(cols):
            if src[x,y]==0:
                new_x=x-u_bar
                new_y=y-v_bar
                u_sq+=pow(new_x,2)
                v_sq+=pow(new_y,2)
                new_x+=(rows+1)/2
                new_y+=(cols+1)/2
                if new_x<rows and new_y<cols:
                    img_zeros[int(new_x),int(new_y)]=0
    u_sqbar=u_sq/N
    v_sqbar=v_sq/N
    mu=v_bar*u_bar
    I = np.mat([[u_sqbar, mu], [mu, v_sqbar]])
    print("MatrixI"+str(I))
    eig_value,eig_vector=np.linalg.eig(I)
    print("eigenvalue:"+str(eig_value))
    print("eigenvector:"+ str(eig_vector))
    return img_zeros
    # for x in range(rows):   
    #     for y in range(cols):
    #         if src[x,y]==0:
    #             new_x=x+125-u_bar
    #             new_y=y+225-v_bar
    #             img_zeros[]
                
   # T=np.float32([[1,0,int(255-v_bar)],[0,1,int(125-u_bar)]])
   # eig_vector=np.linalg.norm(eig_vector)

human=['lzt','lt','yyb','wn','wyx','wzc']
verbs=['/Anonymous','/Alexander','/Elizabeth','/Romanov','/Williams']
for i in range(10):
    '''
    The none-local path
     #mat_path=askopenfilename
    '''

    mat_path="C:/Users/Administrator/Desktop/EE301/"
    wrt_path="C:/Users/Administrator/Desktop/EE301/preprocess/"
    last="_{}.png".format(i+1)

    mat_path= mat_path+human[4]+verbs[0]+last
    wrt_path= wrt_path+human[4]+verbs[0]+last
    print(mat_path)
    mat=cv.imread(mat_path)
    dst = cv.cvtColor(mat,cv.COLOR_BGR2GRAY)
    dst= gamma_adjust(dst,0.7)
    dst=cv.blur(dst,(5,5))
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.morphologyEx(dst, cv.MORPH_OPEN, kernel)
    dst = cv.morphologyEx(dst, cv.MORPH_CLOSE, kernel)
    ret,img1=cv.threshold(dst, 127, 255, cv.THRESH_BINARY)
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # img1 = cv.morphologyEx(img1, cv.MORPH_OPEN, kernel)
    # img1 = cv.morphologyEx(img1, cv.MORPH_CLOSE, kernel)
    img1=rotation_normalization(img1)
    
    #cv.namedWindow('img',0)
    #cv.imwrite(wrt_path,img1)
    cv.imshow('img',img1)
    cv.waitKey(0)
    
