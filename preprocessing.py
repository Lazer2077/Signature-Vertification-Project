#coding utf-8
import cv2 as cv
import numpy as np
import os
import tkinter
from tkinter.filedialog import askopenfilename
def d_OpenDir():
    os.system('explorer C:\\Users\\Administrator\\Desktop\\EE301\\lzt')
    print('文件夹已打开')
    
def rotation_normalization(src):
    total_x=0
    total_y=0
    u_sq=0
    v_sq=0
    N=0
    for x in range(src.shape[0]):   # 图片的高
        for y in range(src.shape[1]):
            if src[x,y]!=0:
                total_x+=x
                total_y+=y
                N=N+1
            
    u_bar=total_x/N
    v_bar=total_y/N
    for x in range(src.shape[0]):   # 图片的高
        for y in range(src.shape[1]):
            if src[x,y]!=0:
                u_sq=pow(x-u_bar,2)
                v_sq=pow(y-v_bar,2)
    u_sqbar=u_sq/N
    v_sqbar=v_sq/N
    mu=v_bar*u_bar
    I = np.mat([[u_sqbar, mu], [mu, v_sqbar]])
    print(I)
    eig_value,eig_vector=np.linalg.eig(I)
    print("eigenvalue:"+str(eig_value))
    print("eigenvector:"+ str(eig_vector))
    eig_vector=np.linalg.norm(eig_vector)
    # indice=np.argsort(eig_value)
    # min_eigv=eig_vector[eig_vector[0]]
    # print(min_eigv)

for i in range(10):
    mat_path="C:/Users/Administrator/Desktop/EE301/lzt/Anonymous_{}.png".format(i+1)
    wrt_path="C:/Users/Administrator/Desktop/EE301/preprocess/lzt/Anonymous_{}.png".format(i+1)
    print(mat_path)
    mat=cv.imread(mat_path)
    dst = cv.cvtColor(mat,cv.COLOR_BGR2GRAY)
    dst=cv.blur(dst,(5,5))
    ret,img1=cv.threshold(dst, 127, 255, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    img1 = cv.morphologyEx(img1, cv.MORPH_OPEN, kernel)
    img1 = cv.morphologyEx(img1, cv.MORPH_CLOSE, kernel)
    rotation_normalization(img1)
    cv.namedWindow('img',0)
    #cv.imwrite(wrt_path,img1)
    # cv.imshow('img',img1)
    # cv.waitKey(0)
    
