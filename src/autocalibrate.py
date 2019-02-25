
from collections import Counter
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import chess

def return_length_width(s):
    x=[]
    y=[]
    for c in s:
        x.append(c[0])
        y.append(c[1])
    l=np.max(x)-np.min(x)
    w=np.max(y)-np.min(y)
    return (l,w)


    


def find_squares(img):
    img = cv.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in [cv.split(img)[0]]:
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv.Canny(gray, 0, 50, apertureSize=5)
                bin = cv.dilate(bin, None)
            else:
                _retval, bin = cv.threshold(gray, thrs, 255, cv.THRESH_BINARY)
            contours, _hierarchy = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv.arcLength(cnt, True)
                cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv.contourArea(cnt) > 1000 and cv.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares

def return_valid_indices(a):
    l=[]
    for (i,k) in enumerate(a):
        if(k>0):
            l.append(i)
    return l

def return_valid_centers(squares):
    a=plot_hist(squares)
    idx=return_valid_indices(a)
    m=np.min([a[k] for k in idx])
    mx=np.max([a[k] for k in idx])
    
    ct=[]
    sx=[]
    sy=[]
    for i in idx:
        x=[]
        y=[]
        for c in squares[i]:
            x.append(c[0])
            y.append(c[1])
        cx=((np.min(x)+np.max(x))/2,(np.min(y)+np.max(y))/2)
        ct.append(cx)
        sx.append(cx[0])
        sy.append(cx[1])
    #plt.scatter(sx,sy)
    return ct,sx,sy,m


def plot_hist(squares):
    x=list()
    y=[]
    a=[]
    b=[]
    for s in squares:
        x=return_length_width(s)[0]
        y=return_length_width(s)[1]
        if((y/x<=1.00005) and (y/x>=0.99999)):
            a.append(x)
        else:
            a.append(0)
        b.append(np.abs(y-x))
    
    #plt.hist(a,bins=50)
    #plt.show()
    return a

def plot_hist_img(img):
    img=cv.split(img)[0]
    k=img.flatten()
    plt.hist(k)
    plt.show()
    d=Counter(k)
    w=d[255]
    b=d[0]
    if(w>=b):
        return 1
    else:
        return 0

def plot_scatter(squares):
    x=list()
    y=[]
    for s in squares:
        x.append(return_length_width(s)[0])
        y.append(return_length_width(s)[1])
    plt.scatter(x,y)
    plt.show()
    
    
def plot_overlap_squares(squares,img):
    ct,sx,sy,m=return_valid_centers(squares)
    #img = cv.imread(fname)
    x=np.min(sx)
    y=np.min(sy)
    for i in range(8):
        for j in range(8):
            xn=x+i*m
            yn=y+j*m
            csx=np.round(xn-m/2)
            csy=np.round(yn-m/2)
            cex=np.round(xn+m/2)
            cey=np.round(yn+m/2)
            img=cv.rectangle(img,(int(csx),int(csy)),(int(cex),int(cey)),(0,255,0),3)
    plt.imshow(img)
    cv.imshow('squares', img)
    cv.waitKey(0)
    cv.destroyAllWindows()



def draw_keypoints(vis, keypoints, color = (0, 0, 255)):
    for kp in keypoints:
            x, y = kp.pt
            vis=cv.circle(vis, (int(x), int(y)), 2, color)
    return vis

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

        

def get_square(ix,iy):
    ct,sx,sy,m=return_valid_centers(squares)
    iy=7-iy
    img = cv.imread(fname)
    x=np.min(sx)
    y=np.min(sy)
    xn=x+ix*m
    yn=y+iy*m
    csx=np.round(xn-m/2)
    csy=np.round(yn-m/2)
    cex=np.round(xn+m/2)
    cey=np.round(yn+m/2)
    #img=cv.rectangle(img,(int(csx),int(csy)),(int(cex),int(cey)),(0,255,0),3)
    im=img[int(csy):int(cey),int(csx):int(cex)]
    #im=im[10:-10,10:-10]
    #im[0:30,0:20]=im[10,50]
    
    #cv.imshow('squares', cv.split(im)[0])
    #ch = cv.waitKey(2)
    #cv.destroyAllWindows()
    return im



