# -*- coding: utf-8 -*-


"""
Created on Fri Feb  8 15:17:36 2019

@author: Sai Krishna
"""


# Python 2/3 compatibility
from collections import Counter
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import chess



def create_map_dic():
    d=dict()
    cnt=0;
    for i in range(8):
        for j in range(8):
            d[cnt]=(j,i)
            cnt+=1
    return d
          
            
def draw_square(squares,id,img):
    img = np.zeros(img.shape, np.uint8)
    s=squares[id]
    x=[]
    y=[]
    for c in s:
        x.append(c[0])
        y.append(c[1])
    l=cv.rectangle(img,(np.min(x),np.min(y)),(np.max(x),np.max(y)),(0,255,0),-1)
    plt.imshow(l)

        


def plot_overlap_squares_cust(img,gx,gy,gm):
    
    x=gx
    y=gy
    m=gm
    
    for i in range(8):
        for j in range(8):
            xn=x+i*m
            yn=y+j*m
            csx=np.round(xn)
            csy=np.round(yn)
            cex=np.round(xn+m)
            cey=np.round(yn+m)
            img=cv.rectangle(img,(int(csx),int(csy)),(int(cex),int(cey)),(0,255,0),3)
    plt.imshow(img)
    cv.imshow('squares', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def get_square_cust(ix,iy,img,gx,gy,gm):
    
    iy=7-iy
    x=gx
    y=gy
    m=gm

    
    xn=x+ix*m
    yn=y+iy*m
    csx=np.round(xn)
    csy=np.round(yn)
    cex=np.round(xn+m)
    cey=np.round(yn+m)
    im=img[int(csy):int(cey),int(csx):int(cex)]
    return im



def extract_features(d,img):
    edges = cv.Canny(img,5,10)
    key_points, description = d.detectAndCompute(edges, None)
    if(len(key_points)==0):
        key_points, description = d.detectAndCompute(edges, None)
    return key_points,edges,description

def extract_features_gray(d,img):
    edges = cv.Canny(img,5,10)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    edges=gray
    key_points, description = d.detectAndCompute(edges, None)
    if(len(key_points)==0):
        key_points, description = d.detectAndCompute(edges, None)
    return key_points,edges,description
    


def pre_compute_features(detector,img_list):
    ft=list()
    for img in img_list:
        kp,e,des = extract_features(detector,img)
        ft.append((kp,e,des))
    return ft
        
def compute_image_matches(detector, img1, img2,idx1, nmatches=10):
    """Draw ORB feature matches of the given two images."""
    kp2,e2,des2 = extract_features(detector,img2)
    kp1,e1,des1 = ft[idx1]
    
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    try:
        matches = bf.match(des1, des2)
        matches = sorted(matches, key = lambda x: x.distance) # Sort matches by distance.  Best come first.
        td=get_distance(matches)
    except:
        try:
            td=[1000]*10
            matches=[]
            kp1,e1,des1 = extract_features_gray(detector,img1)
            kp2,e2,des2 = extract_features_gray(detector,img2)
            matches = bf.match(des1, des2)
            td=get_distance(matches)
        except:
            td=[1000]*10
            matches=[]
        
    return matches,td
def draw_image_matches(detector, img1, img2, nmatches=10):
    """Draw ORB feature matches of the given two images."""
    kp1,e1,des1 = extract_features(detector,img1)
    kp2,e2,des2 = extract_features(detector,img2)
    
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x: x.distance) # Sort matches by distance.  Best come first.
    
    img_matches = cv.drawMatches(img1, kp1, img2, kp2, matches[:nmatches], img2, flags=2) # Show top 10 matches
    plt.figure(figsize=(12, 12))
    plt.title(type(detector))
    plt.imshow(img_matches); plt.show()
    
    
    #print(len(matches))
    td=get_distance(matches)
    return matches,td

def compute_match_vector(detector,img_list,ref_img):
    v=[]
    for (idx,img) in enumerate(img_list):
        m,td=compute_image_matches(detector, img, ref_img,idx, nmatches=10)
        v.append(td)
    #print(v)
    return v


def get_distance(m):
    d=0
    for f in m:
        #print(f.distance)
        d+=f.distance
    if (len(m)==0):
        return 1
    else:
        return d/len(m)
    
def draw_all_matches(r):
    detector = cv.ORB_create()
    for img in imgs:
        draw_image_matches(detector,r, img, nmatches=10)

def show_best_match(r):
    detector = cv.ORB_create()
    v=compute_match_vector(detector,imgs,r)
    idx=np.argmin(v)
    draw_image_matches(detector,r, imgs[idx], nmatches=10)

def detect_piece(square_id,img,gx,gy,gm):
    x,y=map_dic[square_id]
    r=get_square_cust(x,y,img,gx,gy,gm)
    c=detect_color_img(r)
    detector = cv.ORB_create()
    #detector = cv.SURF(400)
    if(np.std(cv.split(r)[0])<=30):
        idx=0
    else:
        v=compute_match_vector(detector,imgs,r)
        idx=p_dic[np.argmin(v)+1]
        if(np.argmin(v)+1==5):
            c=1
    
    return idx,c

def detect_color_img(img):
    img=cv.split(img)[0]
    k=img.flatten()
    d=Counter(k)
    w=d[255]
    b=d[0]
    if(w>=b):
        return 1
    else:
        return 0
    
def create_board(img,gx,gy,gm):
    bd=chess.Board()
    bd.clear_board()
    try:
        for i in range(64):
            p,c=detect_piece(i,img,gx,gy,gm)
            if(p!=0):
                bd._set_piece_at(i,p,c)
        return bd
    except:
        print(i)
        return bd

        
CLICK_GAP=0.1
map_dic=create_map_dic()
PIECE_TYPES = [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING] = range(1, 7)
COLOR_TYPES=[BLACK,WHITE,UNKNOWN]=range(0,3)
p_dic={1:PAWN,2:KNIGHT,3:BISHOP,4:ROOK,5:QUEEN,6:KING,7:ROOK,8:BISHOP,9:ROOK,10:QUEEN,11:ROOK,12:KING,13:PAWN}
c_dic={1:WHITE,2:BLACK,3:BLACK,4:WHITE,5:WHITE,6:BLACK}    


imgs=[]
imgs.append(cv.imread('../data/pawn.png'))
imgs.append(cv.imread('../data/knight.png'))
imgs.append(cv.imread('../data/bishop.png'))
imgs.append(cv.imread('../data/rook.png'))
imgs.append(cv.imread('../data/queen.png'))
imgs.append(cv.imread('../data/king.png'))
imgs.append(cv.imread('../data/rook_black.png'))
imgs.append(cv.imread('../data/bishop_white.png'))
imgs.append(cv.imread('../data/rook_w2.png'))
imgs.append(cv.imread('../data/queen_black.png'))
imgs.append(cv.imread('../data/rook_w3.png'))
imgs.append(cv.imread('../data/king_white.png'))
imgs.append(cv.imread('../data/pawn_black.png'))
detector = cv.ORB_create()
ft=pre_compute_features(detector,imgs)
      
        

        
    













