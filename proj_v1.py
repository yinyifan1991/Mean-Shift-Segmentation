import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from pylab import *

image = cv2.imread('Project #2 Image_Butterfly.jpg')

#set window of size hs * hs 
def sWindow(image,x,y,hs):
    [M,N,C] = np.shape(image);
    upStart = x - hs
    bottomEnd = x + hs
    leftStart = y - hs
    rightEnd = y + hs
    if leftStart < 0:
        leftStart = 0
        rightEnd = 2 * hs
    elif rightEnd > N:
        leftStart = N - 2 * hs - 1
        rightEnd = N
    if upStart < 0:
        upStart = 0
        bottomEnd = 2 * hs
    elif bottomEnd > M:
        upStart = M - 2 * hs - 1
        bottomEnd = M
    window = np.zeros((2*hs+1,2*hs+1,3))
    for i in range(upStart,bottomEnd):
        for j in range(leftStart,rightEnd):
            window[i-upStart,j-leftStart,0] = image[i,j,0]
            window[i-upStart,j-leftStart,1] = image[i,j,1]
            window[i-upStart,j-leftStart,2] = image[i,j,2]
    return window

#compute Eucilidean distance between 2 pixels in BGR space
def rdEucDistance(set,b,g,r):
    [A,B,C] = np.shape(set)
    dist = np.zeros((A,B))
    for p in range(A):
        for q in range(B):
           dist[p,q] = math.sqrt(math.pow((set[p,q,0]-b),2) + math.pow((set[p,q,1]-g),2) + math.pow((set[p,q,2]-r),2))
    return dist

#compute the next density point
def computeY(win,rdDis,b,g,r,hr):
    [A,B,C] = np.shape(win)
    t = np.zeros((A*B,3))
    yi = np.zeros((3))
    xi = np.zeros((3))
    yj = np.zeros((3))
    yi[0] = b
    yi[1] = g
    yi[2] = r
    p = 0
    gk = 0
    for i in range(A):
        for j in range(B):
            if (win[i,j,0] != b) or (win[i,j,1] != g) or (win[i,j,2] != r):
                if (rdDis[i,j] < hr):
                    xi[0] = win[i,j,0]
                    xi[1] = win[i,j,1]
                    xi[2] = win[i,j,2]
                    t[p,0] = xi[0] * ker(yi,xi,hr)
                    t[p,1] = xi[1] * ker(yi,xi,hr)
                    t[p,2] = xi[2] * ker(yi,xi,hr)
                    gk += ker(yi,xi,hr)
                    p += 1
    if gk == 0:
        return yi
    else:
        yj[0] = np.sum(t[0:p,0] / gk)
        yj[1] = np.sum(t[0:p,1] / gk)
        yj[2] = np.sum(t[0:p,2] / gk)
        return yj

#compute convergence point by iteration
def convergence(win,rdDis,b,g,r,hr,iterNum):
    e = 3
    y1 = np.zeros((3))
    yj = np.zeros((3))
    y1[0] = b
    y1[1] = g
    y1[2] = r
    yj = computeY(win,rdDis,y1[0],y1[1],y1[2],hr)
    if iterNum >= 8:
        return yj
    elif (math.sqrt(math.pow((yj[0] - y1[0]),2) + math.pow((yj[1] - y1[1]),2) + math.pow((yj[2] - y1[2]),2))) < e:
        return yj
    else:
        y1 = yj
        iterNum += 1
        yj = convergence(win,rdDis,y1[0],y1[1],y1[2],hr,iterNum)
        return yj

#compute Gaussian kernel
def ker(yi,xi,hr):
    g = 0
    yt = math.sqrt(math.pow((yi[0]-xi[0])/hr,2) + math.pow((yi[1]-xi[1])/hr,2) + math.pow((yi[2]-xi[2])/hr,2))
    y = math.exp(yt*(-1/2))
    return y

#discontinuity preserving filtering
def dpf(image,hs,hr):
    iterNum = 0
    [M,N,Z] = np.shape(image)
    im = np.zeros((M,N,Z))
    imt = np.zeros((M,N,Z),dtype=np.uint8)
    y = np.zeros((3))
    for i in range(M):
        for j in range(N):
            for l in range(Z):
                im[i,j,l] = image[i,j,l]
                imt[i,j,l] = image[i,j,l]
    for x in range(M):
        for y in range(N):
            a = x
            b = y
            window = sWindow(image,a,b,hs)
            dis = rdEucDistance(window,im[a,b,0],im[a,b,1],im[a,b,2])
            con = convergence(window,dis,im[a,b,0],im[a,b,1],im[a,b,2],hr,iterNum)
            imt[a,b,0] = np.uint8(con[0])
            imt[a,b,1] = np.uint8(con[1])
            imt[a,b,2] = np.uint8(con[2])
            iterNum = 0
    return imt

#clustering after dpf
def cluster(img,hs,hr):
    [M,N,Z] = np.shape(img)
    pic = np.zeros((M,N,Z), dtype = np.uint8)
    for o in range(M):
        for p in range(N):
            for q in range(Z):
                pic[o,p,q] = image[o,p,q]

    for i in range(hs,M-hs-1,1):
        for j in range(hs,N-hs-1,1):
            b = int(pic[i,j,0])
            g = int(pic[i,j,1])
            r = int(pic[i,j,2])

            for x in range(i-hs,i+hs+1):
                for  y in range(j-hs,j+hs+1):
                    pic_b = int(pic[x,y,0])
                    pic_g = int(pic[x,y,1])
                    pic_r = int(pic[x,y,2])

                    if (math.sqrt(math.pow((pic_b-b),2) + math.pow((pic_g-g),2) + math.pow((pic_r-r),2))) < hr:                
                        pic[x,y,0] = b
                        pic[x,y,1] = g
                        pic[x,y,2] = r
    return pic

#classify pixels
def classify(clust):
    global th
    [M,N,Z] = np.shape(clust)
    ref = np.zeros((M,N,4), dtype = np.uint8)
    ref[0,0,0] = 1
    for x in range(1,M):
        for y in range(1,N):
            if ref[x,y,0] == 0:                
                b = clust[x,y,0]
                g = clust[x,y,1]
                r = clust[x,y,2]
                ref[x,y,1] = b
                ref[x,y,2] = g
                ref[x,y,3] = r
                th += 1
                for i in range(x,M):
                    for j in range(1,N):
                        if ref[i,j,0] == 0:
                            if ((b == clust[i,j,0]) and (g == clust[i,j,1]) and (r == clust[i,j,2])):
                                ref[i,j,0] = th
                                ref[i,j,1] = b
                                ref[i,j,2] = g
                                ref[i,j,3] = r
    return ref

#eliminate small region by melting it to bigger one
def small(cluss,p):
    global th
    ref = classify(cluss)    
    [M,N,Z] = np.shape(ref)
    clus_r = np.zeros((M,N,3),dtype = np.uint8)
    for a in range(M):
        for b in range(N):
            for c in range(3):
                clus_r[a,b,c] = cluss[a,b,c]
    cl = np.zeros((th,4))
    for q in range(th):
        for x in range(M):
            for y in range(N):
                if ref[x,y,0] == (q + 1):
                    cl[q,0] += 1
    for z in range(th):
        for i in range(M):
            for j in range(N):
                if ref[i,j,0] == z+1:
                    cl[z,1] = int(ref[i,j,1])
                    cl[z,2] = int(ref[i,j,2])
                    cl[z,3] = int(ref[i,j,3])
    for zz in range(th):
        if cl[zz,0] < p:
            for ii in range(M):
                for jj in range(N):
                    if ref[ii,jj,0] == zz+1:
                        iii = ii + 1
                        flag = 0
                        while(iii<M and flag==0):
                            if ref[iii,jj,0] != zz+1:
                                bb = ref[iii,jj,1]
                                gg = ref[iii,jj,2]
                                rr = ref[iii,jj,3]
                                flag = 1
                            iii += 1
                        if flag == 0:
                            iii = ii - 1
                            while(iii>=0 and flag==0):
                                if ref[iii,jj,0] != zz+1:
                                    bb = ref[iii,jj,1]
                                    gg = ref[iii,jj,2]
                                    rr = ref[iii,jj,3]
                                    flag = 1
                                iii -= 1
                        clus_r[ii,jj,0] = bb
                        clus_r[ii,jj,1] = gg
                        clus_r[ii,jj,2] = rr
                            
                        
    return clus_r                   
            
#label region borders    
def segment(clus_t):
    [M,N,Z] = np.shape(clus_t)
    cl = np.ones((M,N), dtype = np.uint8) * 255
    for i in range(M-1):
       for j in range(N-1):
           if ((clus_t[i,j,0] != clus_t[i,j+1,0]) and (clus_t[i,j,1] != clus_t[i,j+1,1]) and (clus_t[i,j,2] != clus_t[i,j+1,2])):
               cl[i,j] = 0;
           if ((clus_t[i,j,0] != clus_t[i+1,j,0]) and (clus_t[i,j,1] != clus_t[i+1,j,1]) and (clus_t[i,j,2] != clus_t[i+1,j,2])):
               cl[i,j] = 0; 
    return cl

def segmentColor(clus_t,image):
    [M,N,Z] = np.shape(clus_t)
#    cl = np.ones((M,N,Z), dtype = np.uint8)
    im_seg = np.ones((M,N,Z), dtype = np.uint8)
    for x in range(M):
        for y in range(N):
            for z in range(Z):
#                cl[x,y,z] = clus_t[x,y,z]
                im_seg[x,y,z] = image[x,y,z]
    for i in range(M-1):
       for j in range(N-1):
           if ((clus_t[i,j,0] != clus_t[i,j+1,0]) and (clus_t[i,j,1] != clus_t[i,j+1,1]) and (clus_t[i,j,2] != clus_t[i,j+1,2])):
               im_seg[i,j,0] = 0
               im_seg[i,j,1] = 0
               im_seg[i,j,2] = 0
           if ((clus_t[i,j,0] != clus_t[i+1,j,0]) and (clus_t[i,j,1] != clus_t[i+1,j,1]) and (clus_t[i,j,2] != clus_t[i+1,j,2])):
               im_seg[i,j,0] = 0
               im_seg[i,j,1] = 0
               im_seg[i,j,2] = 0 
    return im_seg
th = 0
#important parameters: hs, hr and p
hs = 8
hr = 50
p = 200

im = dpf(image,hs,hr)
np.uint8(im)

[M,N,Z] = np.shape(im)
im4c = np.zeros((M,N,Z))
for i in range(M):
        for j in range(N):
            for l in range(Z):
                im4c[i,j,l] = im[i,j,l]
clus = cluster(im4c,hs,hr)

np.uint8(clus)

#seg1 = segment(clus)

combination = small(clus,p)
seg1 = segment(combination)
seg2 = segmentColor(combination,image)

image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
im_rgb = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
clus_rgb = cv2.cvtColor(clus,cv2.COLOR_BGR2RGB)
com_rgb = cv2.cvtColor(combination,cv2.COLOR_BGR2RGB)
seg_rgb = cv2.cvtColor(seg2,cv2.COLOR_BGR2RGB)

plt.subplot(231), 
plt.imshow(image_rgb)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(232), 
plt.imshow(im_rgb)
plt.title('Discontinuity Preserving'), plt.xticks([]), plt.yticks([])
plt.subplot(233), 
plt.imshow(clus_rgb)
plt.title('Cluster'), plt.xticks([]), plt.yticks([])

plt.subplot(234), 
plt.imshow(com_rgb)
plt.title('Eliminate Small Regions'), plt.xticks([]), plt.yticks([])

plt.subplot(235), 
plt.imshow(seg1, cmap ='gray')
plt.title('Gray-Scale Segmentation'), plt.xticks([]), plt.yticks([])

plt.subplot(236), 
plt.imshow(seg_rgb, cmap ='gray')
plt.title('Color segmentation'), plt.xticks([]), plt.yticks([])

plt.show()