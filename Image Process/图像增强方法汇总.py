import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math

def im_show(img_list,title_list):
    ncols=max(int(len(img_list)**0.5),int(math.ceil(len(img_list)/int(len(img_list)**0.5))))
    nrows=min(int(len(img_list)**0.5),int(math.ceil(len(img_list)/int(len(img_list)**0.5))))
    fig,axes=plt.subplots(nrows,ncols,figsize=(10,8),dpi=100)
    # 当调节为多层图形时，axes变为多层数组
    for item in range(len(img_list)):
        if len(img_list[item].shape) == 2:
            axes[int(item//ncols)][int(item%ncols)].imshow(img_list[item][:,:],cmap='gray')
            axes[int(item//ncols)][int(item%ncols)].set_title(title_list[item])
        else:
            axes[int(item//ncols)][int(item%ncols)].imshow(img_list[item][:,:,::-1])
            axes[int(item//ncols)][int(item%ncols)].set_title(title_list[item])
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.show()

#1.图像的缩放
image=cv.imread(r'img/cat.jpeg')
rows,cols = image.shape[:2]
print(rows,cols)
# imge1=cv.resize(image,dsize=(500,374),interpolation=cv.INTER_AREA)#绝对尺寸是（cols，rows）现指定列，后指定行
imge1=cv.resize(image,dsize=None,fx=0.5,fy=0.5,interpolation=cv.INTER_LINEAR)
M=np.float32([[1,0,100],[0,1,50]])  #定义它的平移矩阵，即其平移矩阵为一个2×3的矩阵
dst= cv.warpAffine(image,M,(cols,rows))

#图像的旋转
#生成旋转矩阵
M=cv.getRotationMatrix2D((cols/2,rows/2),45,1)
rotate=cv.warpAffine(image,M,(cols,rows))

#仿射变换
#创建仿射变换矩阵，这个矩阵是先指定原图中的三点坐标，后给出在变换后的三点坐标，最后用函数求出变换矩阵。
pts1=np.float32([[50,50],[200,50],[50,200]])
pts2=np.float32([[100,100],[200,50],[100,250]])
M=cv.getAffineTransform(pts1,pts2)
affine=cv.warpAffine(image,M,(cols,rows))

#透射变换
#构造透射变换矩阵，找到4个点，我们得到任意三个不共线，获取变换矩阵T
pts1=np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2=np.float32([[100,145],[300,100],[80,290],[310,300]])
T=cv.getPerspectiveTransform(pts1,pts2)  #透射矩阵的是怎么求出来的呢？
penetrance=cv.warpPerspective(image,T,(cols,rows))


# 图像的色彩空间的改变
# 图像灰度化
gray = cv.cvtColor(image,cv.COLOR_RGB2GRAY)
# 图像转HSV变换
hsv = cv.cvtColor(image,cv.COLOR_RGB2HSV)
# 图像转HLS变换
hls = cv.cvtColor(image,cv.COLOR_RGB2HLS)
# 图像转lab变换
lab = cv.cvtColor(image,cv.COLOR_RGB2LAB)
# 图像转XYZ变换
xyz = cv.cvtColor(image,cv.COLOR_RGB2XYZ)
# 图像转YUV变换
yuv = cv.cvtColor(image,cv.COLOR_RGB2YUV)

im_show([image,imge1,dst,rotate,affine,penetrance,gray,hsv,hls,lab,xyz,yuv],
        ["原图","图像缩放","图像平移","图像旋转","仿射变换","透射变换","灰度图","H S V","H L S","l a b","X Y Z","Y U V"])

# 图像的各种拼接数据增强
# 图像进行mixup增强

#图像进行mosaic增强
#图像进行cutmix增强
#图像进行随机擦除增强

# 图像的各种降噪和去干扰方法

# #图像金字塔  cv.pyrup(img)为上采样，cv.pyrDown(img)为下采样 首先将图像进行仿射变换，然后在将图像平移放到下部，最后在拼接图像
# up_img = cv.pyrUp(kids)
# down_img = cv.pyrDown(kids)
# cv.imshow('enlarge',up_img)
# cv.imshow('original',kids)
# cv.imshow('shrink',down_img)
# cv.waitKey(0)
# cv.destroyAllWindows()