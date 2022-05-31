#导入工具包
from imutils import contours
import numpy as np
import argparse
import  imutils
import cv2
import myutils
#############
#设置参数
ap = argparse.ArgumentParser()
ap.sdd_argument("-i","--image",required = True,help = "path to input image")
ap.sdd_argument("-t","--template",required = True,help = "path to template OCR-A image")
args = vars(ap.parse_args())
#############
#指定信用卡类型
FIRST_NUMBER = {
    "3":"American Express",
    "4":"Visa",
    "5":"MasterCard",
    "6":"Discover Card"
    }
#绘图展示
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyALLWindows()
#读取一个模板图像
img = cv2.imread(args["template"])
cv_show('img',img)
#灰度图
ref = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv_show('ref',ref)
#二值图像
ref = cv2.threshold(ref,10,255,cv2.THRESH_BINARY_INV)[1]
cv_show('ref',ref)
#############
#计算轮廓
#cv2.findContours()函数接受的参数为二值图，即黑白的（不是灰度图），cv2.RETR_EXTERNAL只检测外轮廓，cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
#返回的list中每个元素都是图像中的一个轮廓
#############
ref_,refCnts,hierarchy = cv2.findContours(ref.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#############
cv2.drawContours(img,refCnts,-1,(0,0,255),3)
cv_show('img',img)
print(np.array(refCnts).shape)
refCnts = myutils.sort_contours(refCnts,method="left-to-right")[0]#排序，从左到右，从上到下
digits = {}

#遍历每一个轮廓
for (i,c) in enumerate(refCnts):
    #计算外接矩形并且resize成合适大小
    (x,y,w,h) = cv2.boundingRect(c)
    roi = ref[y:y + h,x:x + w]
    roi = cv2.resize(roi,(57,88))

    #每一个数字对应每一个模板
    digits[i] = roi

#初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
sqKernel = cv2.getStructuringElement(cv2,MORPH_RECT,(5,5))

#读取输入图像，预处理
imsge = cv2.imread(args["image"])
cv_show('image',image)
image = myutils.resize(image,width=300)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv_show('gray',gray)

#礼帽操作，突出更明亮的区域
tophat = cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,rectKernel)
cv_show('tophat',tophat)
#
gradX = cv2.Sobel(tophat,ddepth=cv2.CV_32F,dx=1,dy=0,ksize=-1)#ksize = -1相当于用3*3的

gradX = np.absolute(gradX)
(minVal,maxVal) = (np.min(gradX),np.max(gradX))
gradX = (255 * ((gradX - minVal)/(maxVal - minVal)))
gradX = gradx.astype("uint8")

print (np.array(gradX).shape)
cv_show('gradX',gradX)

#通过闭操作（先膨胀，再腐蚀）将数字连接在一起
gradX = cv2.morphologyEx(gradX,cv2.MORPH_CLOSE,rectKernel)
cv_show('gradX',gradX)
#THRESH_OTSU会自动找寻合适的阈值，合适双峰，需要把阈值参数设置为0
thresh = cv2.threshold(gradX,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show('thresh',thresh)

#再来一个闭操作
thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,sqKernel)#再来一个闭操作
cv_show('thresh',thresh)

#计算轮廓






    


