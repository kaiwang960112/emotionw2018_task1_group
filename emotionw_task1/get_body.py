#!/usr/bin/python
# _*_ coding:utf-8 _*_
import os,sys
from PIL import Image
import cv2
f = open("/home/kwang/openpose/Validation/key_points.txt","r")
f_jilu = open("val_body_jilu.txt", 'w+')
with f as fp:
    for line in fp:
        line = line.strip()
        u1,u2,u3,u4,u5,u6 = line.split(" ",5)
        p1,p2,p3 = u1.split("_",2)
        p4,p5 = p3.split(".",1)
        p=p1+"_"+p2+"."+p5
        m1,m2,m3,m4,m5,m6,m7,m8 = u1.split("/",7)
        m9,m10 = m8.split(".",1)
        #print m8
        im = Image.open(p).convert('L')
        width,height = im.size
        if (float(u4)-float(u2))>0.1*width:
            if (float(u5)-float(u3))>0.15*height:
                w = float(u4)-float(u2)
                h = float(u5)-float(u3)
                scale_w = 0.15*(float(u4)-float(u2))
                scale_h = 0.15*(float(u5)-float(u3))
                if w+2*scale_w <width or h+2*scale_h<height:
                    im1 = im.crop((int(float(u2))-int(scale_w),int(float(u3))-int(scale_h),int(float(u4))+int(scale_w),int(float(u5))+int(scale_h)))
                    path1='/media/sdb/kwang/emotionW-2018/body/Validation/'+m7+'/'
                    print path1+m9
                    if os.path.exists(path1):
                        im1.save(path1+m9+'.jpg')
                        f_jilu.write(path1+m9+'.jpg'+' '+u6+'\n')
                    else:
                        os.makedirs(path1)
                        im1.save(path1+m9+'.jpg')
                        f_jilu.write(path1+m9+'.jpg'+' '+u6+'\n')
                else:
                    im1 = im.crop((int(float(u2)),int(float(u3)),int(float(u4)),int(float(u5))))
                    path1='/media/sdb/kwang/emotionW-2018/body/Train/'+m7+'/'
                    print path1+m9
                    if os.path.exists(path1):
                        im1.save(path1+m9+'.jpg')
                        f_jilu.write(path1+m9+'.jpg'+' '+u6+'\n')
                    else:
                        os.makedirs(path1)
                        im1.save(path1+m9+'.jpg')
                        f_jilu.write(path1+m9+'.jpg'+' '+u6+'\n')
                
        
