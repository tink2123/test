# encoding=utf-8
import os
from PIL import Image

files = os.listdir('./0/gen')
for file in files:
###   oldfile = './gt/'+f
###   Image.open(oldfile).convert('RGB').save('./gt_rgb/'+f)
###  oldname = './infer_testB/'+f
###  newname = './infer_testB/'+f.replace('fake_','')
###  os.rename(oldname, newname)
#    if file[-2: ] == 'py':
#        continue   #过滤掉改名的.py文件
    name = file.replace('g_A_', 'g/Cycle_Gan_0/build_generator_resnet_9blocks_0/')
    name = file.replace('r', 'build_resnet_block_')
    name = file.replace('_c','/conv2d')
    name = file.replace('_w','/Conv2D_0.conv2d_weights')
    name = file.replace('_b','/Conv2D_0.b_0')
    #new_name = name[20: 30] + name[-4:]   #选择名字中需要保留的部分
    os.rename(file, name)


#g/Cycle_Gan_0/build_generator_resnet_9blocks_1/build_resnet_block_7/conv2d_1/Conv2D_0.b_0
#g_A_r0_c0_w
