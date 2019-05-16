import os
from PIL import Image

files = os.listdir('./infer_testB')
for f in files:
###   oldfile = './gt/'+f
###   Image.open(oldfile).convert('RGB').save('./gt_rgb/'+f)
  oldname = './infer_testB/'+f
  newname = './infer_testB/'+f.replace('fake_','')
  os.rename(oldname, newname)
