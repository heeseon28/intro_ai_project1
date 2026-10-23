"""PIL letterbox with bounded, cue-preserving training geometry."""
import math
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
import full_train as ft
FILL=(124,116,104)
class Letterbox:
 def __init__(self,train):self.train=train;self.jitter=T.ColorJitter(.2,.2,.2,.05);self.norm=T.Normalize((.485,.456,.406),(.229,.224,.225))
 def __call__(self,im):
  w,h=im.size;ratio=224/max(w,h);nw=max(1,round(w*ratio));nh=max(1,round(h*ratio));left=(224-nw)//2;top=(224-nh)//2
  canvas=Image.new('RGB',(224,224),FILL);canvas.paste(im.resize((nw,nh),Image.Resampling.BILINEAR),(left,top))
  if self.train:
   if torch.rand(1).item()<.5:canvas=TF.hflip(canvas);left=224-left-nw
   canvas=self.jitter(canvas)
   requested_scale=float(torch.empty(1).uniform_(.9,1.1));scale=min(requested_scale,224/nw,224/nh)
   # Bound translation to keep the entire source image rectangle in the output.
   minx=scale*(left-112)+112;maxx=scale*(left+nw-112)+112
   miny=scale*(top-112)+112;maxy=scale*(top+nh-112)+112
   lowerx=math.ceil(-minx);upperx=math.floor(224-maxx);lowery=math.ceil(-miny);uppery=math.floor(224-maxy)
   tx=int(round(float(torch.empty(1).uniform_(-22.4,22.4))));ty=int(round(float(torch.empty(1).uniform_(-22.4,22.4))))
   tx=max(lowerx,min(upperx,tx));ty=max(lowery,min(uppery,ty))
   assert minx+tx>=-1e-6 and maxx+tx<=224+1e-6 and miny+ty>=-1e-6 and maxy+ty<=224+1e-6
   canvas=TF.affine(canvas,angle=0,translate=[tx,ty],scale=scale,shear=[0.,0.],interpolation=InterpolationMode.BILINEAR,fill=FILL)
  return self.norm(TF.to_tensor(canvas))
ORIGINAL_CARS=ft.Cars
class V2Cars(ORIGINAL_CARS):
 def __init__(self,df,root,train,preprocessing='center',store=None):
  super().__init__(df,root,train);self.store=store;self.relpaths=df.rel_path.tolist()
  if preprocessing=='letterbox':self.tf=Letterbox(train)
  elif preprocessing!='center':raise ValueError(preprocessing)
 def __getitem__(self,i):
  if self.store is None:return super().__getitem__(i)
  return self.tf(self.store.pil(self.relpaths[i])),int(self.labels[i])
