import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
import pickle

import json
import math
import cv2
import time
import torchvision.transforms as trans
from fnmatch import fnmatch
from pathlib import Path
from itertools import chain
import sys
from DL.dl_ft_1_train_O_ECL import selectFrames

class ek_test(Dataset):

    def __init__(self, shuffle = False,Test = True, kitchen = 'p01'):
      print(f'into initiliazation function of DL')
      self.shuffle = shuffle # I still need to add the shuffle functionality
      self.Test = Test
      self.all_paths = self.get_path(kitchen)
      if self.shuffle:
            random.shuffle(self.all_paths)
      self.data = self.all_paths
      self.PIL = trans.ToPILImage()
      self.TENSOR = trans.ToTensor()
      self.num_frames = 10 # 10 voxels/clip
      self.num_clips_test = 5
      

    def __len__(self):
        return len(self.data)        

    def __getitem__(self,index):
      #I need one clip at a time i.e. 10 voxels
        clip_1,clip_2,clip_3,clip_4,clip_5, clip_class, vid_path   = self.process_data(index)
        return clip_1,clip_2,clip_3,clip_4,clip_5, clip_class, vid_path

    def get_path(self, kitchen):
      PATH = []
      folders = [kitchen]#, 'p08', 'p22']
      for fol in folders:
        if(self.Test==False):
          root = '/home/ad358172/AY/event_summer/phase_1/N-EPIC-Kitchens/ek_train_test/train/' + fol + '_train/'
        else:
          root = '/home/ad358172/AY/event_summer/phase_1/N-EPIC-Kitchens/ek_train_test/test/' + fol + '_test/'
        for path, subdirs, files in os.walk(root):
            for name in files:
                #if fnmatch(name, pattern):
                PATH.append(path)
        PATH = list(set(PATH))
      PATH.sort()
      return PATH
    
    def process_data(self, idx):
        vid_path = self.data[idx].split(' ')[0]
        clip_1,clip_2,clip_3,clip_4,clip_5, clip_class   = self.build_clip(vid_path)
        return clip_1,clip_2,clip_3,clip_4,clip_5, clip_class, vid_path
      
    def build_clip(self, vid_path):
       clip_class = []
       actions = ['put','take','open','close','wash','cut','mix','pour']
       for id, k in enumerate(actions):
           if(vid_path.find(k)!=-1):
               clip_class = id
               break
       clip_class = np.array(clip_class).repeat(self.num_clips_test)
       os.chdir(vid_path) #now we are into the parent directory e.g. P01_01 containg all npy voxels
       p = Path.cwd()
       
           	################################ frame list maker starts here ###########################
       files = list(p.glob("*.npy*"))
       files.sort() #sorting in ascending order 
       frame_count = len(files)
       frames_dense = selectFrames(frame_count, self.num_frames, self.num_clips_test, False)


       
       #now frame_dense is 5x10 i.e. we would have 5 clips
       clip_1 = [];clip_2 = [];clip_3 = [];clip_4 = [];clip_5 = []
       files = np.array(files)
       frames_dense = np.array(frames_dense)
       files = files[frames_dense]
       
       for iii in files[0]: clip_1.append(self.augmentation(np.load(iii),(224,224)))
       for iii in files[1]: clip_2.append(self.augmentation(np.load(iii),(224,224)))
       for iii in files[2]: clip_3.append(self.augmentation(np.load(iii),(224,224)))
       for iii in files[3]: clip_4.append(self.augmentation(np.load(iii),(224,224)))
       for iii in files[4]: clip_5.append(self.augmentation(np.load(iii),(224,224)))
       
       return clip_1,clip_2,clip_3,clip_4,clip_5, clip_class  


    def augmentation(self, image, resize_size):
      x = np.einsum('ijk->jki',image)
      x = x + np.abs(np.min(x))
      x *= 255/(x.max()) 
      x[x>255] = 255
      x[x<0] = 0
      x = x.astype(np.uint8)
      image = self.PIL(x)
      transform = trans.transforms.Resize(resize_size)
      image = transform(image)
      image = trans.functional.to_tensor(image) #range 0-1
      return image
    
def collate_fn_test(batch):
  clip_1 = [];clip_2 = [];clip_3 = [];clip_4 = [];clip_5 = []
  clip_class = []
  vid_path = []
  for item in batch:
     clip_1.append(torch.stack(item[0],dim=0))
     clip_2.append(torch.stack(item[1],dim=0))
     clip_3.append(torch.stack(item[2],dim=0))
     clip_4.append(torch.stack(item[3],dim=0))
     clip_5.append(torch.stack(item[4],dim=0))
     clip_class.append(torch.as_tensor(np.asarray(item[5])))
     vid_path.append(item[6])
      
  clip_1 = torch.stack(clip_1, dim=0)
  clip_2 = torch.stack(clip_2, dim=0)
  clip_3 = torch.stack(clip_3, dim=0)
  clip_4 = torch.stack(clip_4, dim=0)
  clip_5 = torch.stack(clip_5, dim=0)

  return clip_1,clip_2,clip_3,clip_4,clip_5, clip_class, vid_path

  return clip, clip_class,vid_path
    

def vis_frames(clip,name,path):
  #temp = clip[0,:]
  temp = clip.permute(2,3,1,0)
 
  frame_width = 224
  frame_height = 224
  frame_size = (frame_width,frame_height)
  path = path + name + '.avi'
  video = cv2.VideoWriter(path,cv2.VideoWriter_fourcc('p', 'n', 'g', ' '),3,(frame_size[1],frame_size[0]))
  
  for i in range(temp.shape[3]):
    x = np.array(temp[:,:,:,i])
    x *= 255/(x.max()) 
    x[x>255] = 255
    x[x<0] = 0
    x = x.astype(np.uint8)
    video.write(x) 
  video.release()      
if __name__ == '__main__':
  train_dataset = ek_test(shuffle = True)
  print(f'Train dataset length: {len(train_dataset)}')
  train_dataloader = DataLoader(train_dataset,batch_size=2,  collate_fn=collate_fn_test, drop_last = True)
  print(f'Step involved: {len(train_dataset)/2}')
  t=time.time()
  for i, (clip_1,clip_2,clip_3,clip_4,clip_5, clip_class) in enumerate(train_dataloader):
    print(i)
  
  print(f'Time taken to load data is {time.time()-t}')
