import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
#import config as cfg
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
import torchvision
import imageio as iio
import sys

class ek_train(Dataset):

    def __init__(self, shuffle = True, trainKitchen = 'p01', eventDrop = False, eventAugs = ['all'], numClips = 1):
      print(f'into initiliazation function of DL (O)')
      self.shuffle = shuffle # I still need to add the shuffle functionality
      self.all_paths = self.get_path(trainKitchen)
      if self.shuffle:
            random.shuffle(self.all_paths)
      self.data = self.all_paths
      self.PIL = trans.ToPILImage()
      self.TENSOR = trans.ToTensor()
      self.num_frames = 10 # 10 voxels/clip
      self.eventDrop = eventDrop
      self.numClips = numClips
      if "all" in eventAugs:
        self.eventAugs = ["val", "rand", "time", "rect", "pol"]
      else:
        self.eventAugs = eventAugs
      

    def __len__(self):
        return len(self.data)        

    def __getitem__(self,index):
      #I need one clip at a time i.e. 10 voxels
      if self.numClips == 1:
        clip, clip_class,vid_path = self.process_data(index)
        return clip, clip_class,vid_path
      if self.numClips == 2:
        clip, clip1, clip_class,vid_path = self.process_data(index)
        return clip, clip1, clip_class,vid_path

    def get_path(self, trainKitchen):
      PATH = []
      folders = [trainKitchen]#, 'p08', 'p22']
      for fol in folders:
        root = '/home/ad358172/AY/event_summer/phase_1/N-EPIC-Kitchens/ek_train_test/train/' + fol + '_train/'
        #pattern = "*.npy"
        for path, subdirs, files in os.walk(root):
            for name in files:
                #if fnmatch(name, pattern):
                PATH.append(path)
        PATH = list(set(PATH))
      PATH.sort()

      return PATH
    
    def process_data(self, idx):

        vid_path = self.data[idx].split(' ')[0]
        if self.numClips == 1:
          clip, clip_class = self.build_clip(vid_path)
          return clip, clip_class,vid_path
        elif self.numClips == 2:
          clip, clip1, clip_class = self.build_clip(vid_path)
          return clip, clip1, clip_class,vid_path
        #print("vid_path", vid_path, "\nclip_class", actions[clip_class])

        

    def build_clip(self, vid_path):
      clip_class = []
      
      actions = ['put','take','open','close','wash','cut','mix','pour']
      for id, k in enumerate(actions):
          if(vid_path.find(k)!=-1):
              clip_class = id
              break
      os.chdir(vid_path) #now we are into the parent directory e.g. P01_01 containg all npy voxels
      p = Path.cwd()
      
          ################################ frame list maker starts here ###########################
      files = list(p.glob("*.npy*"))
      files.sort() #sorting in ascending order 
      files = np.array(files)
      frame_count = len(files)
      frames_dense = selectFrames(frame_count, self.num_frames, 1, True)
      files_1 = files[frames_dense]
      height = 256
      width = 456
      finalHW = 224
      clips = []
      for nc in range(self.numClips):
        clip = []
        random_array = np.random.rand(10)
        x_erase = np.random.randint(0,finalHW, size = (2,))
        y_erase = np.random.randint(0,finalHW, size = (2,))


        cropping_factor1 = np.random.uniform(0.8, 1) # on an average cropping factor is 80% i.e. covers 64% area
        x0 = np.random.randint(0, width - width*cropping_factor1 + 1) 
        y0 = np.random.randint(0, height - height*cropping_factor1 + 1)

        erase_size1 = np.random.randint(int(height/6),int(height/3), size = (2,))
        erase_size2 = np.random.randint(int(width/6),int(width/3), size = (2,))

        eventHide = np.random.random((finalHW, finalHW))
        eventHide = np.array([eventHide, eventHide, eventHide])
        eventHide = np.einsum('ijk->jki',eventHide)
        ratioHide = np.random.randint(0, 16)/100.00

        timeRatio = 0
        intensityThreshold = np.random.randint(0, 21)/100.00

        #      + vid_path + "\n" + str1)
        for ind, i in enumerate(files_1):
          frame = np.load(i)#frame is the individual voxel
          x = np.einsum('ijk->jki',frame)

          minsaved = np.min(x)
          x = x + np.abs(minsaved)
          maxsaved = x.max()
          shift = int(np.abs(minsaved) * 255/(maxsaved))
          x *= 255/(maxsaved)
          x[x>255] = 255; x[x<0] = 0
          x = x.astype(np.uint8)

          fname = vid_path.rsplit('/', 1)[-1] + "_" + str(ind)
          y= self.augmentation(x,random_array, x_erase, y_erase, cropping_factor1, x0, y0, erase_size1,erase_size2, height, width, finalHW, eventHide, ratioHide, shift, timeRatio, intensityThreshold)

          clip.append(y)
          timeRatio += 0.07
        clips.append(clip)
      if self.numClips == 2:
        return clips[0], clips[1], clip_class
      if self.numClips == 1:
        return clips[0], clip_class



    def augmentation(self, image, random_array, x_erase, y_erase, cropping_factor1, x0, y0, erase_size1,erase_size2, height, width, finalHW, eventHide, ratioHide, shift, timeRatio, intensityThreshold):
      image = self.PIL(image)
      image = trans.functional.resized_crop(image,y0,x0,int(height*cropping_factor1),int(width*cropping_factor1),(finalHW,finalHW))

      if random_array[0] > 0.5:
          image = trans.functional.hflip(image)

      image = np.array(image)

      if (self.eventDrop):
        posThreshold = shift + (255 - shift) * intensityThreshold
        negThreshold = shift * (1- intensityThreshold)
        
        if "val" in self.eventAugs:
          #erase by value
          if random_array[1] > 0.7:
            image[(image < posThreshold) & (image > negThreshold)] = shift

        if "rand" in self.eventAugs:
        #random erase
          if random_array[3] > 0.8:
            #random erase not the same for each channel / time
            eventHide = np.random.random(image.shape)
            ratioHide = np.random.randint(0, 16)/100.00
          if random_array[3] > 0.6:
            image[(eventHide < ratioHide) & (image != shift)] = shift
        if "time" in self.eventAugs:
        #erase with time
          if (random_array[3] > 0.4) and (random_array[3] < 0.6):
            image[eventHide < timeRatio] = shift

        if "rect" in self.eventAugs:
        #erase entire rectangles
          if random_array[4] > 0.6:
            image[x_erase[0]:x_erase[0] + erase_size1[0],y_erase[0]: y_erase[0] + erase_size2[0],:] = shift
          if random_array[5] > 0.6:
            image[x_erase[1]:x_erase[1] + erase_size1[1],y_erase[1]: y_erase[1] + erase_size2[1],:] = shift

        if "pol" in self.eventAugs:
        #erase based on pos/neg
          if random_array[6] > 0.8:
            image[image > shift] = shift
          elif random_array[6] > 0.6:
            image[image < shift] = shift

      image = trans.functional.to_tensor(image)


      return image

def collate_fn2(batch):
  clip = []
  clip1 = []
  clip_class = []
  vid_path = []
  twoClips = False
  for item in batch:
        if not (None in item):
          clip.append(torch.stack(item[0],dim=0)) 
          if (len(item) == 4):
            twoClips = True
            clip1.append(torch.stack(item[1],dim=0))
            clip_class.append(torch.as_tensor(np.asarray(item[2])))
            vid_path.append(item[3])
          else:
            clip_class.append(torch.as_tensor(np.asarray(item[1])))
            vid_path.append(item[2])

      
  clip = torch.stack(clip, dim=0)
  if twoClips:
    clip1 = torch.stack(clip1, dim=0)
    return clip, clip1, clip_class,vid_path
  return clip, clip_class,vid_path
    
def vis_frames(clip,name,path):
  #temp = clip[0,:]
  temp = clip.permute(2,3,1,0)
 
  frame_width = 224
  frame_height = 224
  frame_size = (frame_width,frame_height)
  path = path + '/' +  name + '.avi'
  print(path)
  video = cv2.VideoWriter(path,cv2.VideoWriter_fourcc('p', 'n', 'g', ' '),2,(frame_size[1],frame_size[0]))
  
  for i in range(temp.shape[3]):
    x = np.array(temp[:,:,:,i])
    x *= 255/(x.max()) 
    x[x>255] = 255
    x[x<0] = 0
    x = x.astype(np.uint8)
    #x = np.clip(x, a_min = -0.5, a_max = 0.5)
    video.write(x) 
  video.release()  
        
def find_action(vid_path):
  actions = ['put','take','open','close','wash','cut','mix','pour']
  for id, k in enumerate(actions):
      if(vid_path.find(k)!=-1):
          clip_class = id
          break
  return clip_class

def selectFrames(frame_count, num_frames, num_clips_test, isTrain):
    if(frame_count<num_frames):
        repeat_rate = 2
        for i in range(2,6):
          s_1 = []
          for j in range(frame_count):
              for k in range(i):
                  s_1.append(j)
          if (frame_count*i > num_frames):
            repeat_rate = i
            break
        if (isTrain):
            start = np.random.randint(len(s_1) - num_frames + 1)
            
            frames_dense = np.array(s_1[start:start+num_frames])
        else:
            frames_dense = []
            for j in range(num_clips_test):
                start = np.random.randint((frame_count*repeat_rate-num_frames)/repeat_rate + 1) * repeat_rate
                frames_dense.append(np.array(s_1[start:start+num_frames]))
            frames_dense = np.array(frames_dense)
    else:
        if (isTrain):
            skipRate = int(frame_count/num_frames)#np.random.randint(int(frame_count/num_frames)) + 1
            frames_dense = np.array(np.linspace(0,num_frames-1,num_frames,dtype=int) * skipRate
                            + np.random.randint(frame_count - skipRate * (num_frames-1)))
        else:
            frames_dense = []
            for i in range(num_clips_test):
                skipRate = np.random.randint(int(frame_count/num_frames)) + 1
                #skipRate = int(frame_count/num_frames)
                frames_dense.append(np.linspace(0,num_frames-1,num_frames,dtype=int) * skipRate
                                    + np.random.randint(frame_count - skipRate * (num_frames-1)))
            frames_dense = np.array(frames_dense)
    return frames_dense
    
if __name__ == '__main__':
  actions = ['put','take','open','close','wash','cut','mix','pour']
  train_dataset = ek_train(shuffle = True, trainKitchen = 'p01', eventDrop = False, eventAugs = ['all'], numClips = 2)
  print(f'Train dataset length: {len(train_dataset)}')
  train_dataloader = DataLoader(train_dataset,batch_size=1,shuffle= True,  collate_fn=collate_fn2, drop_last = True)
  t=time.time()
  for i, (clip, clip1, clip_class,vid_path) in enumerate(train_dataloader):
    print(clip.shape)
    print(clip1.shape)
    a1 = find_action(vid_path[0])

    print(i)
  print(f'Time taken to load data is {time.time()-t}')
