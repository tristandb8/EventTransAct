import torch

from datetime import datetime
import sys
import math

from torch.utils.data import DataLoader
import os
import sys
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from argparse import Namespace
import shutil
from nt_xent_original import *

import argparse
sys.path.insert(0, './vtn/')
from parser_sf import parse_args, load_config


def main(params):
  torch.cuda.empty_cache()
  assert params.final_frames*params.skip_rate <= params.num_steps
  save_model='./saved_models/' + params.logFolder + '/'

  if not os.path.exists(save_model):
    os.makedirs(save_model)
  else:
    idx = 0
    save_model = save_model.replace(params.logFolder, params.logFolder + "_" + str(idx))
    while os.path.exists(save_model):
      logFolderb4 = params.logFolder + "_" + str(idx)
      idx += 1
      logFolder = params.logFolder + "_" + str(idx)
      save_model = save_model.replace(logFolderb4, logFolder)
    os.makedirs(save_model)
  
  logFile = open(save_model + 'logfile.txt', 'a')

  if params.dataset == 'NEK':
    from DL.dl_ft_1_train_O_ECL import ek_train, collate_fn2
    from DL.dl_ft_1_test_O_ECL import ek_test, collate_fn_test

    train_dataset = ek_train(shuffle = True, trainKitchen = 'p01', eventDrop = params.eventDrop, eventAugs = params.evAugs, numClips = params.numClips)
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=4,
                                  collate_fn=collate_fn2, drop_last = True)

  elif params.dataset == 'DVS':
    sys.path.append('snntorch/snntorch')
    from spikevision.spikedata.dvs_gesture import DVSGesture
    
    train_dataset = DVSGesture("/home/tr248228/RP_EvT/October/videoMae/DVS/download", train=True, dt = int(500000/params.num_steps), num_steps=params.num_steps,
                           eventDrop = params.eventDrop, eventAugs = params.evAugs, skip_rate = params.skip_rate, final_frames=params.final_frames,
                           randomcrop = params.randomcrop, numClips = params.numClips, train_temp_align = params.train_temp_align, rdCrop_fr = params.rdCrop_fr,
                           changing_sr = params.changing_sr, adv_changing_dt = params.adv_changing_dt, dvs_imageSize = params.dvs_imageSize)
    
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=False, num_workers=4, drop_last = True)
  print(f'Train dataset length: {len(train_dataset)}')
  logFile.write(f'Train dataset length: {len(train_dataset)}\n')

  if params.ECL:
    from vtn_ECL import VTN
  else:
    from vtn import VTN

  args = Namespace(cfg_file='configs/Kinetics/SLOWFAST_4x16_R50.yaml', init_method='tcp://localhost:9999', num_shards=1, opts=[], shard_id=0)

  if params.arch == 'r50':
    args.cfg_file = 'vtn/eventR50_VTN.yaml'
  if params.arch == 'vitb':
    args.cfg_file = 'vtn/eventVIT_B_VTN.yaml'

  cfg = load_config(args)
  if params.dataset == 'NEK':
    cfg.MODEL.NUM_CLASSES = 8
  if params.dataset == 'DVS':
    cfg.MODEL.NUM_CLASSES = 11
  
  
  if params.arch == 'r50':
    model = VTN(cfg, params.weight_rn50_ssl, params.backbone, params.pretrained).cuda()
  elif params.arch == 'vitb':
    model = VTN(cfg, '', '', True).cuda()
    if params.pretrainedVTN:
      pretrained_kvpair = torch.load('vtn/VTN_VIT_B_KINETICS.pyth')['model_state']
      model_kvpair = model.state_dict()
      for layer_name, weights in pretrained_kvpair.items():
          if 'mlp_head.4' in layer_name or 'temporal_encoder.embeddings.position_ids' in layer_name:# in layer_name or 'temporal_encoder.embeddings.position_embeddings' in layer_name:
              print(f'Skipping {layer_name}')
              logFile.write(f'Skipping {layer_name}\n')
              continue 
          model_kvpair[layer_name] = weights
          model.load_state_dict(model_kvpair, strict=True)
      print('model loaded successfully')
      logFile.write('model loaded successfully\n')



  exclusion_name = []
  if params.three_layer_frozen:
    exclusion_name = ['layer4']
  elif params.two_layer_frozen:
    exclusion_name = ['layer3', 'layer4']
  if len(exclusion_name) > 0:
    for name, par in model.named_parameters():
      if 'backbone' in name:
        # still it will have M learnable params
        if not any([exclusion_name_el in name for exclusion_name_el in exclusion_name]):
            print(f'Freezing {name}')
            logFile.write(f'Freezing {name}')
            par.requires_grad = False
        

  if torch.cuda.device_count()>1:
    print(f'Multiple GPUS found!')
    logFile.write(f'Multiple GPUS found!\n')
    model=nn.DataParallel(model)
    model.cuda()
    
  else:
    print('Only 1 GPU is available')
    logFile.write('Only 1 GPU is available\n')
    model.cuda()

  if params.opt == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
  elif params.opt == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=params.learning_rate)
  else:
    exit()

  if params.cosinelr:
    cosine_lr_array = list(np.linspace(0.01,1, 5)) + [(math.cos(x) + 1)/2 for x in np.linspace(0,math.pi/0.99, params.num_epochs-5)]


  if (params.use_sched):
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params.sched_ms, gamma=params.sched_gm)

  
  
  #num_steps_per_update = 4 # accum gradient
  steps = 0
  if params.dataset == "NEK":
    class_count = [164,679,242,210,119,39,1,113]
    weights = 1 - (torch.tensor(class_count)/1567)
    weights = weights.cuda()
    criterion= torch.nn.CrossEntropyLoss(weight=weights.float()).cuda()
  elif params.dataset == "DVS":
    criterion= torch.nn.CrossEntropyLoss().cuda()
  model.train()
  criterion_intra = NTXentLoss(device = 'cuda', batch_size=params.num_segments, temperature=0.1, use_cosine_similarity = False)
  acc = 0
  bestacc = 0

  for epoch in range(params.num_epochs):
    losses, ce_losses, con_losses = [], [], []
    intra_csl2d_logits_predictions = []
    if params.cosinelr:
      learning_rate2 = cosine_lr_array[epoch]*params.learning_rate
      for param_group in optimizer.param_groups:
        param_group['lr']=learning_rate2
        print(f"Learning rate is: {param_group['lr']}")
        logFile.write(f"Learning rate is: {param_group['lr']}\n")
    for i, data in enumerate(train_dataloader, 0):
      if params.ECL:
        inputs, inputs1, labels, pathBS = data
      else:
        inputs, labels, pathBS = data
      if (i == 0) & (epoch == 0):
        print("inputs.shape", inputs.shape, flush = True)
        logFile.write(f"inputs.shape {inputs.shape} \n")
      optimizer.zero_grad()

      inputs = inputs.permute(0,2,1,3,4) #aug_DL output is [120, 16, 3, 112, 112]], #model expects [8, 3, 16, 112, 112]
      inputs = Variable(inputs.cuda())
      if params.ECL:
        inputs1 = inputs1.permute(0,2,1,3,4)
        inputs1 = Variable(inputs1.cuda())
      labels = torch.as_tensor(labels)
      labels = Variable(labels.cuda())

      frameids1= torch.arange(0, inputs.shape[2],1).to(torch.int).repeat(inputs.shape[0], 1).cuda()
      
      if params.ECL:
        per_frame_logits, twoDrep1 = model([inputs, frameids1])
        _, twoDrep2 = model([inputs1, frameids1])
      else:
        per_frame_logits = model([inputs, frameids1])


      ce_loss = criterion(per_frame_logits,labels.long())
      ce_losses.append(ce_loss.cpu().detach().numpy())

      if params.ECL:
        con_loss = 0
        for ii in range(0, twoDrep1.shape[0], inputs.shape[2]):
          temp1, temp2 = criterion_intra(twoDrep1[ii:ii+inputs.shape[2]:params.num_segments,:], twoDrep2[ii:ii+inputs.shape[2]:params.num_segments,:])

          intra_csl2d_logits_predictions.extend(torch.max(temp2, axis=1).indices.cpu().numpy())

          con_loss += temp1
        con_loss/= (twoDrep1.shape[0]/params.final_frames)
        con_losses.append(con_loss.cpu().detach().numpy())
        loss = ce_loss * params.ECL_weight + con_loss
      else:
        loss = ce_loss
      losses.append(loss.cpu().detach().numpy())
      loss.backward()
      optimizer.step()
      steps += 1
      if (steps+1) % 100 == 0: 
        print('Epoch {} average loss: {:.4f}'.format(epoch,np.mean(losses)), flush = True)
        logFile.write('Epoch {} average loss: {:.4f}\n'.format(epoch,np.mean(losses)))
    if (params.use_sched):
      lr_sched.step()
      if((epoch%20==0) and (epoch > 0)):
        print("optimizer", optimizer)
        logFile.write("optimizer\n" + str(optimizer) + "\n")

    signal = "===============================================\n"
    if params.ECL:
      eoe = "End of epoch " + str(epoch)+  ", ECL : " + str(np.mean(con_losses)) + ", mean loss: " + str(np.mean(losses)) + "\n"
    else:
      eoe = "End of epoch " + str(epoch) + ", mean loss: " + str(np.mean(losses)) + "\n"
    print(signal + eoe + signal)
    logFile.write(signal + eoe + signal)

    if params.ECL:
      intra_csl2d_logits_predictions =  np.asarray(intra_csl2d_logits_predictions)
      intracontrastive2d_acc = ((intra_csl2d_logits_predictions == 0).sum())/len(intra_csl2d_logits_predictions)     
      print(f'intra-2D Contrastive Accuracy at Epoch {epoch} is {intracontrastive2d_acc*100 :0.3f}')
      logFile.write(f'intra-2D Contrastive Accuracy at Epoch {epoch} is {intracontrastive2d_acc*100 :0.3f}\n')
    logFile.flush()

    if(epoch%4==0) or (epoch + 10 > params.num_epochs):
      if (params.dataset == "NEK"):
        acc = validate(model, epoch, logFile, ek_test, collate_fn_test, params.testkit, isTest = True)
        if (epoch%20==0):
          for testk in list(set(["p22", "p08", "p01"]) - set([params.testkit])):
            validate(model,epoch, logFile, ek_test, collate_fn_test, testk, isTest = True)
      elif params.dataset == 'DVS':
        acc = validateDVS(model,epoch, logFile, DVSGesture, params.num_steps, params.final_frames, params.skip_rate, ECL = True, dvs_imageSize = params.dvs_imageSize)
      if acc > bestacc:
        bestacc = acc
        print("BEST!!!!")
        logFile.write("BEST!!!! \n")
          
      model.train()

      torch.save(model.state_dict(), save_model+str(epoch).zfill(6)+'.pt')
      now = datetime.now()
      d8 = now.strftime("%d%m%Y")
      current_time = now.strftime("%H:%M:%S")
      weightStatus = d8 + " | " + current_time + " saving weights to: " + save_model +str(epoch)
      print(weightStatus)
      logFile.write(weightStatus + "\n")
  logFile.write("---------------------file close-------------------------------\n")
  logFile.close()


def validate(model,epoch, logFile, ek_test, collate_fn_test, testKitchen, isTest = True):
  
  if (isTest):
    str1 = "Validation"
  else:
    str1 = "Training"
  print(f"*************************{str1} accuracy at epoch {epoch}********************")
  model.eval()
  batch_size = 1
  test_dataset = ek_test(shuffle = False, Test = isTest, kitchen = testKitchen)
  test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers = 8, shuffle=False,collate_fn=collate_fn_test, drop_last = True)
  print(f'{str1} dataset length: {len(test_dataset)}')
  count = 0
  pred_vid = np.zeros((batch_size,1),dtype=(int))
  
  for i,data in enumerate(test_dataloader, 0):
    clip_1,clip_2,clip_3,clip_4,clip_5,labels,pathBS = data
    frameids = torch.arange(0, clip_1.shape[1],1)
    frameids = frameids.to(torch.int).repeat(clip_1.shape[0], 1).cuda()


    clip_1 = clip_1.permute(0,2,1,3,4)
    clip_2 = clip_2.permute(0,2,1,3,4)
    clip_3 = clip_3.permute(0,2,1,3,4)
    clip_4 = clip_4.permute(0,2,1,3,4)
    clip_5 = clip_5.permute(0,2,1,3,4)
    
    clip_1 = Variable(clip_1.cuda())
    clip_2 = Variable(clip_2.cuda())
    clip_3 = Variable(clip_3.cuda())
    clip_4 = Variable(clip_4.cuda())
    clip_5 = Variable(clip_5.cuda())
    
    labels = [(x.numpy()) for x in labels][0]  

    pred_clip_1 = model([clip_1, frameids])[0].squeeze()
    pred_clip_2 = model([clip_2, frameids])[0].squeeze()
    pred_clip_3 = model([clip_3, frameids])[0].squeeze()
    pred_clip_4 = model([clip_4, frameids])[0].squeeze()
    pred_clip_5 = model([clip_5, frameids])[0].squeeze()
    
    sftmx = torch.nn.Softmax(dim=0)
    pred_clip_1 = sftmx(pred_clip_1)
    pred_clip_2 = sftmx(pred_clip_2)
    pred_clip_3 = sftmx(pred_clip_3)
    pred_clip_4 = sftmx(pred_clip_4)
    pred_clip_5 = sftmx(pred_clip_5)
    idxs_mean = []
    for i in range(len(pred_clip_1)):
      idxs_mean.append(np.mean([pred_clip_1.cpu().detach().numpy()[i], pred_clip_2.cpu().detach().numpy()[i], pred_clip_3.cpu().detach().numpy()[i], pred_clip_4.cpu().detach().numpy()[i], pred_clip_5.cpu().detach().numpy()[i]]))
    pred_vid = idxs_mean.index(max(idxs_mean))

    if(pred_vid==labels[0]):
      count+=1
  
  acc = count/len(test_dataset)*100
  print(str(testKitchen), str1, "accuracy:", acc)
  logFile.write(str(testKitchen) + str1 + " accuracy: " + str(acc) + "\n")
  print(f'*****************************************************************************')
  return acc

def validateDVS(model,epoch, logFile, DVSGesture, num_steps, final_frames, skip_rate, ECL = False, dvs_imageSize = 128, val_cr = True, numClips = 5):
    print(f'*************************Test Accuracy********************')
    print(f'Checking Test Accuracy at epoch {epoch}')
    model.eval()
    bs = 1
    num_steps_test = int(np.floor(num_steps / 5 * 18))
    
    test_set = DVSGesture("/home/tr248228/RP_EvT/October/videoMae/DVS/download", train=False,
                            num_steps=num_steps_test, dt=int(500000/num_steps), final_frames = final_frames,
                            skip_rate = skip_rate, numClips = numClips, isVal = True, dvs_imageSize = dvs_imageSize, val_cr = val_cr)
    test_dataloader = DataLoader(test_set, batch_size=bs, shuffle=True, num_workers=4, drop_last = True)

    count = 0
    for i, data in enumerate(test_dataloader, 0):
        clips, clip_label, pathBS = data
        clipPred = []
        for j in range(len(clips[0])):
            video = clips[:,j]
            video = video.permute(0,2,1,3,4)
            input = Variable(video.cuda()) 
            frameids = torch.arange(0, video.shape[2],1).to(torch.int).repeat(video.shape[0], 1).cuda()
            pred = model([input, frameids])
            if ECL: 
                pred = pred[0]
            pred = pred.squeeze()
            sftmx = torch.nn.Softmax(dim=0)
            pred_clip_1 = sftmx(pred)
            clipPred.append(pred_clip_1[None, :])
        clipPred = torch.cat(clipPred, dim=0)
        idxs_mean = []
        for k in range(len(pred_clip_1)):
            idxs_mean.append(torch.mean(clipPred[:,k]))
        pred_vid = idxs_mean.index(max(idxs_mean))
        if(pred_vid==clip_label[0]):
            count+=1
    acc = count/len(test_set)*100

    print("test accuracy: " + str(acc) + "\n")
    print(f'**************************************************************')
    logFile.write("test accuracy: " + str(acc) + "\n")

    return acc


if __name__ == "__main__":
  import argparse, importlib
  parser = argparse.ArgumentParser(description='Script to finetune VTN w/ or w/o ECL')

  parser.add_argument('-c', '--config', type=str, help='Path to the config file')

  args = parser.parse_args()

  spec = importlib.util.spec_from_file_location('params', args.config)
  params = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(params)

  main(params)

