# EventTransAct
![Example Image](images/Model.PNG)

## Downloading dataset  
You can find instructions on downloading the 2 datasets used in the paper below  
DVS: https://research.ibm.com/interactive/dvsgesture/  
N-Epic-Kitchens: https://github.com/EgocentricVision/N-EPIC-Kitchens  

## Preparing dataset for DVS  
Clone snntorch repository:  
```
git clone https://github.com/jeshraghian/snntorch/tree/master
```
Go to the snntorch/snntorch/spikevision/spikedata folder and replace dvs_gesture.py by the dvs_gesture.py file found in the /DL folder of this repo  

## Running the code

Finetuning on N-Epic-Kitchens dataset:
```
python train.py -c configs/config_NEK.py
```
Finetuning on DVS dataset:
```
python train.py -c configs/config_DVS_woECL.py
```
Finetuning on DVS dataset with contrastive loss:
```
python train.py -c configs/config_DVS_wECL.py
```
