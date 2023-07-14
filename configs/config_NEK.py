batch_size=16
num_epochs = 50
logFolder = 'NEK_NoAssignedFolder'

arch = 'vitb'#'r50', 'vitb'
learning_rate = 1e-3
opt = "sgd" #'adam', 'sgd'
use_sched = False
sched_ms = [15, 30, 45]
sched_gm = 0.4
three_layer_frozen = False
two_layer_frozen = False
pretrainedVTN = False
pretrained = True
backbone = 'r50' # 'r18' 'r34' 'r50'
weight_rn50_ssl = '' #"/home/c3-0/ishan/semisup_saved_models/dummy/model_best_e30_loss_2.9230.pth" # '' leave empty to unuse
if (backbone != 'r50'):
  weight_rn50_ssl = ''

cosinelr = True
eventDrop = True
randomcrop = False
dataset = 'NEK'
changing_sr  = False
adv_changing_dt = False
rdCrop_fr  = False
if rdCrop_fr:
  randomcrop = True
# NEK only
trainkit = "p22" 
testkit = "p01"
evAugs = ["rand"]

dvs_imageSize = 128
if arch == 'vitb':
  dvs_imageSize = 224



train_temp_align = True
ECL_weight = 1
ECL = False
num_segments = 5 # for contrastive loss

final_frames = 10
num_steps = 100
skip_rate = 5
assert final_frames*skip_rate <= num_steps



