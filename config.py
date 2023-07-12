batch_size=4
num_epochs = 50

cosinelr = True
eventDrop = True
randomcrop = False
dataset = 'DVS'
arch = 'vitb'#'r50', 'vitb'
learning_rate = 1e-4
opt = "adam"
three_layer_frozen = False
two_layer_frozen = False
numClips = 2
train_temp_align = True
changing_sr  = False
adv_changing_dt = False
rdCrop_fr  = False
ECL_weight = 1
pretrainedVTN = False
trainkit = "p22"
testkit = "p01"
if rdCrop_fr:
  randomcrop = True

evAugs = ["rand"]
num_steps = 100
final_frames = 16
skip_rate = 5
num_segments = 4 # for contrastive loss
assert final_frames*skip_rate <= num_steps

pretrained = True
backbone = 'r50' # 'r18' 'r34' 'r50'
weight_rn50_ssl = '' #"/home/c3-0/ishan/semisup_saved_models/dummy/model_best_e30_loss_2.9230.pth" # '' leave empty to unuse
if (backbone != 'r50'):
  weight_rn50_ssl = ''

dvs_imageSize = 128
if arch == 'vitb':
  dvs_imageSize = 224

trainKitchen = 'p01'
testKitchen = 'p01'
logFolder = 'NoAssignedFolder' ## Should be deleted
note = '' # Should be deleted, easily

use_sched = False
sched_ms = [15, 30, 45]
sched_gm = 0.4