# DVS Gesture citation: A. Amir, B. Taba, D. Berg, T. Melano, J. McKinstry, C. Di Nolfo, T. Nayak, A. Andreopoulos, G. Garreau, M. Mendoza, J. Kusnitz, M. Debole, S. Esser, T. Delbruck, M. Flickner, and D. Modha, "A Low Power, Fully Event-Based Gesture Recognition System," 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, 2017.
# Dataloader adapted from https://github.com/nmi-lab/torchneuromorphic by Emre Neftci and Clemens Schaefer

import struct
import time
import numpy as np
import h5py
import torch.utils.data
from ..neuromorphic_dataset import NeuromorphicDataset
from ..events_timeslices import *
from .._transforms import *
import os
from tqdm import tqdm
import glob
from .._utils import *
import torchvision.transforms as trans



mapping = {
    0: "Hand Clapping",
    1: "Right Hand Wave",
    2: "Left Hand Wave",
    3: "Right Arm CW",
    4: "Right Arm CCW",
    5: "Left Arm CW",
    6: "Left Arm CCW",
    7: "Arm Roll",
    8: "Air Drums",
    9: "Air Guitar",
    10: "Other",
}


class DVSGesture(NeuromorphicDataset):

    """`DVS Gesture <https://www.research.ibm.com/dvsgesture/>`_ Dataset.

    The data was recorded using a DVS128. The dataset contains 11 hand gestures from 29 subjects under 3 illumination conditions.

    **Number of classes:** 11

    **Number of train samples:**  1176

    **Number of test samples:**  288

    **Dimensions:** ``[num_steps x 2 x 128 x 128]``

    * **num_steps:** time-dimension of event-based footage
    * **2:** number of channels (on-spikes for luminance increasing; off-spikes for luminance decreasing)
    * **128x128:** W x H spatial dimensions of event-based footage

    For further reading, see:

        *A. Amir, B. Taba, D. Berg, T. Melano, J. McKinstry, C. Di Nolfo, T. Nayak, A. Andreopoulos, G. Garreau, M. Mendoza, J. Kusnitz, M. Debole, S. Esser, T. Delbruck, M. Flickner, and D. Modha, "A Low Power, Fully Event-Based Gesture Recognition System," 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, 2017.*




    Example::

        from snntorch.spikevision import spikedata

        train_ds = spikedata.DVSGesture("data/dvsgesture", train=True, num_steps=500, dt=1000)
        test_ds = spikedata.DVSGesture("data/dvsgesture", train=False, num_steps=1800, dt=1000)

        # by default, each time step is integrated over 1ms, or dt=1000 microseconds
        # dt can be changed to integrate events over a varying number of time steps
        # Note that num_steps should be scaled inversely by the same factor

        train_ds = spikedata.DVSGesture("data/dvsgesture", train=True, num_steps=250, dt=2000)
        test_ds = spikedata.DVSGesture("data/dvsgesture", train=False, num_steps=900, dt=2000)


    The dataset can also be manually downloaded, extracted and placed into ``root`` which will allow the dataloader to bypass straight to the generation of a hdf5 file.

    **Direct Download Links:**

        `IBM Box Link <https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794>`_

        `Dropbox Link <https://www.dropbox.com/s/cct5kyilhtsliup/DvsGesture.tar.gz?dl=0>`_


    :param root: Root directory of dataset.
    :type root: string

    :param train: If True, creates dataset from training set of dvsgesture, otherwise test set.
    :type train: bool, optional

    :param transform: A function/transform that takes in a PIL image and returns a transforms version. By default, a pre-defined set of transforms are applied to all samples to convert them into a time-first tensor with correct orientation.
    :type transform: callable, optional

    :param target_transform: A function/transform that takes in the target and transforms it.
    :type target_transform: callable, optional

    :param download_and_create: If True, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
    :type download_and_create: bool, optional

    :param num_steps: Number of time steps, defaults to ``500`` for train set, or ``1800`` for test set
    :type num_steps: int, optional

    :param dt: The number of time stamps integrated in microseconds, defaults to ``1000``
    :type dt: int, optional

    :param ds: Rescaling factor, defaults to ``1``.
    :type ds: int, optional

    :return_meta: Option to return metadata, defaults to ``False``
    :type return_meta: bool, optional

    :time_shuffle: Option to randomize start time of dataset, defaults to ``False``
    :type time_shuffle: bool, optional

    Dataloader adapted from `torchneuromorphic <https://github.com/nmi-lab/torchneuromorphic>`_ originally by Emre Neftci and Clemens Schaefer.

    The dataset is released under a Creative Commons Attribution 4.0 license. All rights remain with the original authors.
    """

    # _resources_url = [['Manually Download dataset here: https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/file/211521748942?sb=/details and place under {0}'.format(directory),None, 'DvsGesture.tar.gz']]

    _resources_url = [
        [
            "https://www.dropbox.com/s/cct5kyilhtsliup/DvsGesture.tar.gz?dl=1",
            None,
            "DvsGesture.tar.gz",
        ]
    ]
    # directory = 'data/dvsgesture/'

    def __init__(
        self,
        root,
        train=True,
        isVal = False,
        transform=None,
        target_transform=None,
        download_and_create=True,
        num_steps=None,
        final_frames=None,
        dt=1000,
        ds=None,
        return_meta=False,
        time_shuffle=False,
        eventDrop=True,
        numClips = 1,
        eventAugs = ['all'],
        train_temp_align = True,
        skip_rate = 1,
        randomcrop = False,
        changing_sr = False,
        rdCrop_fr = False,
        evtDropPol = False,
        adv_changing_dt = False,
        dvs_imageSize = 128,
        val_cr = False,
    ):

        self.n = 0
        self.download_and_create = download_and_create
        self.root = root
        self.train = train
        self.dt = dt
        self.return_meta = return_meta
        self.time_shuffle = time_shuffle
        self.hdf5_name = "dvs_gesture.hdf5"
        self.directory = root.split(self.hdf5_name)[0]
        self.resources_local = [self.directory + "/DvsGesture.tar.gz"]
        self.resources_local_extracted = [self.directory + "/DvsGesture/"]
        self.eventDrop = eventDrop
        self.final_frames = final_frames
        self.numClips = numClips
        self.isVal = isVal
        self.train_temp_align = train_temp_align
        self.skip_rate = skip_rate
        self.randomcrop = randomcrop
        self.changing_sr = changing_sr
        self.rdCrop_fr = rdCrop_fr
        self.evtDropPol = evtDropPol
        self.adv_changing_dt = adv_changing_dt
        self.dvs_imageSize = dvs_imageSize
        self.val_cr = val_cr
        if "all" in eventAugs:
            self.eventAugs = ["val", "rand", "time", "rect", "pol"]
        else:
            self.eventAugs = eventAugs

        

        if ds is None:
            ds = 1
        if isinstance(ds, int):
            ds = [ds, ds]

        size = [3, 128 // ds[0], 128 // ds[1]]  # 128//ds[0], 128//ds[1]

        if num_steps is None:
            if self.train:
                self.num_steps = 500
            else:
                self.num_steps = 1800
        else:
            self.num_steps = num_steps

        if self.adv_changing_dt:
            assert self.final_frames == 16
            assert self.dt == 5000
            self.dt = 1000
            self.num_steps = 500
            self.final_frames = 80

        if transform is None:
            transform = Compose(
                [
                    Downsample(factor=[self.dt, 1, ds[0], ds[1]]),
                    ToCountFrame(T=self.num_steps, size=size),
                    ToTensor(),
                    dvs_permute(),
                ]
            )


        if target_transform is not None:
            target_transform = Compose([Repeat(num_steps), toOneHot(11)])

        super(DVSGesture, self).__init__(
            root=root + "/" + self.hdf5_name,
            transform=transform,
            target_transform_train=target_transform,
        )

        with h5py.File(self.root, "r", swmr=True, libver="latest") as f:
            if train:
                self.n = f["extra"].attrs["Ntrain"]
                self.keys = f["extra"]["train_keys"][()]
            else:
                self.n = f["extra"].attrs["Ntest"]
                self.keys = f["extra"]["test_keys"][()]

    def _download(self):
        isexisting = super(DVSGesture, self)._download()

    def _create_hdf5(self):
        create_events_hdf5(
            self.directory,
            self.resources_local_extracted[0],
            self.directory + "/" + self.hdf5_name,
        )

    def __len__(self):
        return self.n

    def __getitem__(self, key):
        
        # print("self.train", self.train) # True
        # print("self.transform", self.transform) #function
        # print("self.target_transform", self.target_transform) #function
        # print("self.return_meta", self.return_meta) # False
        
        # Important to open and close in getitem to enable num_workers>0
        if self.changing_sr:
            assert self.train == True
            self.skip_rate = np.random.randint(4,6)
            self.num_steps = np.random.randint(90,111)
            self.dt = int(500000 / self.num_steps)
            # frames2skip = 30000/ self.dt
            self.transform = Compose(
                [
                    Downsample(factor=[self.dt, 1, 1, 1]),
                    ToCountFrame(T=self.num_steps, size=[3, 128, 128]),
                    ToTensor(),
                    dvs_permute(),
                ]
            )
            assert self.skip_rate*self.final_frames < self.num_steps
        # if self.adv_changing_dt:
        #     self.dt = np.random.randint(3,11) * 1000
        #     self.num_steps = int(500000/self.dt)
        #     self.skip_rate = np.max(int(5000/self.dt), 1)


        with h5py.File(self.root, "r", swmr=True, libver="latest") as f:
            if not self.train:
                key = key + f["extra"].attrs["Ntrain"]
            assert key in self.keys
            data, target, meta_info_light, meta_info_user = sample(
                f, key, T=self.num_steps, shuffle=self.time_shuffle, train=self.train
            )
        # print(data[0])

        if (self.isVal == False):
            clips = []
            start = np.random.randint(self.num_steps - (self.final_frames - 1)*self.skip_rate)
            lengths = []
            starts = []
            for i in range(self.numClips):
                if self.adv_changing_dt:
                    length = np.random.randint(3,11)
                    start_changing_dt = np.random.randint(0,11-length)
                    lengths.append(length)
                    starts.append(start_changing_dt)
                data1 = data.copy()
                if self.transform is not None:
                    data1 = self.transform(data1)
                    if self.dvs_imageSize != 128:
                        data1 = trans.functional.resize(data1, [self.dvs_imageSize,self.dvs_imageSize])
                    if self.train_temp_align == False:
                        start = np.random.randint(self.num_steps - (self.final_frames - 1)*self.skip_rate)
                    if self.adv_changing_dt:
                        data2 = torch.zeros((16,3,self.dvs_imageSize,self.dvs_imageSize))
                        frames = torch.tensor(list(range(start, start + self.final_frames*self.skip_rate)))
                        frames_f0 = torch.tensor(list(range(0, self.final_frames*self.skip_rate)))
                        frame = 0

                        for j in frames_f0[::25]:
                            frames2concat = frames[j:j+10]
                            frames2concat = frames2concat[start_changing_dt: start_changing_dt + length]
                            for k in frames2concat:
                                data2[frame] += data1[k]
                            frame += 1
                        del frames
                        del frames_f0
                        del frame
                        del frames2concat
                    f_frames = list(range(start+self.final_frames*self.skip_rate + 1))[start:start+self.final_frames*self.skip_rate:self.skip_rate]
                    data1 = data1[f_frames]
                    if self.adv_changing_dt:
                        data1 = data2
                        del data2
                    if np.random.random_sample() > 0.5:
                        data1 = self.eventDropAug(data1, key)
                    if self.randomcrop:
                        data1 = self.resizedCrop(data1)
                    clips.append(data1)
            if self.numClips == 2:
                # print(lengths, starts)
                data, data1 = clips[0], clips[1]
            elif self.numClips == 1:
                data = clips[0]
                    

            if self.target_transform is not None:
                target = self.target_transform(target)

            if self.return_meta is True:
                return data, target, meta_info_light, meta_info_user
            else:
                if (self.numClips > 1):
                    return data, data1, target, key
                else:
                    return data, target, key
        if (self.isVal == True):
            clips = []
            for i in range(self.numClips):
                data1 = data.copy()
                data1 = self.transform(data1)
                if self.dvs_imageSize != 128:
                    data1 = trans.functional.resize(data1, [self.dvs_imageSize,self.dvs_imageSize])
                start = np.random.randint(len(data1) - (self.final_frames - 1)*self.skip_rate)
                f_frames = [item for item in list(range(start,start + (self.final_frames - 1)*self.skip_rate + 1))
                            if item % self.skip_rate == start % self.skip_rate]
                data1 = data1[f_frames]
                if self.val_cr:
                    rd_height = np.random.randint(int(self.dvs_imageSize*.9), self.dvs_imageSize)
                    centcrop = trans.transforms.CenterCrop([rd_height, rd_height])
                    resize = trans.transforms.Resize([224,224])
                    data1 = centcrop(data1)
                    data1 = resize(data1)
                # if np.random.random_sample() > 0.5:
                #     data1 = self.eventDropAug(data1, key)
                clips.append(data1[None, :])
            clips = torch.cat(clips, dim=0)
            # print(clips.shape)
            return clips, target, key

    def resizedCrop(self, data):
        finalHW = self.dvs_imageSize
        xmin = int(np.random.randint(0,13)*(self.dvs_imageSize/128))
        xmax = int(np.random.randint(115,128)*(self.dvs_imageSize/128))
        length = xmax-xmin
        rd = np.random.rand()
        yborder = np.random.randint(self.dvs_imageSize - length)
        for i in range(len(data)):
            if self.rdCrop_fr:
                xmin = int(np.random.randint(0,13)*(self.dvs_imageSize/128))
                xmax = int(np.random.randint(115,128)*(self.dvs_imageSize/128))
                length = xmax-xmin
                rd = np.random.rand()
                yborder = np.random.randint(self.dvs_imageSize - length)
            data[i] = trans.functional.resized_crop(data[i],xmin,yborder,length,length,(finalHW,finalHW))
        return data

    def eventDropAug(self, data, key):
        height = width = self.dvs_imageSize
        finalHW = self.dvs_imageSize
        random_array = np.random.rand(10)
        eventHide = torch.rand((3, finalHW, finalHW))
        ratioHide = np.random.randint(0, 16)/100.00
        timeRatio = 0
        maxTR = 0.35
        threshold = 1
        x_erase = np.random.randint(0,height, size = (2,))
        y_erase = np.random.randint(0,width, size = (2,))
        erase_size1 = np.random.randint(int(height/6),int(height/5), size = (2,))
        erase_size2 = np.random.randint(int(width/6),int(width/5), size = (2,))

        if self.eventDrop:
            for image in data:
                timeRatio += maxTR/self.final_frames
                
                if "val" in self.eventAugs:
                    #erase by value
                    if random_array[1] > 0.8:
                        if not (self.evtDropPol):
                            image[image < threshold] = 0
                        else:
                            rd = np.random.rand()
                            if rd > 1/3:
                                image[image < threshold] = 0
                            elif rd > 2/3:
                                image[0][image[0] < threshold] = 0
                            else:
                                image[1][image[1] < threshold] = 0


                if "rand" in self.eventAugs:
                    #random erase
                    if random_array[3] > 0.8:
                        #random erase not the same for each channel / time
                        eventHide = np.random.random(image.shape)
                        ratioHide = np.random.randint(0, 16)/100.00
                    elif random_array[3] > 0.6:
                        if not (self.evtDropPol):
                            image[(eventHide < ratioHide) & (image != 0)] = 0
                        else:
                            rd = np.random.rand()
                            if rd > 1/3:
                                image[(eventHide < ratioHide) & (image != 0)] = 0
                            elif rd > 2/3:
                                image[0][(eventHide[0] < ratioHide) & (image[0] != 0)] = 0
                            else:
                                image[1][(eventHide[1] < ratioHide) & (image[1] != 0)] = 0
                if "time" in self.eventAugs:
                    #erase with time
                    if (random_array[3] > 0.4) and (random_array[3] < 0.6):
                        if (random_array[8] > 0.5):
                            image[eventHide < timeRatio] = 0
                        else:
                            # reverse order
                            image[eventHide > (1-maxTR) + timeRatio] = 0
                            

                #erase entire rectangles
                if "rect" in self.eventAugs:
                    if random_array[4] > 0.75:
                        if not (self.evtDropPol):
                            image[:, x_erase[0]:x_erase[0] + erase_size1[0],y_erase[0]: y_erase[0] + erase_size2[0]] = 0
                        else:
                            rd = np.random.rand()
                            if rd > 1/3:
                                image[:, x_erase[0]:x_erase[0] + erase_size1[0],y_erase[0]: y_erase[0] + erase_size2[0]] = 0
                            elif rd > 2/3:
                                image[0, x_erase[0]:x_erase[0] + erase_size1[0],y_erase[0]: y_erase[0] + erase_size2[0]] = 0
                            else:
                                image[1, x_erase[0]:x_erase[0] + erase_size1[0],y_erase[0]: y_erase[0] + erase_size2[0]] = 0
                    if random_array[5] > 0.75:
                        if not (self.evtDropPol):
                            image[:, x_erase[1]:x_erase[1] + erase_size1[1],y_erase[1]: y_erase[1] + erase_size2[1]] = 0
                        else:
                            rd = np.random.rand()
                            if rd > 1/3:
                                image[:, x_erase[1]:x_erase[1] + erase_size1[1],y_erase[1]: y_erase[1] + erase_size2[1]] = 0
                            elif rd > 2/3:
                                image[0:, x_erase[1]:x_erase[1] + erase_size1[1],y_erase[1]: y_erase[1] + erase_size2[1]] = 0
                            else:
                                image[1:, x_erase[1]:x_erase[1] + erase_size1[1],y_erase[1]: y_erase[1] + erase_size2[1]] = 0

                if "pol" in self.eventAugs:
                    # erase pos/neg
                    if random_array[6] > 0.75:
                        if random_array[7] > 0.5:
                            image[0,:,:] = 0 # erase pos events
                        else:
                            image[1,:,:] = 0 # erase neg events
                
        return data


def sample(hdf5_file, key, T=500, shuffle=False, train=True):
    if train:
        T_default = 500
    else:
        T_default = 1800
    dset = hdf5_file["data"][str(key)]
    label = dset["labels"][()]
    tbegin = dset["times"][0]
    tend = np.maximum(0, dset["times"][-1] - 2 * T * 1000)
    start_time = np.random.randint(tbegin, tend + 1) if shuffle else 0
    # print(start_time)
    # tmad = get_tmad_slice(dset['times'][()], dset['addrs'][()], start_time, T*1000)
    tmad = get_tmad_slice(
        dset["times"][()], dset["addrs"][()], start_time, T_default * 1000
    )
    tmad[:, 0] -= tmad[0, 0]
    meta = eval(dset.attrs["meta_info"])
    return tmad[:, [0, 3, 1, 2]], label, meta["light condition"], meta["subject"]


def create_events_hdf5(directory, extracted_directory, hdf5_filename):
    fns_train = gather_aedat(directory, extracted_directory, 1, 24)
    fns_test = gather_aedat(directory, extracted_directory, 24, 30)
    test_keys = []
    train_keys = []

    assert len(fns_train) == 98

    with h5py.File(hdf5_filename, "w") as f:
        f.clear()

        key = 0
        metas = []
        data_grp = f.create_group("data")
        extra_grp = f.create_group("extra")
        print("\nCreating dvs_gesture.hdf5...")
        for file_d in tqdm(fns_train + fns_test):
            istrain = file_d in fns_train
            data, labels_starttime = aedat_to_events(file_d)
            tms = data[:, 0]
            ads = data[:, 1:]
            lbls = labels_starttime[:, 0]
            start_tms = labels_starttime[:, 1]
            end_tms = labels_starttime[:, 2]
            out = []

            for i, v in enumerate(lbls):
                if istrain:
                    train_keys.append(key)
                else:
                    test_keys.append(key)
                s_ = get_slice(tms, ads, start_tms[i], end_tms[i])
                times = s_[0]
                addrs = s_[1]
                # subj, light = file_d.replace('\\', '/').split('/')[-1].split('.')[0].split('_')[:2]  # this line throws an error in get_slice, because idx_beg = idx_end --> empty batch
                subj, light = file_d.split("/")[-1].split(".")[0].split("_")[:2]
                metas.append(
                    {
                        "key": str(key),
                        "subject": subj,
                        "light condition": light,
                        "training sample": istrain,
                    }
                )
                subgrp = data_grp.create_group(str(key))
                tm_dset = subgrp.create_dataset("times", data=times, dtype=np.uint32)
                ad_dset = subgrp.create_dataset("addrs", data=addrs, dtype=np.uint8)
                lbl_dset = subgrp.create_dataset(
                    "labels", data=lbls[i] - 1, dtype=np.uint8
                )
                subgrp.attrs["meta_info"] = str(metas[-1])
                assert lbls[i] - 1 in range(11)
                key += 1
        extra_grp.create_dataset("train_keys", data=train_keys)
        extra_grp.create_dataset("test_keys", data=test_keys)
        extra_grp.attrs["N"] = len(train_keys) + len(test_keys)
        extra_grp.attrs["Ntrain"] = len(train_keys)
        extra_grp.attrs["Ntest"] = len(test_keys)

        print("dvs_gesture.hdf5 was created successfully.")


def gather_aedat(
    directory, extracted_directory, start_id, end_id, filename_prefix="user"
):
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            "DVS Gestures Dataset not found, looked at: {}".format(directory)
        )

    fns = []
    for i in range(start_id, end_id):
        search_mask = (
            extracted_directory
            + "/"
            + filename_prefix
            + "{0:02d}".format(i)
            + "*.aedat"
        )
        glob_out = glob.glob(search_mask)
        if len(glob_out) > 0:
            fns += glob_out
    return fns
