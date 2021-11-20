from torchvision.datasets import VisionDataset
from spatial_transforms import ToTensor
from PIL import Image
from math import ceil
import numpy as np
import random
import os
import sys
import torch
from random import randrange

IMAGE = 0
LABEL = 1
TEST_USER = 'S2'
# directory containing the x-flows frames
FLOW_X_FOLDER = "flow_x_processed"
# directory containing the y-flows frames
FLOW_Y_FOLDER = "flow_y_processed"
# directory containing the rgb frames
FRAME_FOLDER = "processed_frames2"
RGB_FOLDER = 'rgb'
RGB_FILENAME = 'rgb'
MMAPS_FOLDER = 'mmaps'
MMAP_FILENAME = 'map'


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # functions that loads an image as an rgb pil object
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def grey_scale_pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # functions that loads an image as a grey-scale pil object
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')


class GTEA61(VisionDataset):
    # this class inherites from VisionDataset and represents the rgb frames of the dataset
    def __init__(self, root, split='train', seq_len=16, transform=None, target_transform=None,
                 label_map=None, mmaps=False, mmaps_transform=None, static_frames=False):
        super(GTEA61, self).__init__(root, transform=transform, target_transform=target_transform)
        self.datadir = root
        # split indicates whether we should load the train or test split
        self.split = split
        self.get_mmaps = mmaps
        # seq len tells us how many frames for each video we are going to consider
        # frames will be taken uniformly spaced
        self.seq_len = seq_len
        self.label_map = label_map
        self.mmaps = mmaps
        self.mmaps_transform = mmaps_transform
        self.static_frames = static_frames

        if label_map is None:
            # if the label map dictionary is not provided, we are going to build it
            self.label_map = {}
        # videos is a list containing for each video, its path where you can find all its frames
        # whereas mmaps contains the path of the mmaps
        self.videos = []
        if mmaps:
            self.mmaps = []
        # labels[i] contains the class ID of the i-th video
        self.labels = []
        # n_frames[i] contains the number of frames available for i-th video
        self.n_frames = []
        # check if ToTensor is among the transformations
        check_totensor = [isinstance(tr, ToTensor) for tr in self.transform.transforms]
        self.has_to_tensor = True in check_totensor
        if not self.has_to_tensor:
            raise ValueError("you did NOT provide ToTensor as a transformation for rgbs")
        
        if mmaps:
            check_mmaps_totensor = [isinstance(tr, ToTensor) for tr in self.mmaps_transform.transforms]
            self.mmaps_has_to_tensor = True in check_mmaps_totensor
            if not self.mmaps_has_to_tensor:
                raise ValueError("you did NOT provide ToTensor as a transformation for mmaps")

        # we expect datadir to be GTEA61, so we add FRAME_FOLDER to get to the frames
        frame_dir = os.path.join(self.datadir, FRAME_FOLDER)
        users = os.listdir(frame_dir)
        users_tmp = []
        for i in users:
            if i != '.DS_Store':
                users_tmp.append(i)
        users = users_tmp

        print(users)
        if len(users) < 4:
            raise FileNotFoundError("you specified the wrong directory")
        if TEST_USER not in users:
            raise FileNotFoundError("S2 folder not found")
        if self.split == 'test':
            folders = [users[users.index(TEST_USER)]]
        else:
            users.remove(TEST_USER)
            folders = users

        # folders is a list that contains either :
        #   - 1 element -> the path of the folder of the user S2 if split == 'test'
        #   - 3 elements -> the paths of the folders for S1,S3,S4 if split == 'train'

        if label_map is None:
            # now we build the label map; we take folders[0] just to get all class names
            # since it is GUARANTEED that all users have same classes
            classes = sorted(os.listdir(os.path.join(frame_dir, folders[0])))
            classes_tmp = []
            for i in classes:
                if '.DS_Store' not in i :
                    classes_tmp.append(i)
            classes = classes_tmp
            self.label_map = {act: i for i, act in enumerate(classes)}
        for user in sorted(folders):
            if ".DS_Store" in user:
                continue
            user_dir = os.path.join(frame_dir, user)
            # user dir it's gonna be ../GTEA61/processed_frames2/S1 or any other user
            for action in sorted(os.listdir(user_dir)):
                if ".DS_Store" in action:
                    continue
                action_dir = os.path.join(user_dir, action)
                # inside an action dir we can have 1 or more videos
                for element in sorted(os.listdir(action_dir)):
                    if ".DS_Store" in element:
                        continue

                    # we add rgb to the path since there is an additional folder inside S1/1/rgb
                    # before the frames
                    frames = os.path.join(action_dir, element, RGB_FOLDER)
                    if self.get_mmaps:
                        mmap = os.path.join(action_dir, element, MMAPS_FOLDER)
                        self.mmaps.append(mmap)
                    # we append in videos the path
                    self.videos.append(frames)
                    # in labels the label, using the label map
                    self.labels.append(self.label_map[action])
                    # in frames its length in number of frames
                    self.n_frames.append(len(os.listdir(frames)))

    def __getitem__(self, index):
        # firstly we retrieve the video path, label and num of frames
        vid = self.videos[index]
        label = self.labels[index]
        length = self.n_frames[index]
        if self.transform is not None:
            # this is needed to randomize the parameters of the random transformations
            self.transform.randomize_parameters()
            
        if self.mmaps:
            if self.mmaps_transform is not None:
                self.mmaps_transform.randomize_parameters()

        # sort the list of frames since the name is like rgb002.png
        # so we use the last number as an ordering
        frames = np.array(sorted(os.listdir(vid)))
        # now we take seq_len equally spaced frames between 0 and length
        # linspace with the option int will give us the indices to take
        select_indices = np.linspace(0, length, self.seq_len, endpoint=False, dtype=int)
        # we then select the frames using numpy fancy indexing
        # note that the numpy arrays are arrays of strings, containing the file names
        # nevertheless, numpy will work with string arrays as well
        select_frames = frames[select_indices]
        # append to each file its path
        select_files = [os.path.join(vid, frame) for frame in select_frames]
        # use pil_loader to get pil objects
        sequence = [pil_loader(file) for file in select_files]
        if self.get_mmaps:
            # replace folder
            select_map = [os.path.join(os.path.dirname(file).replace(RGB_FOLDER, MMAPS_FOLDER), os.path.basename(file).replace(RGB_FILENAME, MMAP_FILENAME) ) for file in select_files]
            maps_sequence = [grey_scale_pil_loader(file) for file in select_map]
    
        # Applies preprocessing when accessing the image
        
        if not self.static_frames:
            
            if self.transform is not None:
                sequence = [self.transform(image) for image in sequence]
                # now, if the ToTensor transformation is applied
                # we have in sequence a list of tensor, so we use stack along dimension 0
                # to create a tensor with one more dimension that contains them all
                if self.has_to_tensor:
                    sequence = torch.stack(sequence, 0)

                if self.get_mmaps:
                    maps_sequence = [self.mmaps_transform(mmap) for mmap in maps_sequence]
                    if self.has_to_tensor:
                        maps_sequence = torch.stack(maps_sequence, 0)
                    maps_sequence = maps_sequence.squeeze(1)

                    return sequence, maps_sequence, label

            return sequence, label
        
        else:
            
            random_number = randrange(self.seq_len)
            
            if self.transform is not None:
                sequence = [self.transform(image) for image in sequence]
                # now, if the ToTensor transformation is applied
                # we have in sequence a list of tensor, so we use stack along dimension 0
                # to create a tensor with one more dimension that contains them all
                if self.has_to_tensor:
                    sequence = torch.stack(sequence, 0)
                
                static_sequence = [sequence[random_number] for i in range(self.seq_len)]
                
                if self.has_to_tensor:
                    static_sequence = torch.stack(static_sequence, 0)
                
                if self.get_mmaps:
                    maps_sequence = [self.mmaps_transform(mmap) for mmap in maps_sequence]
                    if self.has_to_tensor:
                        maps_sequence = torch.stack(maps_sequence, 0)
                    maps_sequence = maps_sequence.squeeze(1)

                    return sequence, static_sequence, maps_sequence, label

            return sequence, static_sequence, label
            

    def __len__(self):
        return len(self.videos)


class GTEA61_flow(VisionDataset):
    # this class inherites from VisionDataset and represents the rgb frames of the dataset
    def __init__(self, root, split='train', seq_len=5, transform=None, target_transform=None,
                 label_map=None, n_seq=-1):
        super(GTEA61_flow, self).__init__(root, transform=transform, target_transform=target_transform)
        # we expect datadir to be ../GTEA61
        self.datadir = root
        # split indicates whether we should load the train or test split
        self.split = split
        self.n_seq = n_seq
        # seq len here tells us how many optical frames for each video
        # we are going to consider; note that now
        # frames will be sequential and not uniformly spaced
        self.seq_len = seq_len
        self.label_map = label_map
        if label_map is None:
            # if the label map dictionary is not provided, we are going to build it
            self.label_map = {}
        # x_frames is a list containing for each flow video, its path, where you can find all its frames
        # it will contain the ones under flow_x_processed
        self.x_frames = []
        # y_frames is the same as x_frames, but contains the ones under flow_y_processed
        self.y_frames = []
        # labels[i] contains the class ID of the i-th video
        self.labels = []
        # n_frames[i] contains the number of frames available for i-th video
        self.n_frames = []
        # check if ToTensor is among the transformations
        check_totensor = [isinstance(tr, ToTensor) for tr in self.transform.transforms]
        self.has_to_tensor = True in check_totensor
        if not self.has_to_tensor:
            raise ValueError("you did NOT provide ToTensor as a transformation")

        # we expect datadir to be GTEA61, so we add the flow folder to get to the flow frames
        flow_dir = os.path.join(self.datadir, FLOW_X_FOLDER)
        users = os.listdir(flow_dir)
        if len(users) != 4:
            raise FileNotFoundError("you specified the wrong directory")
        if TEST_USER not in users:
            raise FileNotFoundError("S2 folder not found")
        if self.split == 'test':
            folders = [users[users.index(TEST_USER)]]
        else:
            users.remove(TEST_USER)
            folders = users

        # folders is a list that contains either :
        #   - 1 element -> the path of the folder of the user S2 if split == 'test'
        #   - 3 elements -> the paths of the folders for S1,S3,S4 if split == 'train'

        if label_map is None:
            # now we build the label map; we take folders[0] just to get all class names
            # since it is GUARANTEED that all users have same classes
            classes = os.listdir(os.path.join(flow_dir, folders[0]))
            self.label_map = {act: i for i, act in enumerate(classes)}

        for user in folders:
            # user dir it's gonna be ../GTEA61/flow_x_processed/S1 or any other user
            user_dir = os.path.join(flow_dir, user)
            for action in os.listdir(user_dir):
                # inside an action dir we can have 1 or more videos
                action_dir = os.path.join(user_dir, action)
                for element in os.listdir(action_dir):
                    frames = os.path.join(action_dir, element)
                    # we put in x_frames the path to the folder with all the flow frames
                    self.x_frames.append(frames)
                    # the path for the y_frames is the same as x, except that we replace
                    # flow_x_processed with flow_y_processed in the path
                    # it is GUARANTEED that for each action we have the same number
                    # of x and y frames
                    self.y_frames.append(frames.replace(FLOW_X_FOLDER, FLOW_Y_FOLDER))
                    # put the label in label using the label map dictionary
                    self.labels.append(self.label_map[action])
                    # put here the number of flow frames
                    self.n_frames.append(len(os.listdir(frames)))

    def get_selected_files(self, vid_x, frames_x, frames_y, select_indices):
        # select the frames using numpy fancy indexing
        # note these are arrays of strings, containing the file names
        select_x_frames = frames_x[select_indices]
        select_y_frames = frames_y[select_indices]
        # this will position the elements of select_x_frames and select_y_frames
        # alternatively in a numpy array. remember these file names of the frames
        select_frames = np.ravel(np.column_stack((select_x_frames, select_y_frames)))
        # append to each file the root path. we use the one for  x frames,
        # then replace with y for y frames.x frames are in even positions, y in odd positions
        select_files = [os.path.join(vid_x, frame) for frame in select_frames]
        select_files[1::2] = [y_files.replace('x', 'y') for y_files in select_files[1::2]]
        # create pil objects
        sequence = [grey_scale_pil_loader(file) for file in select_files]
        # Applies preprocessing when accessing the image
        if self.transform is not None:
            # inv=True will create the negative image for x frames
            sequence[::2] = [self.transform(image, inv=True, flow=True) for image in sequence[::2]]
            sequence[1::2] = [self.transform(image, inv=False, flow=True) for image in sequence[1::2]]
            # if the ToTensor transformation is applied
            # 'sequence' is a list of tensors, so we stack along dimension 0 in a single tensor
            # then we apply squeeze along the 1 dimension, because the images are grey-scale,
            # so there is only one channel and we eliminate that dimension
            if self.has_to_tensor:
                sequence = torch.stack(sequence, 0).squeeze(1)
        return sequence

    def __getitem__(self, index):
        # get the paths of the x video, y, label and length
        vid_x = self.x_frames[index]
        vid_y = self.y_frames[index]
        label = self.labels[index]
        length = self.n_frames[index]
        # needed to randomize the parameters of the custom transformations
        self.transform.randomize_parameters()
        # sort list of frames since the name is like flow_x_002.png, last number as ordering
        frames_x = np.array(sorted(os.listdir(vid_x)))
        # do the same for y
        frames_y = np.array(sorted(os.listdir(vid_y)))
        if self.n_seq > 0:
            segments = []
            starting_frames = np.linspace(1, length-self.seq_len+1, self.n_seq, endpoint=False, dtype=int)
            for start_frame in starting_frames:
                select_indices = start_frame + np.arange(0, self.seq_len)
                sequence = self.get_selected_files(vid_x, frames_x, frames_y, select_indices)
                segments.append(sequence)
            segments = torch.stack(segments, 0)

            return segments, label
        else:
            if self.split == 'train':
                # if we are training, we take a random starting frame
                start_frame = random.randint(0, length - self.seq_len)
            else:
                # if we are testing, we take a centered interval
                start_frame = np.ceil((length - self.seq_len) / 2).astype('int')
            # the frames will be sequential, so the select indices are
            # from startFrame to starFrame + seq_len
            select_indices = start_frame + np.arange(0, self.seq_len)
            sequence = self.get_selected_files(vid_x, frames_x, frames_y, select_indices)

            return sequence, label

    def __len__(self):
        return len(self.x_frames)


class GTEA61_2Stream(VisionDataset):
    # this class inherites from VisionDataset and represents both rgb and flow frames of the dataset
    # it does so by wrapping together an instance of GTEA61 for the rgb frames
    # and an instance of GTEA61_flow for the flow frames
    def __init__(self, root, split='train', seq_len=7, stack_size=5, transform=None, target_transform=None):
        super(GTEA61_2Stream, self).__init__(root, transform=transform, target_transform=target_transform)
        # we expect datadir to be ../GTEA61
        self.datadir = root
        # split indicates whether we should load the train or test split
        self.split = split
        # seq len is the number of rgb frames. they will be uniformly spaced
        self.seq_len = seq_len
        # stack size is the number of flow frames. they will be sequential
        self.stack_size = stack_size

        # now we check that we are in the right directory
        frame_dir = os.path.join(self.datadir, FRAME_FOLDER)
        users = os.listdir(frame_dir)
        if len(users) != 4:
            raise FileNotFoundError("you specified the wrong directory")
        if TEST_USER not in users:
            raise FileNotFoundError("S2 folder not found")
        if self.split == 'test':
            folders = [users[users.index(TEST_USER)]]
        else:
            users.remove(TEST_USER)
            folders = users
        # now we build a label map dictionary and we pass it to the instances of GTEA and GTEA_flow
        classes = os.listdir(os.path.join(frame_dir, folders[0]))
        self.label_map = {act: i for i, act in enumerate(classes)}
        # instance the rgb dataset
        self.frame_dataset = GTEA61(self.datadir, split=self.split, seq_len=self.seq_len,
                                    transform=self.transform, label_map=self.label_map)
        # instance the flow dataset
        self.flow_dataset = GTEA61_flow(self.datadir, split=self.split, seq_len=self.stack_size,
                                        transform=self.transform, label_map=self.label_map)

    def __getitem__(self, index):
        # to retrieve an item, we just ask the instances of
        # rgb and flow dataset to do it
        # then we return both the tensors, and the label
        frame_seq, label = self.frame_dataset.__getitem__(index)
        flow_seq, _ = self.flow_dataset.__getitem__(index)
        return flow_seq, frame_seq, label

    def __len__(self):
        return self.frame_dataset.__len__()
