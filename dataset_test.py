import os

import numpy as np
import torch
import torch.utils.data as data
import torchaudio
from torch.utils.data.dataloader import DataLoader
import random
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor
import librosa
from tqdm import tqdm
class BSDataset(data.Dataset):
    def __init__(self, data, data_type='None'):
        self.data = data
        self.len = len(self.data)
        self.data_type = data_type  # train\test\val

    def __getitem__(self, index):
        file_name = self.data[index]['name']
        audio1 = self.data[index]['audio1']
        audio2 = self.data[index]['audio2']
        exp2 = self.data[index]['exp2']
        jawpose2 = self.data[index]['jawpose2']
        neck2 = self.data[index]['neck2']
        return file_name, audio1, audio2, exp2, jawpose2, neck2

    def __len__(self):
        return self.len


def get_metadata(data_path,scale):
    data = []
    if scale == "large":
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    else:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    for wav_path in tqdm(os.listdir(data_path)):
        if wav_path.endswith('.wav'):
            fileshort_name = wav_path.split('.')[0]
            file_meta_dict = dict()
            wav_path1 = os.path.join(data_path,wav_path)
            if not os.path.exists(wav_path1):
                print("lack of",wav_path1)
                continue
            speech_array, sampling_rate = librosa.load(wav_path1, sr=16000)
            audio1 = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)

            if fileshort_name.endswith('speaker1'):
                fileshort_name = fileshort_name.replace('speaker1', 'speaker2')
            elif fileshort_name.endswith('speaker2'):
                fileshort_name = fileshort_name.replace('speaker2', 'speaker1')
            wav_path2 = os.path.join(data_path,fileshort_name + '.wav')
            if not os.path.exists(wav_path2):
                print("lack of",wav_path2)
                continue
            speech_array, sampling_rate = librosa.load(wav_path2, sr=16000)
            audio2 = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
            npz_disk_path2 = os.path.join(data_path, fileshort_name + '.npz')
            if not os.path.exists(npz_disk_path2):
                print("lack of",npz_disk_path2)
                continue
            flame_parms2 = np.load(npz_disk_path2)

            file_meta_dict["audio1"] = audio1
            file_meta_dict["audio2"] = audio2
            file_meta_dict["name"] = fileshort_name  # torch.float16
            file_meta_dict["exp2"] = torch.from_numpy(flame_parms2['exp'])
            file_meta_dict['jawpose2'] = torch.from_numpy(flame_parms2['pose'][:,3:])
            file_meta_dict['neck2'] = torch.from_numpy(flame_parms2['pose'][:,:3])

            data.append(file_meta_dict)

    return data


def read_data(args):
    print("Loading data...")
    random.seed(args.seed)
    test_meta_list = []
    test_meta_list += get_metadata(args.test_data_path,args.scale)
    print('{} sequences in test set'.format(len(test_meta_list)))
    return  test_meta_list


def get_loader(args):
    dataset = dict()
    test_data = read_data(args)
    test_data = BSDataset(test_data, "test")

    dataset['test'] = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,drop_last = True)

    return dataset
