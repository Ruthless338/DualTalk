#-*- coding : utf-8-*-

import numpy as np
import argparse
from scipy.signal import savgol_filter
import os
import torch
from dataset_test import get_loader
from DualTalk import DualTalkModel
import random
# export HF_ENDPOINT=https://hf-mirror.com
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

@torch.no_grad()
def test(args, model, test_loader,epoch):
    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    model.load_state_dict(torch.load(os.path.join(args.save_path, "DualTalk.pth"),map_location=torch.device('cpu')))
    model = model.to(args.device)
    model.eval()
    for file_name, audio1, audio2, exp2, jawpose2, neck2 in test_loader:
        audio1 = audio1.to(args.device)
        audio2 = audio2.to(args.device)
        exp2 = exp2.float().to(args.device)
        jawpose2 = jawpose2.float().to(args.device)
        neck2 = neck2.float().to(args.device)
        blendshape2 = torch.cat((exp2, jawpose2, neck2), dim=2)
        prediction = model(audio1,audio2,blendshape2)
        file_name = file_name[0]
        prediction = prediction.squeeze().detach().cpu().numpy()
        if file_name.endswith('speaker1'):
            file_name = file_name.replace('speaker1', 'speaker2')
        elif file_name.endswith('speaker2'):
            file_name = file_name.replace('speaker2', 'speaker1')
        np.save(os.path.join(result_path, "{}.npy".format( file_name)), prediction)

def main():
    # os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    parser = argparse.ArgumentParser(description='DualTalk')
    parser.add_argument("--test_data_path", type=str, default= "./data/test/", help='path of the test data')
    parser.add_argument("--seed", type=int, default=6666, help='random seed')
    parser.add_argument("--scale",type=str, default="large",help="large or base")
    parser.add_argument("--blendshape_dim", type=int, default=56, help='number of blendshapes:52')
    parser.add_argument("--feature_dim", type=int, default=256, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=25, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--last_train", type=int, default=0, help='last train')
    parser.add_argument("--device", type=str, default="cuda:0", help='cuda:0 or cuda:1')
    parser.add_argument("--save_path", type=str, default="model", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="result_DualTalk", help='path to the predictions')
    parser.add_argument("--max_seq_len", type=int, default=5000, help='max_seq_len')
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    # parser.add_argument("--model_path", type=str, default="/data1/pengziqiao/newdata/pretrained_models/chinese-hubert-large-fairseq-ckpt.pt")

    args = parser.parse_args()
    model = DualTalkModel(args)
    model = model.to(args.device)
    dataset = get_loader(args)
    test(args, model, dataset['test'], args.max_epoch)

if __name__ == "__main__":
    # os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    main()