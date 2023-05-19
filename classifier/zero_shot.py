import torch
import torch.nn as nn 
import numpy as np 
from tqdm import tqdm

from clip.clip import load, tokenize

class ZeroshotCLIP:
    def __init__(self, clip_model):
        self.clip_model, _ = load(clip_model)
        self.clip_model = self.clip_model.eval()

    def fit(self, data):
        prompts = data['prompts']
        self.text_features = []
        with torch.no_grad():
            for per_cls_prompts in prompts:
                per_cls_prompt_embs = tokenize(per_cls_prompts).cuda()
                text_features = self.clip_model.encode_text(per_cls_prompt_embs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_features = text_features.mean(dim=0)
                text_features = text_features / text_features.norm()
                self.text_features.append(text_features)
        self.text_features = torch.stack(self.text_features, dim=0)
        # print(self.text_features.shape)


    def inference(self,image):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            # logit_scale = self.clip_model.logit_scale.exp()
            logits = image_features @ self.text_features.t() * 100
        return logits.float().softmax(dim=-1)

    def accuracy(self, loader, mean_per_class=False):
        total_count=0
        acc_count=0

        if mean_per_class:
            n_class = self.text_features.shape[0]
            acc_per_class = [0 for _ in range(n_class)]
            count_per_class = [0 for _ in range(n_class)]

        for i, (x, y) in tqdm(enumerate(loader)):
            pred_y = self.inference(x.cuda())
            _, top_labels = pred_y.topk(1, dim=-1)

            if not mean_per_class:
                acc_count += (top_labels.view(-1)==y.cuda()).sum().cpu().numpy()
                total_count += y.shape[0]
            else:
                for c in range(n_class):
                    acc_per_class[c] += ((top_labels.view(-1) == y.cuda()) * (y.cuda()== c)).sum().item()
                    count_per_class[c]+=(y.cuda()==c).sum().item()

        if not mean_per_class:
            acc = acc_count*1.0/total_count
            acc = acc.item()
        else:
            acc = [a*1.0/c for (a, c) in zip(acc_per_class, count_per_class)]
            acc = np.array(acc).mean()

        return acc