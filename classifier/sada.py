import imp
from re import S
import random
from matplotlib.pyplot import xcorr
import torch
import torch.nn as nn
from torch.nn import functional as F

from tqdm import tqdm 
from copy import deepcopy 
import numpy as np

from clip.clip import load, tokenize
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

from .utils import build_bicosine_scheduler
from dataset.augmentation import TransformLoader, get_composed_transform
import time
# from attack_type import attackers
import pdb

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(3,3, 7, padding=3, bias=False)
        self.conv2 = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        out = self.conv1(x)
        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out, _ = torch.max(out, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out],dim=1)
        out = self.conv2(out)
        return self.sigmoid(out)

class PromptLearner(nn.Module):
    def __init__(self, class_names, clip_model, n_ctx=16, n_prompt=32, prompt_bsz=4):
        super().__init__()
        ctx_dim = clip_model.ln_final.weight.shape[0]
        dtype = clip_model.dtype

        n_cls = len(class_names)
        self.dtype = dtype
        ctx_vectors = torch.empty(n_prompt, n_ctx, ctx_dim, dtype=self.dtype).cuda()
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        assert n_prompt % prompt_bsz == 0
        self.n_iter = int(n_prompt/prompt_bsz)

        prompt_prefix =' '.join(['X']* n_ctx)
        prompts = [prompt_prefix + ' ' + name + '.' for name in class_names]

        classnames = [name.replace('_', ' ') for name in class_names]
        self.name_lens = [len(_tokenizer.encode(name)) for name in class_names]

        if n_prompt >1:
            self.pos = [0 for _ in range(n_prompt//4)] + [1 for _ in range(n_prompt//4)] + [2 for _ in range(n_prompt//2)]
        else:
            self.pos = [2 for _ in range(n_prompt)]
        random.shuffle(self.pos)
        self.pos = torch.tensor(self.pos, device='cuda')

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts])
        self.tokenized_prompts = tokenized_prompts
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts.cuda()).type(self.dtype)
        self.register_buffer('token_prefix',embedding[:, :1, :]) # Sos, [n_cls, 1, ctx_dim]
        self.register_buffer('token_suffix',embedding[:,1+n_ctx:,:]) # CLS, EOS, [n_cls, -1, ctx_dim]

        nc_prompts = [prompt_prefix+'.' ]
        nc_tokenized_prompts = torch.cat([tokenize(p) for p in nc_prompts])
        self.nc_tokenized_prompts = nc_tokenized_prompts
        with torch.no_grad():
            embedding = clip_model.token_embedding(nc_tokenized_prompts.cuda()).type(self.dtype)
        self.register_buffer('nc_token_prefix', embedding[:, :1,:]) # SOS, [n_cls, 1, ctx_dim]
        self.register_buffer('nc_token_suffix', embedding[:, 1+n_ctx:,:]) # EOS, [n_cls, -1, ctx_dim]

        self.n_cls= n_cls
        self.n_ctx= n_ctx 
        self.n_prompt =n_prompt 
        self.ctx_dim= ctx_dim 
        self.prompt_bsz = prompt_bsz 
        self.prompt_build_mode = 'end'
        self.iter_idx=0

    def forward(self, infer=False):

        if self.n_iter > 1 and (not infer):

            self.iter_step = self.iter_idx % 2
            if self.iter_idx== 0:
                self.start = self.iter_idx // 2
                self.select_idx = torch.randperm(self.n_prompt,device='cuda')
            batch_idx = self.select_idx[self.iter_idx*self.prompt_bsz:(self.iter_idx+1)*self.prompt_bsz]
            ctx = self.ctx[batch_idx]
            pos = self.pos[batch_idx]

            self.iter_idx += 1
            if self.iter_idx == self.n_iter:
                self.iter_idx = 0
        else:
            ctx= self.ctx
            pos = self.pos

        prompt_size = ctx.shape[0]
        tokenized_prompts = self.tokenized_prompts.unsqueeze(1).repeat(1, prompt_size, 1).view(self.n_cls*prompt_size,-1)

        n_cls = self.n_cls

        ctx_end = ctx[pos==2]
        n_end = ctx_end.shape[0]
        prefix = self.token_prefix.unsqueeze(1).repeat(1, n_end, 1, 1)
        suffix = self.token_suffix.unsqueeze(1).repeat(1, n_end, 1, 1)
        ctx_end = ctx_end.unsqueeze(0).repeat(n_cls,1, 1, 1)
        prompts_end = torch.cat([prefix, ctx_end, suffix], dim=2)

        ctx_middle = ctx[pos==1]
        n_middle = ctx_middle.shape[0]
        prompts_middle =[]
        half_n_ctx = self.n_ctx // 2
        for i in range(n_cls):
            name_len = self.name_lens[i]
            prefix_i = self.token_prefix[i:i+1, :,:].unsqueeze(1).repeat(1, n_middle, 1, 1)
            class_i = self.token_suffix[i:i+1,:name_len, :].unsqueeze(1).repeat(1, n_middle, 1, 1)
            suffix_i = self.token_suffix[i:i+1, name_len:, :].unsqueeze(1).repeat(1, n_middle, 1, 1)
            ctx_i_half1 = ctx_middle[:, :half_n_ctx, :].unsqueeze(0)
            ctx_i_half2 = ctx_middle[:, half_n_ctx:,:].unsqueeze(0)
            prompt = torch.cat([
                prefix_i,#(1, n_middle, 1, dim)
                ctx_i_half1,#(1, n_middle, n_ctx//2, dim)
                class_i,#(1, n_middle, name_len, dim)
                ctx_i_half2,#(1, n_middle, n_ctx//2, dim)
                suffix_i #（1,n_middle,*, dim)
            ],dim=2)
            prompts_middle.append(prompt)
        prompts_middle = torch.cat(prompts_middle, dim=0)

        ctx_front = ctx[pos==0]
        n_front = ctx_front.shape[0]
        prompts_front =[]
        for i in range(self.n_cls):
            name_len = self.name_lens[i]
            prefix_i = self.token_prefix[i:i+1,:,:].unsqueeze(1).repeat(1, n_front, 1, 1)
            class_i = self.token_suffix[i:i+1, :name_len, :].unsqueeze(1).repeat(1, n_front, 1, 1)
            suffix_i = self.token_suffix[i:i+1, name_len:, :].unsqueeze(1).repeat(1, n_front, 1, 1)
            ctx_i = ctx_front.unsqueeze(0)
            prompt = torch.cat([
                prefix_i,#（1,n_front，1, dim)
                class_i, #(1, n_front, name_len, dim)
                ctx_i,#（1,n_front, n_ctx, dim)
                suffix_i#（1,n_front，*，dim)
            ],dim=2)
            prompts_front.append(prompt)
        prompts_front = torch.cat(prompts_front, dim=0)

        prompts = torch.cat([prompts_end,prompts_middle, prompts_front], dim=1).view(prompt_size*n_cls,-1, self.ctx_dim)

        if infer:
            return prompts, tokenized_prompts
        else:
            nc_prompts, nc_tokenized_prompts = self.only_prefix()
            return prompts, tokenized_prompts, nc_prompts, nc_tokenized_prompts

    def only_prefix(self):
        ctx = self.ctx
        prompt_size = ctx.shape[0]
        nc_tokenized_prompts = self.nc_tokenized_prompts.repeat(prompt_size, 1)
        prefix = self.nc_token_prefix.repeat(prompt_size, 1, 1)
        suffix = self.nc_token_suffix.repeat(prompt_size, 1, 1)
        nc_prompts = torch.cat([prefix, ctx, suffix],dim=1)
        return nc_prompts, nc_tokenized_prompts

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, x, tokenized_prompts):
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class CLIP(nn.Module):
    def __init__(self, class_names, clip_model, n_shot, n_ctx=16, n_prompt=32, prompt_bsz=4):
        super().__init__()

        self.n_class = len(class_names)
        self.n_prompt = n_prompt
        n_shot = n_shot
        #text enoder
        self.text_encoder = TextEncoder(clip_model)
        if torch.cuda.device_count() > 1:
            self.text_encoder = nn.DataParallel(self.text_encoder)

        # prompt learner
        ctx_dim = clip_model.ln_final.weight.shape[0]
        dtype = clip_model.dtype
        self.prompt_learner = PromptLearner(class_names, clip_model, n_ctx=n_ctx, n_prompt=n_prompt, prompt_bsz=prompt_bsz)

        #image encoder
        self.image_encoder = clip_model.visual
        if torch.cuda.device_count() >1:
            self.image_encoder = nn.DataParallel(self.image_encoder)
        self.logit_scale = clip_model.logit_scale

        #vision prompt
        self.sa = SpatialAttention().cuda()
        self.mean= dict()
        self.var= dict()
        self.curfeature = dict()
        prototype = torch.empty(self.n_class, 8*n_shot, 1024, dtype=self.dtype).cuda()
        nn.init.normal_(prototype, std=0.02)
        self.prototype = nn.Parameter(prototype)
        self.flag=0
        for n in range(self.n_class):
            self.curfeature[int(n)]=[]


    def forward(self, image, labels=None, test=False,epoch=0,adv=None):

        # import pdb;pdb.set_trace()
        image = image.detach()
        sa_w = self.sa(image)
        image_sa = 0.1*sa_w + image

        image_features = self.image_encoder(image_sa.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # image_features =image_features.detach()


        n_class = self.n_class

        if test:
            #pdb.set_trace()
            proto_ts=self.mean
            proto_re = torch.unsqueeze(proto_ts, 0)
            image_re = image_features.reshape(-1,1,proto_ts.shape[1])
            dis_o = torch.norm(proto_re-image_re, p=2, dim=-1)
            dis_on= dis_o/dis_o.sum()
            similar = F.softmax(1.0/dis_on, dim=-1)
            ima_fea = similar @ proto_ts
            image_features = 0.2*ima_fea + 0.8*image_features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)


            text_features = self.text_features
            logit_scale= self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t() #
            return logits

        else:
            assert labels is not None
#############################prototype##############################
            # pdb.set_trace()
            self.lab_idx = labels.cpu().numpy().tolist()
            if epoch==0 and adv==None:
                lab_idx = labels.cpu().numpy().tolist()
                feature_cl = image_features.clone().detach().unsqueeze(1)
                #pdb.set_trace()
                for i in range(len(lab_idx)):
                    if self.curfeature[int(lab_idx[i])] == []:
                        self.curfeature[int(lab_idx[i])] = (feature_cl[i])
                    else:
                        self.curfeature[int(lab_idx[i])] = torch.cat((self.curfeature[int(lab_idx[i])],feature_cl[i]), dim=0)
            if epoch>0 and self.flag==0 and adv==None:
                prototype = [self.curfeature[key] for key in range(n_class)]
                self.prototype.data = torch.stack(prototype,dim=0)
                self.flag = 1
            self.mean = self.prototype.mean(dim=1)
            self.var = torch.sqrt(self.prototype.var(dim=1)+1e-6)
            image_features = 0.8*image_features + 0.2*self.mean[self.lab_idx]

#####################################################################################
            text_prompt, tokenized_prompts, nc_prompts, nc_tokenized_prompts = self.prompt_learner()
            n_prompt = text_prompt.shape[0]//n_class
            text_features = self.text_encoder(text_prompt, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.view(n_class, n_prompt, -1)
            text_feature = text_features.detach()
            text_mean =text_features.mean(dim=1)
            text_means= text_mean.detach()
            text_var = torch.sqrt(text_feature.var(unbiased=False,dim=1))

            loss_dis =((self.mean-text_means).norm() + (self.var-text_var).norm()) / self.n_class

            logit_scale = self.logit_scale.exp()
            logits =logit_scale * image_features @ text_mean.t()
            batch_size = labels.shape[0]
            # pdb.set_trace()
            text_features= text_features - text_mean.unsqueeze(1)  #unsqueeze
            diag_cov_martix = text_features.permute(2,0,1) @ text_features.permute(2,1,0)
            diag_cov_martix /= n_prompt + 1
            refined_logits = torch.einsum("bd, dik -> bik", [image_features**2, diag_cov_martix])

            sigma = refined_logits[torch.arange(batch_size), labels, labels].unsqueeze(-1) + \
                refined_logits[:, torch.arange(n_class), torch.arange(n_class) ] - \
                2 * refined_logits[torch.arange(batch_size), labels, : ]

            logits += 0.5*(logit_scale**2)*sigma.view(-1, n_class)

            nc_text_features = self.text_encoder(nc_prompts, nc_tokenized_prompts)
            nc_text_features = nc_text_features / nc_text_features.norm(dim=-1, keepdim=True)
            dis = nc_text_features @ nc_text_features.permute(1, 0)
            loss_m = dis[~torch.eye(self.n_prompt, dtype=torch.bool, device='cuda')].abs().mean()

            return logits, loss_m, sa_w, loss_dis


    @torch.no_grad()
    def set_classifier(self):
        text_prompt, tokenized_prompts = self.prompt_learner(infer=True)
        try:
            text_features = self.text_encoder(text_prompt, tokenized_prompts)
        except:
            text_features =[]
            batch_size=1000
            for bi in range(text_prompt.shape[0]//batch_size):
                batch_text_features = self.text_encoder(text_prompt[bi*1000:(bi+1)*1000], tokenized_prompts[bi*1000:(bi+1)*1000])
                text_features.append(batch_text_features)
            text_features = torch.cat(text_features, dim=0)
        n_dim = text_features.shape[-1]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.view(self.n_class, self.n_prompt, -1)
        text_features = text_features.mean(dim=1)
        self.text_features = text_features

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype


class SADA:
    def __init__(self, args, n_shot, n_ctx=16, use_float32=False, use_grad_checkpoint=False):
        clip_model, _ = load(args.ckpt_path)
        clip_model.eval()
        if use_float32:
            clip_model.float()
        self.clip_model = clip_model
        self.use_grad_checkpoint = use_grad_checkpoint
        self.n_shot=n_shot
        self.n_ctx=n_ctx
        self.n_prompt =args.n_prompt
        self.lr = args.lr*args.prompt_bsz*args.img_bsz/20
        self.conv_lr=args.conv_lr
        self.wd =args.wd
        self.epochs= args.epochs
        self.prompt_bsz= args.prompt_bsz
        self.img_bsz =args.img_bsz
        self.args= args
        self.eps=args.eps
        self.nb_iter=args.nb_iter
        self.eps_iter=args.eps_iter
        self.attack_type = args.attack_type 
        self.initial_const = args.initial_const 
        self.sa_dict={}


    def fit(self, data):

        train_loader = data['train_loader']

        iter_per_batch = max(int(self.n_prompt/self.prompt_bsz),1)

        per_epoch_steps = len(train_loader) * iter_per_batch

        if len(train_loader.dataset)< self.img_bsz:
            real_img_bsz = len(train_loader.dataset)
            self.lr = self.lr * real_img_bsz / self.img_bsz 
        else:
            real_img_bsz = self.img_bsz



        self.init_model(class_names=data['class_names'], per_epoch_steps=per_epoch_steps)

        # attacker_train = attackers[self.attack_type](loss_fn=nn.CrossEntropyLoss(), \
        #                                             eps=self.eps, nb_iter=self.nb_iter,\
        #                                             eps_iter=self.eps_iter, num_classes=self.n_class, \
        #                                             initial_const=self.initial_const)

        aug_list = get_composed_transform()
        self.sa_dict['base']=[]
        for p in self.model.sa.parameters():
            self.sa_dict['base'].append(p)
        for sa_i in range(len(aug_list)):
            self.sa_dict[sa_i] = self.sa_dict['base']


        self.model.eval()

        for epoch in range(self.epochs):
            for idx,(x, y) in enumerate(train_loader):


                img_list = TransformLoader.get_img(x, aug_list)

                for iter_idx in range(iter_per_batch):
                    #pdb.set_trace()
                    aug_idx = int(iter_idx // (iter_per_batch / len(aug_list)))
                    cur_iter_idx = epoch * per_epoch_steps + idx * iter_per_batch + iter_idx
                    self.scheduler.step(cur_iter_idx)
                    #pdb.set_trace()
                    #
                    self.model.sa.conv1.weight = self.sa_dict[aug_idx][0]
                    self.model.sa.conv2.weight = self.sa_dict[aug_idx][1]
                    x = img_list[aug_idx]
                    output, loss_m, sa_w, loss_dis = self.model(x.cuda(),y.cuda(),epoch=epoch)
                    loss = F.cross_entropy(output, y.cuda())
                    loss += (0.1*loss_m)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if epoch >0:
                        self.optimizer2.zero_grad()
                        loss_dis.backward()
                        self.optimizer2.step()

                    sa_i = 0
                    for p in self.model.sa.parameters():
                        self.sa_dict[aug_idx][sa_i] = p 
                        sa_i += 1
                #
                self.sa_dict['base'][0].data = (self.sa_dict[0][0].data + self.sa_dict[1][0].data
                                                + self.sa_dict[2][0].data + self.sa_dict[3][0].data)/4 
                self.sa_dict['base'][1].data = (self.sa_dict[0][1].data + self.sa_dict[1][1].data
                                                + self.sa_dict[2][1].data + self.sa_dict[3][1].data)/4


        self.model.set_classifier()

    def init_model(self, class_names, per_epoch_steps):

        self.n_class= len(class_names)
        clip_model = deepcopy(self.clip_model)
        #pdb.set_trace()
        self.model = CLIP(class_names, clip_model, self.n_shot, self.n_ctx, self.n_prompt, self.prompt_bsz)
        if self.use_grad_checkpoint:
            try:
                self.model.text_encoder.transformer.use_gradient_checkpoint = True
            except:
                self.model.text_encoder.module.transformer.use_gradient_checkpoint = True 

        prompt_params = [p for p in self.model.prompt_learner.parameters() if p.requires_grad]
        conv_params = [q for q in self.model.sa.parameters() if q.requires_grad]
        proto_params = [r for r in self.model.parameters() if r.requires_grad]
        self.optimizer = torch.optim.SGD([{'params':conv_params, 'weight_decay': 0, 'lr': self.conv_lr},
        {'params':prompt_params, 'lr': self.lr}], weight_decay=self.wd)
        self.optimizer2 = torch.optim.SGD([{'params':proto_params, 'weight_decay': 0, 'lr': 10*self.conv_lr}], weight_decay=self.wd)
        self.scheduler = build_bicosine_scheduler(
            self.optimizer,
            lr=[self.lr, self.conv_lr],
            total_step=self.epochs*per_epoch_steps)


    @torch.no_grad()
    def inference(self,image):
        self.model.sa.conv1.weight.data = self.sa_dict['base'][0].data 
        self.model.sa.conv2.weight.data = self.sa_dict['base'][1].data 
        logits = self.model(image, test=True)
        return logits.float().softmax(dim=-1)

    @torch.no_grad()
    def accuracy(self, loader, mean_per_class=False):
        if mean_per_class:
            return self._accuracy_mpc(loader)
        else:
            return self._accuracy(loader)

    def _accuracy_mpc(self, loader):
        n_class = self.n_class
        acc_per_class = [0 for _ in range(n_class)]
        count_per_class = [0 for _ in range(n_class)]
        for i,(x, y) in enumerate(loader):
            pred_y = self.inference(x.cuda())
            _, top_labels = pred_y.topk(1, dim=-1)
            for c in range(n_class):
                acc_per_class[c] += ((top_labels.view(-1) == y.cuda()) * (y.cuda()== c)).sum().item()
                count_per_class[c] += (y.cuda() == c).sum().item()
        acc = [a*1.0/c for (a, c) in zip(acc_per_class, count_per_class)]
        acc = np.array(acc).mean()
        return acc

    def _accuracy(self, loader):
        total_count=0
        acc_count =0
        for i, (x, y) in enumerate(loader):
            pred_y = self.inference(x.cuda())
            _, top_labels = pred_y.topk(1, dim=-1)
            acc_count += (top_labels.view(-1)==y.cuda()).sum().cpu().numpy()
            total_count += y.shape[0]
        acc = acc_count*1.0/total_count
        acc = acc.item()
        return acc

