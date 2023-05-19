import torch
import random
import numpy as np
import argparse
import os
import math

from classifier.coop import CoOp
from classifier.sada import SADA

from dataset.build_dataset import build_dataset, build_dataset_fs
from dataset.augmentation import TransformLoader, get_composed_transform

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def vlmodel_infer(n_shot, args):


    results =[]
    for n in range(args.num_runs):
        setup_seed(args.seed+n)

        train_db, _, test_db = build_dataset_fs(args.db_name, args.root, n_shot=n_shot, transform_mode=args.aug)
        
        train_loader = torch.utils.data.DataLoader(
            train_db,
            batch_size=args.img_bsz,
            num_workers=4,
            shuffle=True,
            pin_memory=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_db,
            batch_size=128,
            num_workers=4,
            pin_memory=True,
        )

        if args.model == 'coop':
            model = CoOp(args=args)
        elif args.model == 'sada':
            model = SADA(args=args,n_shot=n_shot)

        # adaption

        data = {'train_loader': train_loader, 'class_names': train_db.get_classes()}


        model.fit(data)

        # evaluation
        acc = model.accuracy(test_loader, mean_per_class=args.mean_per_class)

        results.append({'seed': args.seed+n, 'select_index': data['train_loader'].dataset.select_labels, 'acc':acc})
        print(n+1, acc*100)

        if args.save_path is not None:
            save_model_path = os.path.join(args.save_path, '{}shot_{}.pt'.format(n_shot, args.seed+n))
            torch.save(model.model.prompt_learner.state_dict(), save_model_path)

        del model
        for _ in range(5):
            torch.cuda.empty_cache()

        if args.save_path is not None:
            save_result_path = os.path.join(args.save_path, 'result_{}shot.npy'.format(n_shot))
            np.save(save_result_path, results)

        return results


def parse_option():
    parser = argparse.ArgumentParser('Prompt Learning for CLIP', add_help=False)

    parser.add_argument("--root", type=str, default='/data/wangrunqi/',help='root')
    parser.add_argument("--aug",type=str, default='flip', help='root')

    parser.add_argument("--mean_per_class", action='store_true', help='mean_per_class')
    parser.add_argument("--db_name", type=str, default='cifar10', help='dataset name')
    parser.add_argument("--num_runs", type=int, default=3, help='num_runs')
    parser.add_argument("--seed", type=int, default=0, help='random seed')

    parser.add_argument("--arch", type=str, default='RN50', help='arch')
    parser.add_argument("--ckpt_path", type=str, default=None, help='ckpt_path')
    parser.add_argument("--save_path", type=str, default='/home/wangrq/SADA/save/', help='save_path')

    # optimization setting
    parser.add_argument("--lr", type=float, default=5e-4, help='num_runs')
    parser.add_argument("--conv_lr", type=float, default=1e-3, help='num_runs')
    parser.add_argument("--wd", type=float, default=0.0, help='num_runs')
    parser.add_argument("--epochs", type=int, default=50, help='num_runs')
    parser.add_argument("--img_bsz", type=int, default=20, help='num_runs')
    parser.add_argument("--train_batch", type=int, default=20, help='num_runs')

    #model setting
    parser.add_argument("--model", type=str, default='sada', help='model')
    parser.add_argument("--n_prompt", type=int, default=32, help='num_runs')
    parser.add_argument("--prompt_bsz", type=int, default=4, help='num_runs')

    #adv
    parser.add_argument('--nb_iter', help='Adversarial attack iteration', type=int, default=3)
    parser.add_argument('--eps', help='Adversarial attack maximal perturbation', type=float, default=8/255)
    parser.add_argument('--eps_iter', help='Adversarial attack step size', type=float, default=2/255)
    parser.add_argument('--initial_const', help='initial value of the constant c', type=float, default=0.1)
    parser.add_argument('--attack_type', help='type of adversarial attack', type=str, default='fgsm_delta')

    args, unparsed = parser.parse_known_args()


    args.mean_per_class = False
    if args.db_name in ['oxfordpets', 'fgvc_aircraft', 'oxford_flowers', 'caltech101']:
        args.mean_per_class = True

    if args.db_name in ['cifar10', 'eurosat','stl1e']:
        args.num_runs= 10
    # args.num_runs = 1

    if args.ckpt_path is None:
        args.ckpt_path = '/home/wangrq/pretrain/{}.pt'.format(args.arch)

    return args


def main(args):
    print("{}".format(args).replace(',',',\n'))

    n_shots = [1, 2, 4, 8, 16]

    print('start training: {}, num_runs: {}, mean_per_class:{}'.format(args.db_name, args.num_runs, args.mean_per_class))
    for n_shot in n_shots:
        results = vlmodel_infer(n_shot,args)
        
        acc = np.array([res['acc'] for res in results])
        mean = acc.mean()
        std = acc.std()
        c95 = 1.96*std/math.sqrt(acc.shape[0])

        print('dataset:{}, model: {}, arch:{}, n_shot:{}, acc: {:.2f}+{:.2f}'.format(
            args.db_name, args.model, args.arch, n_shot, mean*100, c95*100))

if __name__ == '__main__':
    args = parse_option()
    main(args)