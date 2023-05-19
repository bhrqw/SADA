# from matplotlib.transforms import Transform
import itertools
import torch
import torchvision
import torchvision.transforms as transforms
# from torch.utils.data import Dataset

class TransformLoader():
    def get_img(img, transform):
        img_list = []
        for i in transform:
            img = i(img)
            img_list.append(img)
        return img_list

def parse_transform(transform_type, n_px=224):

    if transform_type == 'RandomColorJitter':
        return torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)], p=1.0)
    if transform_type == 'RandomGrayscale':
        return torchvision.transforms.RandomGrayscale(p=0.1)
    elif transform_type == 'RandomGaussianBlur':
        return torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur(kernel_size=(5, 5))], p=0.3)
    elif transform_type == 'CenterCrop':
        return torchvision.transforms.CenterCrop(n_px)
    elif transform_type == 'Scale':
        return torchvision.transforms .Resize([int(n_px),int(n_px)])
    elif transform_type == 'RandomResizedCrop':
        return torchvision.transforms.RandomResizedCrop(n_px)
    elif transform_type == 'RandomCrop':
        return torchvision.transforms.RandomCrop(0.7*n_px)
    elif transform_type == 'Resize_up':
        return torchvision.transforms .Resize([int(n_px * 1.15), int(n_px * 1.15)])
    elif transform_type =='Flip':
        return transforms.RandomHorizontalFlip()
    elif transform_type == 'RandomRotation':
        return transforms.RandomRotation(45)


    else:
        method = getattr(torchvision.transforms, transform_type)
        return method()

def get_composed_transform():
##############################################Cifar10############H
    transform_list1 = ['Flip','CenterCrop']

    transform_list3 = ['RandomGaussianBlur']

    transform_list2 = ['RandomGrayscale']

    transform_list4 = ['Flip','CenterCrop']

    transform_list = [transform_list1,transform_list2,transform_list3,transform_list4]
############################################################################
    # transform_list1 = ['RandomGrayscale']
    # transform_list2 = ['Flip','CenterCrop']
    # transform_list3 = ['RandomGaussianBlur']
    # transform_list4 =['RandomRotation']
    # transform_lists = ['Flip', 'Randomcrop','scale']
    # transform_list6 = ['RandomColorJitter']
    # transform_list = [transform_list1,transform_list2,transform_list3,transform_list4,transform_list5, transform_list6]
    transform=[]
    for i in range (len(transform_list)):
        transform_funcs = [parse_transform(x) for x in transform_list[i]]
        transform.append(transforms.Compose(transform_funcs))
    return transform