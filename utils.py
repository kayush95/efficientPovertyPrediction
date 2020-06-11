from sklearn.ensemble import GradientBoostingClassifier
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import random
from scipy.stats.stats import pearsonr
import math
import sys
import os
import re
import torch
import torchvision.transforms as transforms
import torchvision.datasets as torchdata
import torchvision.models as torchmodels
import numpy as np
import pdb
import shutil
from random import randint, sample

from fmow_dataloader import CustomDatasetFromImages, SatelliteDatasetFromImages

def save_args(__file__, args):
    shutil.copy(os.path.basename(__file__), args.cv_dir)
    with open(args.cv_dir+'/args.txt','w') as f:
        f.write(str(args))

def performance_stats(policies, rewards, matches):
    # Print the performace metrics including the average reward, average number
    # and variance of sampled num_patches, and number of unique policies
    policies = torch.cat(policies, 0)
    rewards = torch.cat(rewards, 0)
    accuracy = torch.cat(matches, 0).mean()

    reward = rewards.mean()
    num_unique_policy = policies.sum(1).mean()
    variance = policies.sum(1).std()

    policy_set = [p.cpu().numpy().astype(np.int).astype(np.str) for p in policies]
    policy_set = set([''.join(p) for p in policy_set])

    return accuracy, reward, num_unique_policy, variance, policy_set

def performance_stats_gbdt(policies, rewards, mse, counts_diff):
    # Print the performace metrics including the average reward, average number
    # and variance of sampled num_patches, and number of unique policies
    policies = torch.cat(policies, 0)
    rewards = torch.cat(rewards, 0)

    counts_diff = torch.cat(counts_diff, 0)
    counts_diff = counts_diff.mean()

    
    if mse == []:
        mse = 0.0
    else:
        mse = torch.cat(mse, 0)
        mse = mse.mean()

    reward = rewards.mean()
    num_unique_policy = policies.sum(1).mean()
    variance = policies.sum(1).std()

    policy_set = [p.cpu().numpy().astype(np.int).astype(np.str) for p in policies]
    policy_set = set([''.join(p) for p in policy_set])

    return mse, reward, num_unique_policy, variance, policy_set, counts_diff

def gbdt_score(v_inputs, targets):
    X_test = v_inputs.cpu().detach().numpy()
    y_true = targets
    filename = 'gbdt_model_stratified_corrected_1.sav'
    model = pickle.load(open(filename, 'rb'))
    y_pred = torch.tensor(list(model.predict(X_test)))
    score = (y_true - y_pred)**2
    return score, y_pred

def compute_reward_gbdt(gbdt_sc, policy, counts, counts_hr):
    counts_diff = counts - counts_hr
    counts_diff_norm = (torch.norm(counts_diff, p=1, dim=1)).cuda().double()

    patch_use = policy.sum(1).float() / policy.size(1)
    sparse_reward = 1.0 - patch_use**2
    reward = 2.0*sparse_reward.double() - 0.1*counts_diff_norm

    counts_diff_norm = counts_diff_norm.unsqueeze(1)
    reward = reward.unsqueeze(1)
    return reward, counts_diff_norm

def compute_reward(preds, targets, policy, penalty):
    # Reward function favors policies that drops patches only if the classifier
    # successfully categorizes the image
    patch_use = policy.sum(1).float() / policy.size(1)
    sparse_reward = 1.0 - patch_use**2

    _, pred_idx = preds.max(1)
    match = (pred_idx==targets).data

    reward = sparse_reward
    reward[1-match] = penalty
    reward = reward.unsqueeze(1)

    return reward, match.float()

def load_weights_to_flatresnet(source_model, target_model):
    # compatibility for nn.Modules + checkpoints
    if hasattr(source_model, 'state_dict'):
        source_model = {'state_dict': source_model.state_dict()}
    source_state = source_model['state_dict']
    target_state = target_model.state_dict()

    # remove the module. prefix if it exists (thanks nn.DataParallel)
    if source_state.keys()[0].startswith('module.'):
        source_state = {k[7:]:v for k,v in source_state.items()}

    common = set(['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var','fc.weight', 'fc.bias'])
    for key in source_state.keys():

        if key in common:
            target_state[key] = source_state[key]
            continue

        if 'downsample' in key:
            layer, num, item = re.match('layer(\d+).*\.(\d+)\.(.*)', key).groups()
            translated = 'ds.%s.%s.%s'%(int(layer)-1, num, item)
        else:
            layer, item = re.match('layer(\d+)\.(.*)', key).groups()
            translated = 'blocks.%s.%s'%(int(layer)-1, item)


        if translated in target_state.keys():
            target_state[translated] = source_state[key]
        else:
            print(translated, 'block missing')

    target_model.load_state_dict(target_state)
    return target_model

def load_checkpoint(rnet, agent, load):
    if load=='nil':
        return None

    checkpoint = torch.load(load)
    if 'resnet' in checkpoint:
        rnet.load_state_dict(checkpoint['resnet'])
        print('loaded resnet from', os.path.basename(load))
    if 'agent' in checkpoint:
        agent.load_state_dict(checkpoint['agent'])
        print('loaded agent from', os.path.basename(load))
    # backward compatibility (some old checkpoints)
    if 'net' in checkpoint:
        checkpoint['net'] = {k:v for k,v in checkpoint['net'].items() if 'features.fc' not in k}
        agent.load_state_dict(checkpoint['net'])
        print('loaded agent from', os.path.basename(load))

def get_transforms(rnet, dset):

    if dset=='satellite':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])

    elif dset=='C10' and rnet=='R32':
        mean = [x/255.0 for x in [125.3, 123.0, 113.9]]
        std = [x/255.0 for x in [63.0, 62.1, 66.7]]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])

    elif dset=='C100' or dset=='C10' and rnet!='R32':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            ])

    elif dset=='ImgNet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([
            transforms.Scale(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])

        transform_test = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])

    elif dset=='fMoW':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([
           transforms.Scale(224),
           transforms.RandomCrop(224),
           transforms.ToTensor(),
           transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
           transforms.Scale(224),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           transforms.Normalize(mean, std)
        ])

    elif dset=='CARS':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([
           transforms.Resize(256),
           transforms.RandomCrop(224),
           transforms.RandomHorizontalFlip(),
           transforms.ToTensor(),
           transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
           transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           transforms.Normalize(mean, std)
        ])

    return transform_train, transform_test

def agent_input(input_org, downsampled_size, original_size):
    """ Downsample the Original Images for The Policy Network
    """
    input_lr = torch.nn.functional.interpolate(input_org, (downsampled_size, downsampled_size))

    return input_lr

def agent_chosen_input(input_org, policy, mappings, interval):
    """ Generate masked images w.r.t policy learned by the agent.
    """
    input_full = input_org.clone()
    sampled_img = torch.zeros([input_org.shape[0], input_org.shape[1], input_org.shape[2], input_org.shape[3]])
    for pl_ind in range(policy.shape[1]):
        mask = (policy[:, pl_ind] == 1).cpu()
        sampled_img[:, :, mappings[pl_ind][0]:mappings[pl_ind][0]+interval, mappings[pl_ind][1]:mappings[pl_ind][1]+interval] = input_full[:, :, mappings[pl_ind][0]:mappings[pl_ind][0]+interval, mappings[pl_ind][1]:mappings[pl_ind][1]+interval]
        sampled_img[:, :, mappings[pl_ind][0]:mappings[pl_ind][0]+interval, mappings[pl_ind][1]:mappings[pl_ind][1]+interval] *= mask.unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
    input_org = sampled_img

    return input_org.cuda()


def prob_counts(counts_org, policy):
    counts_full = counts_org.clone()
    counts_full = torch.sum(counts_full, dim=2)
    counts_full = counts_full.view(1, -1)
    policy_full = policy.clone()
    policy_full = policy_full.view(1, -1)
    counts_full = counts_full.detach()
    policy_full = policy_full.detach()

    return counts_full, policy_full

def agent_chosen_input_counts(counts_org, policy):
    """ Generate masked counts w.r.t policy learned by the agent.
    """
    counts_full = counts_org.clone()
    sampled_counts = torch.zeros(counts_org.shape[0], counts_org.shape[2])
    for pl_ind in range(policy.shape[1]):
        mask = (policy[:, pl_ind] == 1).cpu()
        sampled_counts[:, :] += counts_full[:, pl_ind, :].float()*mask.unsqueeze(1).float()
    counts_org = sampled_counts
    return counts_org

def action_space_model(dset):
    if dset == 'satellite':
        img_size = 224*17
        interval = 224       
    elif dset == 'C10' or dset == 'C100':
        img_size = 32
        interval = 8
    elif dset == 'fMoW':
        img_size = 224
        interval = 56
    elif dset == 'ImgNet':
        img_size = 224
        interval = 56
    elif dset == 'CARS':
        img_size = 224
        interval = 56

    # Model the action space by dividing the image space into equal size patches
    mappings = []
    for cl in range(0, img_size, interval):
        for rw in range(0, img_size, interval):
            mappings.append([cl, rw])

    return mappings, img_size, interval

# Pick from the datasets available and the hundreds of models we have lying around depending on the requirements.
def get_dataset(model, root='efficient_17X17'):
    rnet, dset = model.split('_')
    transform_train, transform_test = get_transforms(rnet, dset)

    if dset=='satellite':
        trainset = SatelliteDatasetFromImages(root, 'train_gt_stratified_corrected_1_sentinel.npy', transform_train)
        testset = SatelliteDatasetFromImages(root, 'test_gt_stratified_corrected_1_sentinel.npy', transform_test)        
    elif dset=='C10':
        trainset = torchdata.CIFAR10(root=root, train=True, download=True, transform=transform_train)
        testset = torchdata.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    elif dset=='C100':
        trainset = torchdata.CIFAR100(root=root, train=True, download=True, transform=transform_train)
        testset = torchdata.CIFAR100(root=root, train=False, download=True, transform=transform_test)
    elif dset=='ImgNet':
        trainset = torchdata.ImageFolder(root+'/ImageNet/train/', transform_train)
        testset = torchdata.ImageFolder(root+'/ImageNet/test/', transform_test)
    elif dset=='fMoW':
        trainset = CustomDatasetFromImages(root+'/fMoW/train.csv', transform_train)
        testset = CustomDatasetFromImages(root+'/fMoW/test.csv', transform_test)
    elif dset=='CARS':
        trainset = CustomDatasetFromImages(root+'/data/scars/train_upd.csv', transform_train)
        testset = CustomDatasetFromImages(root+'/data/scars/val_upd.csv', transform_test)

    return trainset, testset

def set_parameter_requires_grad(model, feature_extracting):
    # When loading the models, make sure to call this function to update the weights
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_model(model):

    from models import resnet, base

    if model=='policy_satellite':
        # agent = resnet.Policy224([1,1,1,1], num_blocks=289, num_feat=20)
        # agent = resnet.Policy224GRU([1,1,1,1], num_blocks=289, num_feat=128)
        # agent = resnet.PolicySeq()
        agent = resnet.Policy2x2([1,1,1,1], num_blocks=4)
        rnet_hr = None
        rnet_lr = None

    elif model=='R32_C10':
        layer_config = [5, 5, 5]
        rnet_hr = resnet.FlatResNet32(base.BasicBlock, layer_config, num_classes=10)
        rnet_lr = resnet.FlatResNet32(base.BasicBlock, layer_config, num_classes=10)
        agent = resnet.Policy32([1,1,1], num_blocks=16)

    elif model=='R32_C100':
        layer_config = [5, 5, 5]
        rnet_hr = resnet.FlatResNet32(base.BasicBlock, layer_config, num_classes=100)
        rnet_lr = resnet.FlatResNet32(base.BasicBlock, layer_config, num_classes=100)
        agent = resnet.Policy32([1,1,1], num_blocks=16)

    elif model=='R50_ImgNet':
        agent = resnet.Policy224([1,1,1,1], num_blocks=16)
        """ High Res. Classifier """
        rnet_hr = torchmodels.resnet50(pretrained=False)
        set_parameter_requires_grad(rnet_hr, False)
        num_ftrs = rnet_hr.fc.in_features
        rnet_hr.fc = torch.nn.Linear(num_ftrs, 1000)
        """ Low Res. Classifier """
        rnet_lr = torchmodels.resnet50(pretrained=False)
        set_parameter_requires_grad(rnet_lr, False)
        num_ftrs = rnet_lr.fc.in_features
        rnet_lr.fc = torch.nn.Linear(num_ftrs, 1000)

    elif model=='R34_fMoW':
        agent = resnet.Policy224([1,1,1,1], num_blocks=16)
        """ High Res. Classifier """
        rnet_hr = torchmodels.resnet34(pretrained=True)
        set_parameter_requires_grad(rnet_hr, False)
        num_ftrs = rnet_hr.fc.in_features
        rnet_hr.fc = torch.nn.Linear(num_ftrs, 62)
        """ Low Res. Classifier """
        rnet_lr = torchmodels.resnet34(pretrained=True)
        set_parameter_requires_grad(rnet_lr, False)
        num_ftrs = rnet_lr.fc.in_features
        rnet_lr.fc = torch.nn.Linear(num_ftrs, 62)

    elif model=='R34_CARS':
        agent = resnet.Policy224([1,1,1,1], num_blocks=16)
        """ High Res. Classifier """
        rnet_hr = torchmodels.resnet34(pretrained=True)
        set_parameter_requires_grad(rnet_hr, False)
        num_ftrs = rnet_hr.fc.in_features
        rnet_hr.fc = torch.nn.Linear(num_ftrs, 196)
        """ Low Res. Classifier """
        rnet_lr = torchmodels.resnet34(pretrained=True)
        set_parameter_requires_grad(rnet_lr, False)
        num_ftrs = rnet_lr.fc.in_features
        rnet_lr.fc = torch.nn.Linear(num_ftrs, 196)

    return rnet_hr, rnet_lr, agent
