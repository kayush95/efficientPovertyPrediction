
import os
from tensorboard_logger import configure, log_value
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.utils.data as torchdata
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import utils
import torch.optim as optim
import sys
from torch.distributions import Multinomial, Bernoulli, Categorical
from sklearn.metrics import r2_score
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import argparse
parser = argparse.ArgumentParser(description='Tile Drop Pre-Training')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--model', default='R32_C10', help='R<depth>_<dataset> see utils.py for a list of configurations')
parser.add_argument('--ckpt_hr_cl', help='checkpoint directory for the high resolution classifier')
parser.add_argument('--data_dir', default='data', help='data directory')
parser.add_argument('--load', default=None, help='checkpoint to load agent from')
parser.add_argument('--cv_dir', default='cv/tmp/', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epoch_step', type=int, default=10000, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=10000, help='total epochs to run')
parser.add_argument('--parallel', action ='store_true', default=False, help='use multiple GPUs for training')
parser.add_argument('--penalty', type=float, default=-0.5, help='gamma: reward for incorrect predictions')
parser.add_argument('--alpha', type=float, default=0.6, help='probability bounding factor')
parser.add_argument('--sigma', type=float, default=0.1, help='multiplier for the entropy loss')
args = parser.parse_args()

if not os.path.exists(args.cv_dir):
    os.system('mkdir ' + args.cv_dir)
utils.save_args(__file__, args)


def train(epoch):
    # This steps trains the policy network only
    agent.train()

    mse, rewards, rewards_baseline, policies = [], [], [], []
    counts_diff = []
    for batch_idx, (inputs, targets, counts, cluster) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):
        # print('inputs.shape', inputs.shape)
        # print('targets.shape', targets.shape)
        # print('counts.shape', counts.shape)
        batch_size = inputs.shape[0]

        if not args.parallel:
            inputs = inputs.cuda()

        inputs = inputs.squeeze(0)
        counts = counts.squeeze(0)
        # print('inputs.shape', inputs.shape)
        # print('counts.shape', counts.shape)

        probs = agent(inputs)
        probs = probs*args.alpha + (1-args.alpha) * (1-probs)

        # print('probs', probs.shape)
        # # Sample the policies from the Bernoulli distribution characterized by agent's output
        distr = Bernoulli(probs)
        policy_sample = distr.sample()

        # print('policy_sample.shape', policy_sample.shape)
        # Test time policy - used as baseline policy in the training step
        policy_map = probs.data.clone()
        policy_map[policy_map<0.5] = 0.0
        policy_map[policy_map>=0.5] = 1.0
        policy_map = Variable(policy_map)

        # print('policy_map.shape', policy_map.shape)

        policy_hr = policy_map.data.clone()
        # print(policy_hr)
        policy_hr[:] = 1.0
        # print(policy_hr)
        policy_hr = Variable(policy_hr)

        # print('policy_hr.shape', policy_hr.shape)
        # print('policy_hr', policy_hr)

        # Agent sampled high resolution images
        counts_map = utils.agent_chosen_input_counts(counts, policy_map)
        counts_sample = utils.agent_chosen_input_counts(counts, policy_sample.int())
        counts_hr = utils.agent_chosen_input_counts(counts, policy_hr)

        # print('counts_map.shape', counts_map.shape)

        # gbdt_sc_map, _ = utils.gbdt_score(counts_map, targets)
        # gbdt_sc_sample, _ = utils.gbdt_score(counts_sample, targets)
        gbdt_sc_map = None
        gbdt_sc_sample = None

        # print('gbdt_sc_map.shape', gbdt_sc_map.shape)

        # Find the reward for baseline and sampled policy
        reward_map, counts_diff_map = utils.compute_reward_gbdt(gbdt_sc_map, policy_map.data, counts_map, counts_hr)
        reward_sample, counts_diff_sample  = utils.compute_reward_gbdt(gbdt_sc_sample, policy_sample.data, counts_sample, counts_hr)
        # print('counts_diff_sample', counts_diff_sample.data)
        # print('reward_map.shape', reward_map.shape)
        # sys.exit()
        advantage = reward_sample.cuda().float() - reward_map.cuda().float()

        # Find the loss for only the policy network
        loss = -distr.log_prob(policy_sample)
        # print('loss.shape', loss.shape)
        loss = loss * Variable(advantage).expand_as(policy_sample)
        # print('loss.shape', loss.shape)
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # mse.append(torch.from_numpy(np.array([gbdt_sc_sample])))
        # mse.append(gbdt_sc_sample.cpu())

        counts_diff_sample_all = counts_diff_sample.sum(0).unsqueeze(0)
        # print('counts_diff_sample_all', counts_diff_sample_all.shape)

        counts_diff.append(counts_diff_sample_all)
        rewards.append(reward_sample.cpu())
        rewards_baseline.append(reward_map.cpu())

        plcy = policy_sample.view(1, -1)
        # print('plcy', plcy.shape)
        policies.append(plcy.data.cpu())

    mse, reward, sparsity, variance, policy_set, counts_diff = utils.performance_stats_gbdt(policies, rewards, mse, counts_diff)
    
    print('Train: %d | Mse: %.3f | Rw: %.2E | S: %.3f | V: %.3f | #: %d | Diff: %.3f'%(epoch, mse, reward, sparsity, variance, len(policy_set), counts_diff))
    log_value('train_mse', mse, epoch)
    log_value('train_reward', reward, epoch)
    log_value('train_sparsity', sparsity, epoch)
    log_value('train_variance', variance, epoch)
    log_value('train_baseline_reward', torch.cat(rewards_baseline, 0).mean(), epoch)
    log_value('train_unique_policies', len(policy_set), epoch)

def test(epoch):
    # Test the policy network and the high resolution classifier
    agent.eval()

    mse, rewards, policies, y_true, y_preds = [], [], [], [], []
    counts_diff = []
    rewards_half = []
    for batch_idx, (inputs, targets, counts, cluster) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
        batch_size = inputs.shape[0]
        if not args.parallel:
            inputs = inputs.cuda()

        # print(counts.shape)
        # counts_check = counts.sum(dim=2)
        # print(counts_check)
        # counts_check[counts_check>0] = 1
        # print(counts_check)
        # print(counts_check.sum(dim=1))
        # sys.exit()

        inputs = inputs.squeeze(0)
        counts = counts.squeeze(0)

        probs = agent(inputs)

        # print('probs', probs.shape)
        # print('probs', probs)
        # Sample the policy from the agents output
        policy = probs.data.clone()
        policy[policy<0.5] = 0.0
        policy[policy>=0.5] = 1.0
        policy = Variable(policy)
        # print(policy)

        policy_hr = probs.data.clone()
        policy_hr[:] = 1.0
        policy_hr = Variable(policy_hr)
        # print('policy_hr', policy_hr)

    
        counts_map = utils.agent_chosen_input_counts(counts, policy)
        counts_hr = utils.agent_chosen_input_counts(counts, policy_hr)
        # counts_half = utils.agent_chosen_input_counts(counts, policy_half)

        # print('counts_map', counts_map.shape)
        counts_map_all = counts_map.sum(0).unsqueeze(0)
        counts_hr_all = counts_hr.sum(0).unsqueeze(0)
        # print('counts_map_all', counts_map_all.shape)

        gbdt_sc_map, y_pred = utils.gbdt_score(counts_map_all, targets)

        # gbdt_sc_hr, y_pred = utils.gbdt_score(counts_hr, targets)
        # print('y_pred', y_pred)
        # print('targets', targets.numpy())
        y_preds += list(y_pred)
        y_true += list(targets.numpy())
        # print('y_preds', y_preds)
        # print('y_true', y_true)

        reward, counts_diff_map = utils.compute_reward_gbdt(gbdt_sc_map, policy.data, counts_map, counts_hr)
        # reward_half, counts_diff_map_half = utils.compute_reward_gbdt(gbdt_sc_map, policy_half.data, counts_half, counts_hr)
        # print('counts_diff_map', counts_diff_map)

        counts_diff_map_all = counts_diff_map.sum(0).unsqueeze(0)
        counts_diff.append(counts_diff_map_all)
        rewards.append(reward)
        # rewards_half.append(reward_half)
        plcy = policy.view(1, -1)
        policies.append(plcy.data)
        # mse.append(torch.from_numpy(np.array([gbdt_sc_map])))
        mse.append(gbdt_sc_map.cpu())

    y_true = [float(i) for i in y_true]
    y_preds = [float(i) for i in y_preds]
    # print(y_preds)
    # print(y_true)
    # print('len(y_true)', len(y_true))
    # print('len(y_pred)', len(y_preds))
    # print('len(policies)', len(policies))
    # print(policies)
    r2 = r2_score(y_true, y_preds) 
    mse, reward, sparsity, variance, policy_set, counts_diff = utils.performance_stats_gbdt(policies, rewards, mse, counts_diff)

    print('Test - Mse: %.3f | R2: %.3f | Rw: %.2E | S: %.3f | V: %.3f | #: %d | Diff: %.3f'%(mse, r2, reward, sparsity, variance, len(policy_set), counts_diff))
    # sys.exit()
    # log_value('test_rewards_objects', rewards_half, epoch)
    log_value('test_mse', mse, epoch)
    log_value('test_r2', r2, epoch)
    log_value('test_reward', reward, epoch)
    log_value('test_sparsity', sparsity, epoch)
    log_value('test_variance', variance, epoch)
    log_value('test_unique_policies', len(policy_set), epoch)

    # save the model --- agent
    agent_state_dict = agent.module.state_dict() if args.parallel else agent.state_dict()

    state = {
      'agent': agent_state_dict,
      'epoch': epoch,
      'reward': reward,
      'mse': mse,
      'r2': r2,
      'sparsity': sparsity,
      'counts_diff': counts_diff
    }
    torch.save(state, args.cv_dir+'/ckpt_E_%d_M_%.3f_r2_%.3f_R_%.2E_S_%.3f_D_%.3f'%(epoch, mse, r2, reward, sparsity, counts_diff))

#--------------------------------------------------------------------------------------------------------#
trainset, testset = utils.get_dataset(args.model, args.data_dir)
trainloader = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testloader = torchdata.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
_, _, agent = utils.get_model(args.model)

# Save the args to the checkpoint directory
configure(args.cv_dir+'/log', flush_secs=5)

num_patches = 289 # Fixed in the paper, but can be changed
mappings, img_size, interval = utils.action_space_model(args.model.split('_')[1])

start_epoch = 0
print('args.load', args.load)
if args.load is not None:
    checkpoint = torch.load(args.load)
    agent.load_state_dict(checkpoint['agent'])
    start_epoch = checkpoint['epoch'] + 1
    print('loaded agent from', args.load)

# if args.ckpt_hr_cl is not None:
#     checkpoint = torch.load(args.ckpt_hr_cl)
#     if args.model.split('_')[1] == 'C10' or args.model.split('_')[1] == 'C100':
#         rnet.load_state_dict(checkpoint['state_dict'])
#     else:
#         utils.load_weights_to_flatresnet(checkpoint, rnet)
#     print('loaded the high resolution classifier')

# Parallelize the models if multiple GPUs available - Important for Large Batch Size
if args.parallel:
    agent = nn.DataParallel(agent, device_ids=[0,1])

agent.cuda() # Only agent is updated

# Update the parameters of the policy network
optimizer = optim.Adam(agent.parameters(), lr=args.lr)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [600])

for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
    train(epoch)
    if epoch % 5 == 0:
        test(epoch)
    if epoch % 20 == 0:
        if args.alpha < 0.95:
            args.alpha = args.alpha + 0.05
    lr_scheduler.step()