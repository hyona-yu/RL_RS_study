#https://github.com/reinforcement-learning-kr/pg_travel/blob/master/mujoco/agent/ppo_gae.py
#PPO + GAE
import numpy as np
import torch
import random
import math
import os
import gym
import argparse
from collections import deque
from tensorboard import SummaryWriter

class HyperParams:
    gamma = 0.99
    lamda = 0.98
    hidden = 64
    critic_lr = 0.0003
    actor_lr = 0.0003
    batch_size = 64
    l2_rate = 0.001
    max_kl = 0.01
    clip_param = 0.2
hp = HyperParams()

class Actor(nn.Module):
    def __init__(self, in,out):
        self.in = in
        self.out = out
        super(Actor,self).__init__()
        self.l1 = torch.nn.Linear(self.in, hp.hidden)
        self.l2 = torch.nn.LInear(hp.hidden, hp.hidden)
        self.l3 = torch.nn.Linear(hp.hidden, self.out)
        self.tanh = torch.nn.functional.tanh()

        torch.nn.init.xavier_normal_(l1.weight)
        torch.nn.init.xavier_normal_(l2.weight)
        torch.nn.init.xavier_normal_(l3.weight)

    def forward(self, x):
        out = self.l1(x)
        out = self.tanh(out)
        out = self.l2(out)
        out = self.tanh(out)
        out = self.l3(out)
        logstd = torch.zeros_like(out)
        std = torch.exp(logstd)
        return out, std, logstd

class Critic(nn.Module):
    def __init__(self,in,out):
        self.in = in
        self.out = out
        super(Critic,self).__init__()
        self.l1 = torch.nn.Linear(self.in, hp.hidden)
        self.l2 = torch.nn.Linear(hp.hidden, hp.hidden)
        self.l3 = torch.nn.Linear(hp.hidden, self.out)
        self.tanh = torch.nn.functional.tanh()

        torch.nn.init.xavier_normal_(l1.weight)
        torch.nn.init.xavier_normal_(l2.weight)
        torch.nn.init.xavier_normal_(l3.weight)

    def forward(self, x):
        out = self.l1(x)
        out = self.tanh(out)
        out = self.l2(out)
        out = self.tanh(out)
        out = self.l3(out)
        return out

def get_action(mu, std):
    action = torch.normal(mu, std)
    action = action.data.numpy()
    return action
def log_density(x, mu, std, logstd):#얘는 왜있냐
    var = std.pow(2)
    log_density = -(x-mu).pow(2) / (2*var) - 0.5 * math.log(2*math.pi) - logstd
    return log_density.sum(1, keepdim = True)

def kl_divergence(new_actor, old_actor, states):
    mu, std, logstd = new_actor(torch.Tensor(states))
    mu_old, std_old, logstd_old = old_actor(torch.Tensor(states))
    mu_old = mu_old.detach()
    std_old = std_old.detach()
    logstd_old = logstd_old.detach()
    # Dtv p||q = 1/2 * expect(p -q)
    #Dkl pi||pi' = pi(a) * log(pi(a)/pi*(a))
    kl = logstd_old - logstd + (std_old.pow(2)+ (mu_old - mu).pow(2)) / (2.0 * std.pow(2)) - 0.5
    #constrained optimization 이거 minimize하면 됨?
    return kl.sum(1, keepdim = True)

def save_checkpoint(state, filename = 'checkpoint.pth.tar'):
    torch.save(state, filename)


def get_gae(rewards, masks, values):
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)

    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + hp.gamma * running_returns * masks[t]
        running_tderror = rewards[t] + hp.gamma * previous_value * masks[t] - values.data[t]
        running_advants = running_tderror + hp.gamma * hp.lamda* running_advants * masks[t]

        returns[t] = running_returns
        previous_value = values.data[t]
        advants[t] = running_advants

    advants = (advants - advants.mean()) / advants.std()
    return returns, advants
def surrogate_loss(actor, advants, states, old_policy, acions, ind):
    mu, std, logstd = actor(torch.Tensor(states))
    new_policy = log_density(actions, mu, std, logstd)
    old_policy = old_policy[ind]
    ratio = torch.exp(new_policy - old_policy)
    surrogate = ratio * advants  #이게 대리다?
    return surrogate, ratio

def train_model(actor, critic, memory, actor_optim, critic_optim):
    memory = np.array(memory) #얘는 replay buffer인걸까?
    states = np.vstack(memory[:,0])
    actions = list(memory[:,1]) #얜 왜 list로 받냐
    rewards = list(memory[:,2])
    masks = list(memory[:,3])
    values =  critic(torch.Tensor(states))

    # GAE , old policy 의 log prob 가져오기
    returns, advants = get_gae(rewards, masks, values)
    mu, std, logstd = actor(torch.Tensor(states))
    old_policy = log_density(torch.Tensor(actions), mu, std, logstd)
    old_values = critic(torch.Tensor(states))

    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)

    ## value loss, actor loss 받아오고 update actor&critic
    for epoch in range(10): #왜 열번밖에 안도냐?
        np.random.shuffle(arr)

        for i in range(n//hp.batch_size):
            batch_index = arr[hp.batch_size * i : hp.batch_size*(i+1)]
            batch_index = torch.LongTensor(batch_index)
            inputs = torch.Tensor(states)[batch_index]
            returns_samples=  returns.unsqueeze(1)[batch_index]
            advants_samples=  advants.unsqueeze(1)[batch_index]
            actions_samples = torch.Tensor(actions)[batch_index]
            oldvalue_samples = old_values[batch_index].detach()
            loss, ratio = surrogate_loss(actor, advanes_samples, inputs, old_policy.detach(), action_samples, batch_index)
            values = critic(inputs) #state
            clipped_values = oldvalue_samples + torch.clamp(values - oldvalue_samples, - hp.clip.param, hp.clip_param)
            #clamp(input,min, max) -> input을 min max 사이값만 뽑아오기
            critic_loss1 = criterion(clipped_values, returns_samples) #MSELoss
            critic_loss2 = criterion(values, returns_samples)
            critic_loss = torch.max(critic_loss1, critic_loss2).mean()

            clipped_ratio = torch.clamp(ratio, 1.0 - hp.clip_param, 1.0 + ,hp.clip_param) #클립!
            cliped_loss = clipped_ratio * advants_samples
            actor_loss = -torch.min(loss, clipped_loss).mean() # 드디어!

            loss = actor_loss + 0.5*critic_loss #이건 어디서나온 식일까?
            critic_optim.zero_grad()
            loss.backward(retain_graph= True)
            critic_optim.step()

            actor_optim.zero_grad()
            loss.backward()
            actor_optim.step()
