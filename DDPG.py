import torch
import numpy as np
import time
import random
import argparse
import copy
import gym
import os
from collections import deque

gym.undo_logger_setup()
class Actor(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        self.l1 = torch.nn.Linear(input_size, 64)
        self.l2 = torch.nn.Linear(64,64)
        self.l3 = torch.nn.Linear(64,output_size)
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(l1.weight)
        torch.nn.init.xavier_normal_(l2.weight)
        torch.nn.init.xavier_normal_(l3.weight)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.tanh(out)
        return out

class Critic(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Critic, self).__init__()
        self.l1 = torch.nn.Linear(input_size, 64)
        self.l2 = torch.nn.Linear(64,64)
        self.l3 = torch.nn.Linear(64,output_size)
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(l1.weight)
        torch.nn.init.xavier_normal_(l2.weight)
        torch.nn.init.xavier_normal_(l3.weight)

    def forward(self, x, actions):
        out = self.l1(x)
        out = self.relu(out)
        out = torch.cat((out, actions),1)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.tanh(out)
        return out
def soft_update(target, main, tau):
    for target_param, param in zip(target.parameters(), main.parameters()):
        target_param.data.copy_(target_param.data*(1.0 - tau) + param.data*tau)
def hard_update(target, main, tau):
    for target_param, param in zip(target.parameters(), main.parameters()):
        target_param.data.copy_(param.data)

def obs2state(observation):
    """Converts observation dictionary to state tensor"""
    l1 = [val.tolist() for val in list(observation.values())]
    l2 = []
    for sublist in l1:
        try:
            l2.extend(sublist)
        except:
            l2.append(sublist)
    return torch.FloatTensor(l2).view(1, -1)


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.limit = buffer_size
        self.data = deque(maxlen = self.limit)
    def sample_batch(self, batch_size):
        if len(self.data) < batch_size:
            print('batch size error')
            return None
        else:
            batch = random.sample(self.data, batch_size) #sample(set , 샘플 수 ) 안겹침
            state_t = [b[0] for b in batch]
            action_t = [b[1] for b in batch]
            state_n = [b[2] for b in batch]
            reward = [b[3] for b in batch]
            terminal = [b[4] for b in batch]
        return state_t, action_t, state_n, reward, terminal
    def append(self, element):
        self.data.append(element)

class DDPG(object):
    def __init__(self, nb_states, nb_actions, args, env):
        if args.seed > 0:
            self.seed(args.seed)
        self.env = env
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.batch_size = 100
        self.tau = 0.8
        self.discount = 0.99
        self.epsilon = 0.5
        self.checkpoint_dir = './checkpoints/manipulator/'
        self.actor = Actor(self.nb_states, self.nb_actions)
        self.actor_target = Actor(self.nb_states, self.nb_actions)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr = 1e-2)

        self.critic = Critic(self.nb_states, self.nb_actions)
        self.critic_target = Critic(self.nb_states, self.nb_actions)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr = 1e-2)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)
        self.replay_buffer = ReplayBuffer(buffer_size = 100000)

    def get_action(self, state, action_noise):
        noise = self.epsilon * torch.autograd.Variable(torch.FloatTensor(action_noise()), volatile = True)
        #self.actor.eval()
        action = self.actor(state)
        return action + noise

    def get_Q_target(self, state_n_batch, reward_t_batch, terminal_batch):
        target_batch = torch.FloatTensor(reward_t_batch)
        state_n_batch = torch.cat(state_n_batch)
        next_action_batch = self.actor_target(state_n_batch)
        next_action_batch.volatile = True
        Q = self.critic_target(state_n_batch,next_action_batch)
        terminal = torch.ByteTensor(tuple(map(lambda s: s !=True, terminal_batch)))
        terminal = self.discount * terminal.type(torch.FloatTensor)
        target_batch += terminal * Q.squeeze().data

        return torch.autograd.Variable(target_batch, volatile = False)


    def train(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        print('Training!')

        for i in range(1000):
            time_step = self.env.reset()
            reward = 0

            while not time_step.last():
                state_t = torch.autograd.Variable(obs2state(time_step.observation), volatile = True)
                #자동미분 ! 계산하려면 Variable.backward()하면 됨
                self.actor.eval() #change model mode
                action = self.get_action(state_t)
                state_t.volatile = False
                action.volatile = False
                self.actor.train()

                #time_step
                time_step = sefl.env.step(action.data)
                next_state = torch.autograd.Variable(obs2state(time_step.observation), volatile = True)
                time_reward = time_step.reward
                reward += time_reward
                terminal = time_step.last()

                self.replay_buffer.append((state_t, action, next_state, time_reward, terminal))

                if len(self.replay_buffer) >= 50:
                    state_t_batch, action_batch, state_n_batch, reward_t_batch, terminal_batch = self.replay_buffer.sample_batch(self.batch_size)
                    state_t_batch = torch.cat(state_t_batch) # cat 여기서 굳이 해줘야하나?
                    action_batch = torch.cat(action_batch)
                    pred_batch = self.critic(state_t_batch, action_batch)
                    target_batch = self.get_Q_target(state_n_batch,reward_t_batch, terminal_batch)
                #Critic - Minimize loss
                self.critic_optim.zero_grad()
                critic_loss = self.torch.MSELoss(pred_batch, target_batch)
                critic_loss.backward()
                self.critic_optim.step()
                print('Critic Loss {}'.format(critic_loss))

                #Actor - Maximize E
                self.actor_optim.zero_grad()
                actor_loss = self.torch.MSELoss(pred_batch, target_batch)
                actor_loss.backward()
                self.actor_optim.step()

                soft_update(self.actor_target, self.actor, self.tau)
                soft_update(self.critic_target, self.critic, self.tau)
            if i %20 ==0:
                print(i)

if __name__ == "__main__":
    env = gym.make(args.env, **kwargs)
