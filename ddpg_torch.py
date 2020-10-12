#Continuous car
#https://arxiv.org/pdf/1509.02971.pdf
import sys
import gym
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
from collections import deque
from torch.autograd import Variable
import time

class Actor(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        self.l1 = torch.nn.Linear(input_size, 32)
        self.l2 = torch.nn.Linear(32,32)
        self.l3 = torch.nn.Linear(32,output_size)
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.batch_norm = torch.nn.BatchNorm1d(input_size)
        torch.nn.init.xavier_normal_(self.l1.weight)
        torch.nn.init.xavier_normal_(self.l2.weight)
        torch.nn.init.xavier_normal_(self.l3.weight)

    def forward(self, x):
        #print(x.shape)
        x = x.view(-1, 2)
        #print(x.shape)
        out = self.batch_norm(x)
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
        self.l1 = torch.nn.Linear(input_size, 16)
        self.l2 = torch.nn.Linear(16 + 1,64+ 1)
        self.l3 = torch.nn.Linear(64+1,output_size)
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.batch_norm = torch.nn.BatchNorm1d(input_size)

        torch.nn.init.xavier_normal_(self.l1.weight)
        torch.nn.init.xavier_normal_(self.l2.weight)
        torch.nn.init.xavier_normal_(self.l3.weight)

    def forward(self, x, actions):
        out = self.batch_norm(x)
        out = self.l1(x)
        out = self.relu(out)
        out = torch.cat((out, actions),1)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.tanh(out)
        return out

class DDPG():
    def __init__(self,state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = 1e-3
        self.epsilon = 0.99
        self.discount_factor = 0.99
        self.tau = 0.5
        self.batch_size = 64
        self.load_model = False
        self.actor = Actor(self.state_size, self.action_size)
        self.actor_target = Actor(self.state_size, self.action_size)#Q'
        self.critic = Critic(self.state_size, self.action_size)#Q fcn
        self.critic_target = Critic(self.state_size, self.action_size)# u'
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr = self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr= self.lr)

        self.replay_buffer = deque(maxlen = 10000)

        if self.load_model:
            self.actor= torch.load('./models/mountain_ddpg_actor.pt')
            self.actor.load_state_dict = torch.load('./models/mountain_ddpg_actor_state.pt')
            self.critic = torch.load('./models/mountain_ddpg_critic.pt')
            self.critic.load_state_dict = torch.load('./models/mountain_ddpg_critic_state.pt')


    def soft_target_update(self, model, target):# supervised learning에 가깝게 action-value fcn learning 조절
        for target_param, param in zip(target.parameters(), model.parameters()):
            target_param.data = self.tau * param.data + (1-self.tau)* target_param.data
            #data : parameter tensor
    def target_update(self,model, target):
        for target_param, param in zip(target.parameters(), model.parameters()):
            target_param.data = param.data

    def make_replay_buffer(self, state, action, reward, next_state, terminal):
        self.replay_buffer.append((state, action, reward, next_state, terminal))



    def get_action(self, state, noise = 0.01): #배열형태로 반환할 수 있
        #print(noise)
        if np.random.rand() <= self.epsilon:
            return [random.randrange(-1,1)]

        else:
            with torch.no_grad():
                self.actor.eval()
                state = torch.FloatTensor(np.array(state))
                y = self.actor(state)[0].detach().numpy() + noise #2차원

            return y


    def train(self):
        if self.epsilon > 0.01:
            self.epsilon *= 0.99

        batch = random.sample(self.replay_buffer, self.batch_size)
        actions, terminals, rewards = [], [], []#, []
        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        for i in range(self.batch_size):
            #states.append(batch[i][0])
            actions.append(batch[i][1])
            rewards.append(batch[i][2])
            terminals.append(batch[i][-1])
            #next_states.append(batch[i][3])
            states[i] = batch[i][0]
            next_states[i] = batch[i][3]
        states= torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))

        with torch.no_grad():
            self.actor.eval()
            self.actor_target.eval()
            self.critic.eval()
            self.critic_target.eval()

            y = rewards + self.discount_factor * self.critic_target(states, self.actor_target(states)).detach().numpy()

        self.update_critic(y, states, actions)
        self.update_actor(states, actions)
        self.soft_target_update(self.actor, self.actor_target)
        self.soft_target_update(self.critic, self.critic_target)


    def save_model(self):
        torch.save(self.actor, './models/mountain_ddpg_actor.pt')
        torch.save(self.actor.state_dict(), './models/mountain_ddpg_actor_state.pt')
        torch.save(self.critic, './models/mountain_ddpg_critic.pt')
        torch.save(self.critic.state_dict(), './models/mountain_ddpg_critic_state.pt')

    def update_critic(self, y,states, actions):
        self.critic.train()
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        y = torch.FloatTensor(y)
        h_x = self.critic(states,actions)
        #loss = torch.nn.MSELoss()(h_x,y)
        loss = (torch.pow(torch.FloatTensor(y) - self.critic(states, actions), 2))/self.batch_size#,requires_grad = True)
        self.critic_optim.zero_grad()
        loss.sum().backward()
        self.critic_optim.step()
        #print('critic loss', loss.mean())

    def update_actor(self,states, actions): #policy gradient
        self.critic.eval()
        self.actor.train()
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()
        #print('actor loss', loss.mean())







if __name__ == "__main__":
    torch.manual_seed(42)
    env = gym.make('MountainCarContinuous-v0')
    action_size = env.action_space.shape[-1] #action -1~1
    #print(action_size)
    state_size = env.observation_space.shape[-1]
    agent = DDPG(state_size, action_size)
    episode = 200
    #print(env.action_space)# Box(1,)
    #print(env.observation_space)
    #state[0] = pos, state[1] = vel
    #Critic: minimize loss, Actor:maximize E
    #print(env.step([0.1]))
    #step return : (arr, float, bool, dict) 이렇게 4개
    for e in range(episode):
        terminal = False
        state = env.reset()
        state = state.reshape(1,-1)
        score = 0
        start_time = time.time()
        while not terminal:
            env.render()
            noise = np.random.normal(0, 0.1)
            action = agent.get_action(state, noise)
            observation, reward, terminal, _ = env.step(action)#action은 arr 형태로                        reward = reward if not terminal
            # if abs(observation[0] - env.goal_position) <= 0.03:
            #     reward +=50

            agent.make_replay_buffer(state, action, reward, observation ,terminal)
            if len(agent.replay_buffer) > agent.batch_size:
                agent.train()

            score += reward
            if terminal and abs(state[0] - env.goal_position) <= 0.03:
                print("episode", e, "score", score, "time",  time.time()-start_time)
                # print(state[0], state[1])
                # print(observation[0], observation[1])
                agent.save_model()
                env.close()
            else:
                terminal = False
            state = observation
