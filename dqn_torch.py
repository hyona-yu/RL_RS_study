#Catpole
# keras 버전 참조.
import sys
import gym
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
from collections import deque
#Q-value
class DQN_model(torch.nn.Module):
    def __init__(self,state_size, action_size):
        super(DQN_model, self).__init__()

        self.l1 = torch.nn.Linear(state_size, 32)
        self.relu = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(32, 32)
        self.l3 = torch.nn.Linear(32, action_size)
        torch.nn.init.kaiming_uniform_(self.l1.weight)
        torch.nn.init.kaiming_uniform_(self.l2.weight)
        torch.nn.init.kaiming_uniform_(self.l3.weight)
        #self.softmax = torch.nn.Softmax(dim =1)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        #out = self.softmax(out)

        return out

class DQN:
    def __init__(self, state_size, action_size):
        self.load_model = False
        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN 하이퍼파라미터
        self.discount_factor = 0.99
        self.lr = 1e-3
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000

        # 리플레이 메모리, 최대 크기 2000
        self.memory = deque(maxlen=2000)

        # 모델과 타깃 모델 생성
        self.model = DQN_model(self.state_size, self.action_size)
        self.target_model = DQN_model(self.state_size, self.action_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        self.loss =torch.nn.MSELoss()
        # 타깃 모델 초기화
        self.update_target_model()

        if self.load_model:
            self.model = torch.load('./models/cartpole_dqn.pt')
            self.model.load_state_dict(torch.load('./models/cartpole_dqn_state_dict.pt'))

    def update_target_model(self):
        with torch.no_grad():
            self.target_model.l1.weight = self.model.l1.weight
            self.target_model.l2.weight = self.model.l2.weight
            self.target_model.l3.weight = self.model.l3.weight
        torch.save(self.model, './models/cartpole_dqn.pt')
        torch.save(self.model.state_dict(), './models/cartpole_dqn_state_dict.pt')

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                self.model.eval()
                state = torch.FloatTensor(np.array(state))
                y = self.model(state).detach().numpy() #2차원
            return np.argmax(y[0])

    def replay_buffer(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))

    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        batch = random.sample(self.memory, self.batch_size)

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
            self.model.eval()
            self.target_model.eval()
            q_value = self.model(states)
            next_q_value = self.target_model(next_states)

        for i in range(self.batch_size):
            if terminals[i]:
                q_value[i][actions[i]] = rewards[i]
            else:
                q_value[i][actions[i]] = rewards[i] + self.discount_factor* np.amax(next_q_value[i].detach().numpy())
        self.model.train()
        #print(next_q_value[i].detach().numpy())
        self.optimizer.zero_grad()
        h_x = self.model(states)
        cost = self.loss(h_x,q_value)
        cost.backward()
        self.optimizer.step()

if __name__ == "__main__":
    torch.manual_seed(42)
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    episode = 200

    agent = DQN(state_size, action_size)
    scores = []

    for e in range(episode):
        terminal = False
        score = 0
        state = env.reset()
        state = state.reshape(1,-1)

        while not terminal:

            env.render()
            action =agent.get_action(state)
            next_state, reward, terminal, info =env.step(action)
            next_state = np.resize(next_state, (1,4))#next_state shpae 3
            reward = reward if not terminal or score == 499 else -100
            #print(action)
            agent.replay_buffer(state, action,reward, next_state, terminal)

            if len(agent.memory)>= agent.train_start:
                agent.train_model()

            score += reward
            state = next_state
            if terminal:
                agent.update_target_model()
                score = score if score ==500 else score + 100

                print("episode",e,"score",score,"len",len(agent.memory), 'epsilon',agent.epsilon)

                env.close()
