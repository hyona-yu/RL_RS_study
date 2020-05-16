import numpy as np
import os
import time

class MonteCarlo:
    def __init__(self, state_array,action_array, discount_factor):
        self.state = np.array(state_array) #가능한 state가 유한하다고 가정
        self.action = np.array(action_array)# 가능한 action도 유한하다고 가정
        self.value ={} #(state,action)쌍에 대한 value 저장
        self.discount_factor= discount_factor
        self.action_state= []



    def epsilon_soft_on_policy(self,soft_policy, epochs, batch, epsilon = 0.5):
        policy = np.random(soft_policy)
        returns = np.array(len(self.state),len(self.action))
        while True:
            G = 0
            for e in epochs:
                next_state, next_action, next_reward=  step(next_action) #함수 생성 필요
                G = self.discount_factor * G + next_reward
                TF = 0
                for as in self.action_state:
                    if as == [next_state,next_action]:
                        TF=  1

                if !TF:
                    self.action_state.append(next_state, next_action)
                    self.value[next_state,next_action]= G.mean()
                    next_action = np.argmax(self.value[next_state])

                    # policy
