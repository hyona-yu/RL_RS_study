#미완성
import numpy as np
import math
import itertools #싱기!
import multiprocessing as mp #와대박 이게 있다고????
# import os
# import pandas as pd
# import torch
## Example Chapter 4 - car rental
##policy evaluate & imporvement  ==> policy iteration
reward_dollar = 10
exchange_dollar= 2
max_car= 20
max_exchange = 5

class Policy_iteration:
    def __init__(self, pi= 0.1, discount_factor= 0.8):
        self.action = np.arange(0, max_car + 1)
        self.value = np.zeros((max_car+1, max_car+1))
        self.pi = pi
        self.discount_factor = discount_factor
        self.policy = np.zeros(self.values.shape)
        self.state = []
        #print(self.action)

    def poisson(self,lam, n):
        return math.exp(-lam)*math.pow(lam,n)/math.factorial(n)

    def get_value(self,state, policy, value):
        if
        return 0
    def policy_evaluation(self):
        delta = 0
        while True:
            last_value = np.deepcopy(self.value)
            # states = ((i, j)for i,j in itertools.product(np.arange(max_cars+1),np.arange(max_cars+1)))
            # #product : 곱집합 구하기 쌉가능
            # with mp.Pool(4) as p:  #pool은 한번애 process만큼 처리한다는 . 시간효율!
            #     cook = partial(policy, self.value)
            #     ## partial 첫번째 인자인 함수에 나머지 인자 인수값이 들어감.
            #     #이거도 싱기
            #    print('cook: ',cook)
            #    results = p.map(cook, states)
            #    print('어떻게 mapping됐을까',results)
            #for v,i,j in results:
            #    new_value[i,j] = k

            for i in range(max_car +1):
                for j in range(max_car + 1):
                    self.value[i][j] = get_value([i,j], self.policy[i,j],self.value)
            delta = (np.abs(last_value - value).sum(), delta).max()

            if delta < self.pi:
                break
    def policy_action(self, i, j):


        return prob, new_s, reward
    def policy_imporvement(self):
        TF= True
        new_policy = np.deepcopy(self.policy)
        new_action = np.zeros((mas_cars+1, max_cars+1))

        for i in range(max_car +1):
            for j in range(max_car + 1):
                last_action = self.policy[i,j]
                prob, new_s, reward =policy_action(i,j)
                self.policy[i,j] = argmax((prob*(reward + discount_factor*value[new_s])).sum())
                if last_action != self.policy[i,j] :
                    TF= False
        print('policy is changed:',np.bool(TF))

if __name__ == '__main__':
    policy = Policy_iteration()
    policy.test()
