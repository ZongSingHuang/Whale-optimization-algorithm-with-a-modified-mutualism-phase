# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 21:59:58 2020

@author: ZongSing_NB

https://doi.org/10.1016/j.cie.2020.107086
"""

import numpy as np
import matplotlib.pyplot as plt

class WOAmM():
    def __init__(self, fitness, D=30, P=20, G=500, ub=1, lb=0, 
                 b=1, a_max=2, a_min=0, a2_max=-1, a2_min=-2, l_max=1, l_min=-1):
        self.fitness = fitness
        self.D = D
        self.P = P
        self.G = G
        self.ub = ub*np.ones([self.P, self.D])
        self.lb = lb*np.ones([self.P, self.D])
        self.a_max = a_max
        self.a_min = a_min
        self.a2_max = a2_max
        self.a2_min = a2_min
        self.l_max = l_max
        self.l_min = l_min
        self.b = b
        
        self.F = np.zeros([self.P])
        self.gbest_X = np.zeros([self.D])
        self.gbest_F = np.inf
        self.loss_curve = np.zeros(self.G)
        
        
    def opt(self):
        # 初始化
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=[self.P, self.D])
        
        # 迭代
        for g in range(self.G):
            # SOS
            for i in range(self.P):
                menu = np.arange(self.P).tolist()
                menu.remove(i)
                m, n = np.random.choice(menu, 2, replace=False)
                F_old1, F_old2, F_old3 = self.fitness(self.X[i]), self.fitness(self.X[m]), self.fitness(self.X[n])
                if F_old2<F_old3:
                    MV = (self.X[i]+self.X[m])/2
                    
                    r1 = np.random.uniform()
                    BF1 = np.random.choice([1, 2], 1)
                    X_new1 = self.X[i] + r1*(self.X[m]-MV*BF1)
                    X_new1 = np.clip(X_new1, self.lb[0], self.ub[0])
                    F_new1 = self.fitness(X_new1)
                    
                    r2 = np.random.uniform()
                    BF2 = np.random.choice([1, 2], 1)
                    X_new2 = self.X[n] + r1*(self.X[m]-MV*BF2)
                    X_new2 = np.clip(X_new2, self.lb[0], self.ub[0])
                    F_new2 = self.fitness(X_new2)
                    
                    if F_new1<F_old1:
                        self.X[i] = X_new1.copy()
                        self.F[i] = F_new1.copy()
                    if F_new2<F_old2:
                        self.X[m] = X_new2.copy()
                        self.F[m] = F_new2.copy()
                else:
                    MV = (self.X[i]+self.X[n])/2
                    
                    r1 = np.random.uniform()
                    BF1 = np.random.choice([1, 2], 1)
                    X_new1 = self.X[i] + r1*(self.X[n]-MV*BF1)
                    X_new1 = np.clip(X_new1, self.lb[0], self.ub[0])
                    F_new1 = self.fitness(X_new1)
                    
                    r2 = np.random.uniform()
                    BF2 = np.random.choice([1, 2], 1)
                    X_new3 = self.X[n] + r1*(self.X[n]-MV*BF2)
                    X_new3 = np.clip(X_new3, self.lb[0], self.ub[0])
                    F_new3 = self.fitness(X_new3)
                    
                    if F_new1<F_old1:
                        self.X[i] = X_new1.copy()
                        self.F[i] = F_new1.copy()
                    if F_new3<F_old3:
                        self.X[n] = X_new3.copy()
                        self.F[n] = F_new3.copy()
            
            # # 適應值計算
            # F = self.fitness(self.X)
            
            # 更新最佳解
            if self.F.min() < self.gbest_F:
                idx = self.F.argmin()
                self.gbest_X = self.X[idx].copy()
                self.gbest_F = self.F.min().copy()
            
            # 收斂曲線
            self.loss_curve[g] = self.gbest_F
            
            # 更新
            a = self.a_max - (self.a_max-self.a_min)*(g/self.G)
            a2 = self.a2_max - (self.a2_max-self.a2_min)*(g/self.G)
            
            for i in range(self.P):
                p = np.random.uniform()
                r1 = np.random.uniform()
                r2 = np.random.uniform()
                r3 = np.random.uniform()
                A = 2*a*r1 - a #(2.3)
                C = 2*r2 # (2.4)
                l = (a2-1)*r3 + 1 # (???)
                
                if p>0.5:
                    D = np.abs(self.gbest_X - self.X[i, :])
                    self.X[i, :] = D*np.exp(self.b*l)*np.cos(2*np.pi*l)+self.gbest_X # (2.5)
                else:
                    if np.abs(A)<1:
                        D = np.abs(C*self.gbest_X - self.X[i, :]) # (2.1)
                        self.X[i, :] = self.gbest_X - A*D # (2.2)
                    else:
                        X_rand = self.X[np.random.randint(low=0, high=self.P, size=self.D), :]
                        X_rand = np.diag(X_rand).copy()
                        D = np.abs(C*X_rand - self.X[i, :]) # (2.7)
                        self.X[i, :] = X_rand - A*D # (2.8)
            
            # 邊界處理
            self.X = np.clip(self.X, self.lb, self.ub)
        
    def plot_curve(self):
        plt.figure()
        plt.title('loss curve ['+str(round(self.gBest_curve[-1], 3))+']')
        plt.plot(self.gBest_curve, label='loss')
        plt.grid()
        plt.legend()
        plt.show()
            