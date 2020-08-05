#!/usr/bin/env python
'''
******************
created by: Tiantian Gong
ver: Apr 08, 2020
******************
'''

'''
input - alpha: undercutter mining power
        beta: honest mining power
        gamma: prize proportion
        winning depth d: winning length, eg 6 for current Bitcoin system
        starting point (m,n): m,n are number of blocks on two chains; we use m for main chain, n for fork
        finish line: by default the game finishes when one chain arrives at the winning depth

output - often return series of size d, but the actual size depends on the starting point

resp4 -> stay rational
p1 -> main chain wins
'''
import sys
from collections import defaultdict 
import math
import numpy as np
import scipy.integrate as integrate

def mnProduct(M= 4, N= 4, rate1= 0, rate2= 0):
    pr = integrate.dblquad(lambda x,y: math.pow(rate1, M+1)*math.exp(-rate1*x)*math.pow(x, M)*
    math.pow(rate2, N+1)*math.exp(-rate2*y)*math.pow(y, N)/math.factorial(M)/math.factorial(N), 0, np.inf, lambda x: 0, lambda x: x)
    return pr[0]

class StateGraph:
    def __init__(self, vertices, alpha, beta, gamma, d):
        self.V = vertices
        self.graph = defaultdict(list)
        self.return_series = [0]*6 # return series for the undercutter
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.depth = d+1
        self.src = 0
        self.finishline = []
        self.signal = defaultdict(bool) # indicate where honest miners are, 0 - main chian, 1 - fork
        self.signal2 = defaultdict(bool) # indicate where the rational miners are, 0 - main chian, 1 - fork
        # states
        self.LPoF = defaultdict(float)
        self.ratef = defaultdict(float)
        self.ratem = defaultdict(float)
        self.PoF = defaultdict(float)
        self.shift = defaultdict(float)
        self.roundRet = defaultdict(float)
        # initialization 
        # self.LPoF[0] = 0
        # self.ratef[0] = 0
        # self.ratem[0] = 1/10
        # self.PoF[0] = alpha
        # self.shift[0] = 0

    def addEdge(self, u, v):
        self.graph[u].append(v)

    def genGraph(self):
        i = 0
        while i <= self.V - self.depth - 1:
            if i % self.depth == self.depth-1:
                i += 1
                continue
            self.addEdge(i, i + 1)
            self.addEdge(i, i + self.depth)
            i += 1
    
    def findAllPathsUtil(self, s, visited, path):
        visited[s] = True
        path.append(s)
        if s in self.finishline:
            # print(path)
            # calculate the probabolity of entering one path
            prob = 1
            prev = self.src
            for v in path:
                if v == prev + 1:
                    prob = prob * self.PoF[prev]
                elif v == prev + self.depth:
                    prob = prob * (1 - self.PoF[prev])
                prev = v
            # calculate return series
            prev = 0
            for v in path:
                if self.roundRet[v] > 0:
                    self.return_series[prev] += prob*self.roundRet[v]
                    prev += 1
        else:
            for vt in self.graph[s]:
                if vt % self.depth == (self.depth-1):
                    continue
                if visited[vt] == False:
                    if vt not in self.finishline:
                        # conditional on length of the path
                        if len(path) == 1: # just starting
                            if vt - path[0] == 1: # the fork extends by one
                                self.LPoF[vt] = self.alpha
                                # calculate shift
                                self.ratef[vt] = self.LPoF[vt]/10
                                self.ratem[vt] = (1 - self.LPoF[vt])/10
                                m = self.depth - 1 - vt%self.depth - 1  # remaining blocks for the fork
                                n = self.depth - 1 - vt//self.depth - 1
                                p = mnProduct(M = m, N = n, rate1 = self.ratef[vt], rate2 = self.ratem[vt])
                                # add the augmenting factor
                                p = min(p*(1+self.gamma),1)
                                self.gamma = 0
                                if p>=0.5:
                                    self.shift[vt] = self.alpha
                                    self.signal2[vt] = 1
                                else:
                                    self.shift[vt] = 0
                                    self.signal2[vt] = 0
                                # calculate the mining power on fork
                                self.PoF[vt] = self.alpha + self.shift[vt]
                                self.roundRet[vt] = 0
                                self.signal[vt] = 0
                            else: # main chain extends by one
                                self.LPoF[vt] = 0
                                self.PoF[vt] = self.alpha
                                self.roundRet[vt] = self.alpha/(1-self.PoF[s])
                                self.signal[vt] = 0
                                self.signal2[vt] = 0
                            # self.shift[vt] = 0
                        elif len(path) >= 2: # visited one vertex already

                            if vt - path[-1] == 1: # the fork extends by one
                                # calculate expected lpof      
                                if self.signal2[s] == 1: # avenger on fork
                                    self.LPoF[vt] = 2*self.alpha + self.beta*self.signal[s]  
                                else: 
                                    self.LPoF[vt] = self.alpha + self.beta*self.signal[s]                      
                                if s % self.depth == 0: # the first time the fork extends
                                    self.LPoF[vt] = self.alpha
                                if self.LPoF[vt] < 0 or self.LPoF[vt] > 1:
                                    print("LPoF mis-calculation: ", self.LPoF[vt])
                                    print(path[-1] - path[-2])
                                    raise
                                # calculate shift
                                self.ratef[vt] = self.LPoF[vt]/10
                                self.ratem[vt] = (1 - self.LPoF[vt])/10
                                m = self.depth - 1 - vt%self.depth - 1  # remaining blocks for the fork
                                n = self.depth - 1 - vt//self.depth - 1
                                p = mnProduct(M = m, N = n, rate1 = self.ratef[vt], rate2 = self.ratem[vt])
                                # add the augmenting factor
                                p = min(p*(1+self.gamma),1)
                                self.gamma = 0
                                if self.signal[s] == 1: # honest miners already on the fork
                                    self.shift[vt] = 0
                                    self.signal[vt] = 1
                                else:
                                    if m < n:
                                        self.shift[vt] = self.beta
                                        self.signal[vt] = 1 # update the signal
                                    else:
                                        self.shift[vt] = 0
                                        self.signal[vt] = 0
                                if self.signal2[s] == 0: # avenger on main
                                    if p>0.5:
                                        self.shift[vt] += self.alpha
                                        self.signal2[vt] = 1
                                    else:
                                        self.signal2[vt] = 0
                                else:
                                    self.signal2[vt] = 1
                                # calculate the mining poewr on fork
                                self.PoF[vt] = self.PoF[s] + self.shift[vt]
                                # if self.signal2[s] == 1:
                                #     self.roundRet[vt] = self.alpha/self.PoF[s]
                                # else:
                                self.roundRet[vt] = 0
                            else: # main chain extends by one
                                # calculate expected lpof 
                                self.LPoF[vt] = self.LPoF[s]
                                # calculate shift
                                self.ratef[vt] = self.LPoF[vt]/10
                                self.ratem[vt] = (1 - self.LPoF[vt])/10
                                m = self.depth - 1 - vt//self.depth - 1  # remaining blocks for main chain
                                n = self.depth - 1 - vt%self.depth - 1
                                p = mnProduct(M = m, N = n, rate1 = self.ratem[vt], rate2 = self.ratef[vt])
                                if self.signal[s] == 0: # honest miners currently on main chain
                                    self.shift[vt] = 0
                                    self.signal[vt] = 0
                                else:
                                    if m < n:
                                        self.shift[vt] = - self.beta
                                        self.signal[vt] = 0 # update the signal
                                    else:
                                        self.signal[vt] = 1
                                if self.signal2[s] == 1: # avenger on fork
                                    if p>0.5:
                                        self.shift[vt] -= self.alpha
                                        self.signal2[vt] = 0
                                    else:
                                        self.signal2[vt] = 1
                                else:
                                    self.signal2[vt] = 0 # avenger on main chain
                                if self.PoF[s] == 1:
                                    self.signal[vt] = 1
                                    self.signal2[vt] = 1
                                    self.shift[vt] = 0
                                # calculate the mining poewr on fork
                                self.PoF[vt] = self.PoF[s] + self.shift[vt]
                                if self.signal2[s] == 0: 
                                    self.roundRet[vt] = self.alpha/(1-self.PoF[s])
                                else:
                                    self.roundRet[vt] = 0
                        else:
                            print("Wrong length of path.")
                            raise
                    else: # arrive at finish line
                        if self.signal2[s] == 0:
                            self.roundRet[vt] = self.alpha/(1-self.PoF[s])
                        else:
                            self.roundRet[vt] = 0
                    self.findAllPathsUtil(vt, visited, path)
        path.pop()
        visited[s] = False

    def calculateProfits(self, m, n):
        visited = [False]*(self.V)
        path = []
        self.src = m*self.depth + n
        # initialization 
        self.LPoF[self.src] = 0
        self.ratef[self.src] = 0
        self.ratem[self.src] = 1/10
        self.PoF[self.src] = alpha
        self.shift[self.src] = 0
        self.signal[self.src] = m<n # start from main chain
        self.signal2[self.src] = 0
        self.roundRet[self.src] = self.alpha

        i = self.depth*(self.depth-1)
        while i < self.V:
            self.finishline.append(i)
            i += 1
        self.findAllPathsUtil(self.src, visited, path)
        return self.return_series

if __name__ == '__main__':
    alpha = 0.45
    beta = 0.1 # the responser has 45% mining power
    gamma = 0.9
    d = 6
    m = 1
    n = 0
    if len(sys.argv) == 7: # ignore if the complete input parameter set is not given
        alpha = float(sys.argv[1])
        beta = float(sys.argv[2])  # honest miners proportion
        d = int(sys.argv[3])
        m = int(sys.argv[4])
        n = int(sys.argv[5])
        gamma = float(sys.argv[6]) # prize proportion
    node_cnt = int(math.pow((d+1),2)-1) # we stop when one chain is longer, eg (6,6) with d=6 is not valid
    g = StateGraph(node_cnt, alpha, beta, gamma, d) 
    g.genGraph()
    return_series = g.calculateProfits(m, n)
    for i in return_series:
        print(i)