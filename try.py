'''
File description: main file for HMM proj3
Author: Yilun Zhang
'''
import numpy as np
import os


class HMM(n_states=None, n_obs=None, Pi=None, A=None, B=None):
    def __init__(self, n_states, n_obs, Pi=None, A=None, B=None):
        self.N=n_states#N
        self.M=n_obs#M
        self.Pi=Pi# (N, 1)
        self.A=A # (N, N)
        self.B=B #(N, M)


    def log_forward(self, obs_sequence):
        T=len(obs_sequence)#obs_sequence: (T, 1)
        alpha=np.zeros([T,self.N])
        #init alpha
        alpha[0] = np.multiply(self.Pi, self.B[:, obs_sequence[1]])

        #induc alpha
        for t in range(0,T-1):
            alpha[t+1]=alpha[t].dot(self.A).T*self.B[:, obs_sequence[t+1]]


    def log_backward(self,obs_sequence):
        T = len(obs_sequence)
        beta=np.zeros([T,self.N])
        #init beta
        beta[-1]=1
        #induc beta
        for t in range(T-2,-1,-1):
            beta[t]=self.A.dot((self.B[:,obs_sequence[t+1]]*beta[t+1]).reshape(-1,1))


    def baum_welch(self, obs_sequence_list, max_iter=100):



def data2observation():
	
	return ob




if __name__=='__main__':
	

	traindata
	hhm=HHM()