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

        self.Pi_log = np.log(self.Pi)  # (N, 1)
        self.A_log = np.log(self.A)  # (N, N)
        self.B_log = np.log(self.B)  # (N, M)

    def log_forward(self, obs_sequence):
        T=len(obs_sequence)#obs_sequence: (T, 1)
        alpha_log=np.zeros([T,self.N])
        #init alpha
        alpha_log[0] = self.Pi_log+self.B_log[:, obs_sequence[1]]

        #induc alpha
        for t in range(0,T-1):
            alpha_log[t+1]=np.log(np.sum(np.exp(alpha_log[t]+self.A_log),axis=1))+self.B_log[:, obs_sequence[t+1]]
        self.alpha_log=alpha_log
        # alpha /= np.sum(alpha, axis=1)
        pass

    def log_backward(self,obs_sequence):
        T = len(obs_sequence)
        beta_log=np.zeros([T,self.N])
        #init beta
        beta_log[-1]=np.log(1)
        #induc beta
        for t in range(T-2,-1,-1):
            beta_log[t]=np.sum(np.exp(self.A_log+(self.B_log[:,obs_sequence[t+1]]+beta_log[t+1]).reshape(-1,1)), axis=0)

        self.beta_log=beta_log
        pass

    def baum_welch(self, obs_sequence_list, max_iter=100):
        T=len(obs_sequence_list)
        #E-step
        eps_log=np.zeros([self.N,self.N,T])
        gamma_log=np.zeros([self.N, T])

        for iter in range(max_iter):

            # CAL eps
            for t in range(T-1):
                eps_log[:,:,t]=self.A_log+self.alpha_log[t]+(self.beta_log[t+1]+self.B_log[:,obs_sequence_list[t+1]]).reshape(-1,1)
            norm_eps=np.log(np.sum(np.exp(eps_log),axis=2))
            eps_log-=norm_eps
            #cal gamma
            gamma_log=np.log(np.sum(np.exp(eps_log),axis=1))


            #M-step
            gamma=np.exp(gamma_log)
            eps=np.exp(eps_log)
            self.Pi_log=gamma_log[:,0]
            self.A_log=np.log(np.sum(eps[:,:,:-1],axis=2))-np.log(np.sum(gamma[:,:-1],axis=1))
            for vk in range(np.shape(self.B_log)[1]):
                self.B_log[:,vk]=np.log(np.sum(gamma[:,obs_sequence_list==vk],axis=1))-np.log(np.sum(gamma,axis=1))

            crit=np.linalg.norm(self.A_log)
            print("diference with last iteration: %f"%crit)

    def initPara(self):



# def data2observation():
#
# 	return ob




if __name__=='__main__':
	

	traindata
	hhm=HHM()