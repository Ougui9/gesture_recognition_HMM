'''
File description: main file for HMM proj3
Author: Yilun Zhang
'''
import numpy as np
import os
from helper import impData
from ukf import exportUKF
from ukf import processA,processW,caldQ
from sklearn.cluster import KMeans

mode=0#0:train
trainset_folder='./train_data'

M=16#observation symbol number
N=10#num of hidden states


class HMM:
    def __init__(self, n_states, n_obs, Pi=None, A=None, B=None):
        self.N=n_states#N
        self.M=n_obs#M
        self.Pi=Pi# (N, 1)
        self.A=A # (N, N)
        self.B=B #(N, M)

        self.Pi_log = np.log(self.Pi)  # (N, 1)
        self.A_log = np.log(self.A)  # (N, N)
        self.B_log = np.log(self.B)  # (N, M)

    # def log_forward(self, obs_sequence):
    #     T=len(obs_sequence)#obs_sequence: (T, 1)
    #     alpha_log=np.zeros([T,self.N])
    #     #init alpha
    #     alpha_log[0] = self.Pi_log.T+self.B_log[:, obs_sequence[0]]
    #
    #     #induc alpha
    #     for t in range(0,T-1):
    #         if t==592:
    #             print(1)
    #         alpha_log[t+1]=np.log(np.sum(np.exp(alpha_log[t:t+1].T+self.A_log),axis=1))+self.B_log[:, obs_sequence[t+1]]
    #
    #     # alpha /= np.sum(alpha, axis=1)
    #     alpha_log[np.isinf(alpha_log)]=np.finfo(np.float64).eps
    #     self.alpha_log = alpha_log
    #     pass
    def forward(self, obs_sequence):
        T=len(obs_sequence)#obs_sequence: (T, 1)
        alpha=np.zeros([T,self.N])
        #init alpha
        alpha[0] = self.Pi.T*self.B[:, obs_sequence[0]]
        ct=np.zeros([T,1])
        #induc alpha
        for t in range(0,T-1):
            ans=alpha[t:t+1].dot(self.A)*self.B[:, obs_sequence[t+1]]
            ct[t,0]=1/np.sum(ans)
            alpha[t + 1]=ans*ct[t,0]
        # alpha /= np.sum(alpha, axis=1)
        alpha[np.isinf(alpha)]=np.finfo(np.float64).eps
        self.alpha = alpha
        self.ct=ct
        pass

    # def log_backward(self,obs_sequence):
    #     T = len(obs_sequence)
    #     beta_log=np.zeros([T,self.N])
    #     #init beta
    #     beta_log[-1]=np.log(1)
    #     #induc beta
    #     for t in range(T-2,-1,-1):
    #         beta_log[t]=np.log(np.sum(np.exp(self.A_log+(self.B_log[:,obs_sequence[t+1]]+beta_log[t+1]).reshape(-1,1)), axis=0))
    #     beta_log[np.isinf(beta_log)] = np.finfo(np.float64).eps
    #     self.beta_log=beta_log
    #
    #     pass
    def backward(self,obs_sequence):
        T = len(obs_sequence)
        beta=np.zeros([T,self.N])
        #init beta
        beta[-1]=1
        #induc beta
        for t in range(T-2,-1,-1):
            beta[t]=(self.A.dot(self.B[:,obs_sequence[t+1]].reshape(-1,1)).T*beta[t+1])[0]
            beta[t]*=self.ct[t]
        beta[np.isinf(beta)] = np.finfo(np.float64).eps
        self.beta=beta

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
            norm_eps=np.log(np.sum(np.sum(np.exp(eps_log),axis=1),axis=0))
            eps_log -= norm_eps.reshape(1, 1, T)
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


def initPara(N,M):
    Pi=np.random.rand(N,1)
    A=np.random.rand(N,N)
    B=np.random.rand(N,M)
    return A,B,Pi



# def data2observation():
#
# 	return ob

def obseqGeneration(q,M):
    kmeans=KMeans(n_clusters=M).fit(q)

    return kmeans.labels_



if __name__=='__main__':
    if mode==0:#train mode
        for file in os.listdir(trainset_folder):
            if file.endswith(".txt") and file[0]=='b':
                path_train=os.path.join(trainset_folder, file)
                break

        A,W,ts=impData(path_train)
        q=exportUKF(A,W,ts).T#(n, 4)
        obseq_labels=obseqGeneration(q,M)
        A,B,Pi=initPara(N,M)

        hmm=HMM(n_states=N,n_obs=M,Pi=Pi,A=A,B=B)
        # print(1)
        hmm.forward(obseq_labels)
        hmm.backward(obseq_labels)
        hmm.baum_welch(obseq_labels)
        print(1)

    # hhm=HHM()