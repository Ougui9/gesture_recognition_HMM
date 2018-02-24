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
import pickle
train_class='beat3'
mode=1#0:train
trainset_folder='./train_data'
testset_folder='./train_data'
M=16#observation symbol number
N=10#num of hidden states
mx_it=1000
max_trial=5
class HMM:
    def __init__(self, n_states, n_obs, Pi=None, A=None, B=None):
        self.N=n_states#N
        self.M=n_obs#M
        self.Pi=Pi# (N, 1)
        self.A=A # (N, N)
        self.B=B #(N, M)

        # self.Pi_log = np.log(self.Pi)  # (N, 1)
        # self.A_log = np.log(self.A)  # (N, N)
        # self.B_log = np.log(self.B)  # (N, M)

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
        ct_log=np.log(ct[:-1])
        ct_log[np.isinf(ct_log)+np.isnan(ct_log)]=np.finfo(np.float64).eps
        pro=-np.sum(ct_log)
        print(pro)
        return pro
        # # pass
        # pass

    def backward(self,obs_sequence):
        T = len(obs_sequence)
        beta=np.zeros([T,self.N])
        #init beta
        beta[-1]=1
        #induc beta
        for t in range(T-2,-1,-1):
            beta[t]=(self.A.dot(self.B[:,obs_sequence[t+1]].reshape(-1,1)).T*beta[t+1])[0]
            beta[t]*=self.ct[t]
            # ans=(self.A.dot(self.B[:, obs_sequence[t + 1]].reshape(-1, 1)).T * beta[t + 1])[0]
            # beta[t] =ans/ np.sum(ans)
        # p = self.alpha * beta
        # p[np.isinf(p)] = np.finfo(np.float64).eps

        beta[np.isinf(beta)] = np.finfo(np.float64).eps
        self.beta=beta
        # p=np.sum(np.log(np.sum(p,axis=1)))
        pass
        # return p
    def baum_welch(self, obs_sequence_list, max_iter=2000):
        T=len(obs_sequence_list)
        #E-step
        eps=np.zeros([self.N,self.N,T])
        gamma=np.zeros([self.N, T])

        for iter in range(max_iter):
            old_A=self.A
            # CAL eps
            for t in range(T-1):
                ans=self.alpha[t:t+1].T*self.A*(self.beta[t+1]*self.B[:,obs_sequence_list[t+1]])
                norm_eps=np.sum(ans)
                eps[:, :, t]=ans/norm_eps
            eps[np.isinf(eps)+np.isnan(eps)] = np.finfo(np.float64).eps
            #cal gamma
            gamma=np.sum(eps,axis=1)
            gamma[np.isinf(gamma) + np.isnan(gamma)] = np.finfo(np.float64).eps


            #M-step
            self.Pi=gamma[:,0]
            self.Pi[np.isinf(self.Pi)] = np.finfo(np.float64).eps

            self.A=np.sum(eps[:,:,:-1],axis=2)/np.sum(gamma[:,:-1],axis=1)
            self.A[np.isinf(self.A)] = np.finfo(np.float64).eps
            for vk in range(np.shape(self.B)[1]):
                self.B[:,vk]=np.sum(gamma[:,obs_sequence_list==vk],axis=1)/np.sum(gamma,axis=1)
            self.B[np.isinf(self.B)] = np.finfo(np.float64).eps
            # print(1)
            pro=self.forward(obs_sequence_list)
            # self.backward(obs_sequence_list)

            print("No.%d probability: %f"%(iter,pro))
        dict={'N':self.N,'M':self.M,'A':self.A,'B':self.B,'Pi':self.Pi,'alpha':self.alpha,'beta':self.beta}
        pickle.dump(dict, open("%s.pkl"%train_class, "wb"))



def initPara(N,M):
    Pi=np.random.rand(N,1)
    A=np.random.rand(N,N)
    B=np.random.rand(N,M)
    return A,B,Pi



def obseqGeneration(q,M):
    kmeans=KMeans(n_clusters=M).fit(q)

    return kmeans.labels_



if __name__=='__main__':
    if mode==0:#train mode
        q=np.empty((0,4))
        for file in os.listdir(trainset_folder):
            if file.endswith(".txt") and file[0]==train_class[0]:
                path_train=os.path.join(trainset_folder, file)
                break

        A,W,ts=impData(path_train)
        q=np.append(q,exportUKF(A,W,ts).T,axis=0)#(n, 4)
        obseq_labels=obseqGeneration(q,M)
        A,B,Pi=initPara(N,M)

        hmm=HMM(n_states=N,n_obs=M,Pi=Pi,A=A,B=B)
        # print(1)
        hmm.forward(obseq_labels)
        _=hmm.backward(obseq_labels)
        hmm.baum_welch(obseq_labels,max_iter=mx_it)

        # print(1)
    elif mode==1:




        for file in os.listdir(testset_folder):
            if file.endswith(".txt") and file=='wave01.txt':
                path_test=os.path.join(testset_folder, file)
                # break

                A,W,ts=impData(path_test)
                q=exportUKF(A,W,ts).T#(n, 4)
                # obseq_labels=
                obseq_labels = np.zeros([max_trial, q.shape[0]],dtype=int)
                for mm in range(max_trial):


                    obseq_labels[mm]=obseqGeneration(q,M)
                for filePKL in os.listdir('./'):
                    if filePKL.endswith(".pkl"):
                        mdl=pickle.load(open( filePKL, "rb" ))
                        hmm = HMM(n_states=mdl['N'], n_obs=mdl['M'], Pi=mdl['Pi'], A=mdl['A'], B=mdl['B'])
                        pro=0
                        for mm in range(max_trial):
                            pro+=hmm.forward(obseq_labels[mm])
                        print('test proba of %s on %s:%f'%(file,filePKL,pro/max_trial))

    # hhm=HHM()