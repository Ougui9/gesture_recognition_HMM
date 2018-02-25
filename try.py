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
import operator
import pandas as pd
# test_file='eight04.txt'
# train_class='eight'
mode=0#0:train
trainset_folder='./train_data'
testset_folder='./train_data'
M=30#observation symbol number
N=10#num of hidden states
mx_it=1000
max_trial=5
# scoreboard={'beat3.pkl':0, 'beat4.pkl':0, 'eight.pkl':0, 'inf.pkl':0,'wave.pkl':0,'circle.pkl':0}
proboard={}
# dict={}
class HMM:
    def __init__(self, n_states, n_obs,mdl=None, Pi=None, A=None, B=None):
        self.N=n_states#N
        self.M=n_obs#M
        self.Pi=Pi# (N, 1)
        self.A=A # (N, N)
        self.B=B #(N, M)
        self.mdl=mdl
    def forward(self, obs_sequence):
        self.T=len(obs_sequence)#obs_sequence: (T, 1)
        alpha=np.zeros([self.T,self.N])
        #init alpha
        alpha[0] = self.Pi.T*self.B[:, obs_sequence[0]]
        ct=np.zeros([self.T,1])
        ct[0] = 1. / np.maximum(1E-10, np.sum(alpha[0]))
        #induc alpha
        for t in range(0,self.T-1):
            ans=alpha[t:t+1].dot(self.A)*self.B[:, obs_sequence[t+1]]
            ans[np.isinf(ans)] = np.finfo(np.float64).eps
            ct[t+1,0]=1/np.maximum(np.sum(ans),1e-16)
            alpha[t + 1]=ans*ct[t,0]
            alpha[t + 1] = np.clip(alpha[t + 1], a_min=1E-100, a_max=1)
        # alpha /= np.sum(alpha, axis=1)
        alpha[np.isinf(alpha)]=np.finfo(np.float64).eps
        self.alpha = alpha
        self.ct=ct
        ct_log=np.log(ct[:-1])
        ct_log[np.isinf(ct_log)+np.isnan(ct_log)]=np.finfo(np.float64).eps
        pro=-np.sum(ct_log)
        # print(pro)
        return pro
        # # pass
        # pass

    def backward(self,obs_sequence):
        # self.T = len(obs_sequence)
        beta=np.zeros([self.T,self.N])
        #init beta
        beta[-1]=1
        beta[-1] *= self.ct[-1]
        #induc beta
        for t in range(self.T-2,-1,-1):
            beta[t]=self.A.dot((self.B[:,obs_sequence[t+1]]*beta[t+1]).reshape(-1,1))[:,0]
            beta[t]*=self.ct[t]
            # ans=(self.A.dot(self.B[:, obs_sequence[t + 1]].reshape(-1, 1)).T * beta[t + 1])[0]
            # beta[t] =ans/ np.sum(ans)
        # p = self.alpha * beta
        # p[np.isinf(p)] = np.finfo(np.float64).eps

        # beta[np.isinf(beta)] = np.finfo(np.float64).eps
        self.beta=beta
        # p=np.sum(np.log(np.sum(p,axis=1)))
        pass
        # return p
    def baum_welch(self, obs_sequence_list, max_iter=200):
        # T=len(obs_sequence_list)
        #E-step
        eps=np.zeros([self.N,self.N,self.T])
        gamma=np.zeros([self.N, self.T])
        old_pro=np.inf
        for iter in range(max_iter):
            # old_A=self.A
            # CAL eps
            for t in range(self.T-1):
                ans=self.alpha[t:t+1].T*self.A*(self.beta[t+1:t+2]*self.B[:,obs_sequence_list[t+1]])
                norm_eps=np.sum(ans)
                eps[:, :, t]=ans/norm_eps
            eps[np.isinf(eps)+np.isnan(eps)] = np.finfo(np.float64).eps
            #cal gamma
            gamma=np.sum(eps,axis=1)
            gamma[np.isinf(gamma) + np.isnan(gamma)] = np.finfo(np.float64).eps


            #M-step
            self.Pi=gamma[:,0]
            # self.Pi[np.isinf(self.Pi)] = np.finfo(np.float64).eps

            self.A=np.sum(eps[:,:,:-1],axis=2)/np.sum(gamma[:,:-1],axis=1)
            self.A[np.isinf(self.A)] = np.finfo(np.float64).eps
            for vk in range(np.shape(self.B)[1]):
                self.B[:,vk]=np.sum(gamma[:,obs_sequence_list==vk],axis=1)/np.sum(gamma,axis=1)
            self.B[np.isinf(self.B)] = np.finfo(np.float64).eps
            # print(1)
            pro=self.forward(obs_sequence_list)
            self.backward(obs_sequence_list)

            print("%s No.%d probability: %f"%(self.mdl,iter,pro))
            # old_pro
            # if iter%100==0:
                # print(1)
            if np.abs(old_pro-pro)<=1e-2 or iter==max_iter-1:
                dict['N']=self.N
                dict['M']=self.M
                dict['A']=self.A
                dict['B']=self.B
                dict['Pi']=self.Pi
                dict['alpha']=self.alpha
                dict['beta']=self.beta
                dict['ob']=obs_sequence_list
                pickle.dump(dict, open("%s.pkl"%self.mdl, "wb"))
                break
            old_pro=pro



def initPara(N,M):
    Pi=np.random.rand(N,1)
    Pi/=np.sum(Pi)
    A=np.random.rand(N,N)
    A/=np.maximum(1e-10,np.sum(A,axis=1))
    B=np.random.rand(N,M)
    B /= np.maximum(1e-10, np.sum(B, axis=0))
    return A,B,Pi



def obseqGeneration(q,M,cluster=None):
    if mode==0:
        kmeans=KMeans(n_clusters=M).fit(q)
        dict['cluster']=kmeans.cluster_centers_
    elif mode==1:
        kmeans=KMeans(n_clusters=M,init=cluster).fit(q)
    return kmeans.labels_



if __name__=='__main__':
    if mode==0:#train mode
        q=np.empty((0,4))
        for file in os.listdir(trainset_folder):
            if file.endswith(".txt") :
                dict={}
                path_train=os.path.join(trainset_folder, file)
                # break
                # AW=impData(path_train)
                A,W,ts=impData(path_train)
                q=np.append(q,exportUKF(A,W,ts).T,axis=0)#(n, 4)
                obseq_labels=obseqGeneration(q,M,dict)
                A,B,Pi=initPara(N,M)

                hmm=HMM(n_states=N,n_obs=M,mdl=file,Pi=Pi,A=A,B=B)
                # print(1)
                hmm.forward(obseq_labels)
                _=hmm.backward(obseq_labels)
                hmm.baum_welch(obseq_labels,max_iter=mx_it)

        # print(1)
    elif mode==1:



        df=pd.DataFrame()
        test_list=[]
        pred_list=[]
        for file in os.listdir(testset_folder):
            if file.endswith(".txt") :
                path_test=os.path.join(testset_folder, file)
                test_list.append(file)
                # break
                # AW=impData(path_test)
                A,W,ts=impData(path_test)
                q=exportUKF(A,W,ts).T#(n, 4)

                print(file)
                scoreboard = {'beat3': 0, 'beat4': 0, 'eight': 0, 'inf': 0, 'wave': 0,
                              'circle': 0}
                proboard={}
                for filePKL in os.listdir('./'):
                    if filePKL.endswith(".pkl"):
                        mdl=pickle.load(open( filePKL, "rb" ))
                        obseq_labels = obseqGeneration(q, M,mdl['cluster'])
                        hmm = HMM(n_states=mdl['N'], n_obs=mdl['M'], Pi=mdl['Pi'], A=mdl['A'], B=mdl['B'])
                        # pro=0
                        # for mm in range(max_trial):
                        proboard[filePKL]=hmm.forward(obseq_labels)
                        # print('test proba of %s on %s:%f'%(file,filePKL,proboard[filePKL]))
                rank=sorted(proboard.items(), key=operator.itemgetter(1))[::-1]
                for ii in range(5):
                    rank_sco=(4-ii)*0.1
                    if rank[ii][0][0:5]=='beat3':
                        scoreboard['beat3']+=1+rank_sco*1.1
                    elif rank[ii][0][0:5]=='beat4':
                        scoreboard['beat4'] += 1+rank_sco*1.1
                    elif rank[ii][0][0:5] == 'eight':
                        scoreboard['eight'] += 1+rank_sco
                    elif rank[ii][0][0:5]=='circl':
                        scoreboard['circle'] += 1+rank_sco
                    elif rank[ii][0][0]=='i':
                        scoreboard['inf'] += 1+rank_sco
                    elif rank[ii][0][0]=='w':
                        scoreboard['wave'] += 1+rank_sco/(4/18)
                print(scoreboard)
                df=df.append(scoreboard,ignore_index=True)
                pred_list.append(max(scoreboard.items(), key=operator.itemgetter(1))[0])
                print('Ground truth:%s \nPredicition:%s'%(file,max(scoreboard.items(), key=operator.itemgetter(1))[0]))
                print('\n')
                # scoreboard[win]+=1
                # print(scoreboard)
        df.insert(loc=0, column='GroundTruth', value=test_list)
        df.insert(loc=df.shape[1], column='Pred', value=pred_list)
        print(df)
        df.to_csv('./res/pred_result.csv', index=False)


    # hhm=HHM()