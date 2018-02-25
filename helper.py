import numpy as np
import os
import pandas as pd

# def preprocessData(data_in):
from ukf import processA,processW,caldQ

# 	return q_out

# def importData(path):
#     rawdata=pd.read_csv(path, sep='\t', header=None)
#
#     return q_out


def impData(path):
    sensitivity_A = 0.33  # unit:mv/g
    scale_A = 3.3 / 1023 / sensitivity_A
    Az_addbias =-1/scale_A

    sensitivity_W = 3.3   # unit:mv/deg/s
    scale_W = 3300 / 1023 / (sensitivity_W*180/np.pi)


    data = pd.read_csv(path, sep='\t', header=None).values.T.astype(float)
    # AW=data[1:,:]
    A,W,ts = data[1:4,:],data[4:,:],data[0:1,:]

    # A[0:1] *= -1
    A_bias=np.mean(A[:,:150],axis=1)
    A_bias[-1]=A_bias[-1]+Az_addbias
    # A[0:1] *= -1
    A=(A-A_bias.reshape(3,1))*scale_A
    # A[0:1] *= -1


    W_bias = np.mean(W[:, :200], axis=1)
    W = (W - W_bias.reshape(3,1)) * scale_W
    W[[0,1,2]]=W[[1,2,0]]

    return A,W,ts
    # return AW.T

#
# def














#
#
#
# def vec2quat(v):
#     '''
#     :param v: (3, n)
#     :return:(4, n)
#     '''
#     theta=np.sqrt(np.sum(v**2,axis=0).reshape(1,-1))
#     norm =np.linalg.norm(v,axis=0)
#     vec_unit=np.divide(v,norm)
#     q=np.zeros(np.array(list(v.shape))+[1,0])
#     q[0],q[1:]=np.cos(theta/2),vec_unit*np.sin(theta/2)
#     # q=vecNormorlize(q)
#     q[np.isnan(q)]=0
#     q[np.isinf(q)] = 0
#     return q





#
#
#
#
#
# def acc2rp(acc):
#     r = -np.arctan2(acc[1] , acc[2])
#     p=np.arctan2(acc[0],np.sqrt(acc[1]**2+acc[2]**2))
#     y=np.zeros_like(r)
#     return r,p,y
#
#
#
#
#
# def quatMulti(a,b):
#     '''
#     :param a: (4, 1)
#     :param b: (4, n)
#     :return:(4, n)
#     '''
#     a=a.astype(float)
#     a0, a1, a2, a3 = a[0],a[1],a[2],a[3]
#     b0, b1, b2, b3 = b[0],b[1],b[2],b[3]
#
#     q=np.zeros([4,max(a.shape[1],b.shape[1])])
#     q[0],q[1],q[2],q[3]=np.multiply(a0,b0) - np.multiply(a1,b1) - np.multiply(a2,b2) - np.multiply(a3,b3),\
#                         np.multiply(a0,b1) + np.multiply(b0,a1) + np.multiply(b3,a2) - np.multiply(b2,a3), \
#                         np.multiply(a0,b2) + np.multiply(b0,a2) + np.multiply(b1,a3) - np.multiply(b3,a1), \
#                         np.multiply(a0 , b3) + np.multiply(b0,a3) + np.multiply(b2,a1) - np.multiply(b1,a2)
#     # q=vecNormorlize(q)
#     # q[np.isnan(q)] = 0
#     # q[np.isinf(q)] = 0
#     return q
#
#
# def rpy2rot(r,p,y):
#     rot=np.zeros([3,3,len(r)])
#     a = y
#     b = p
#     c = r
#     rot[0,0]=np.multiply(np.cos(a),np.cos(b))
#     rot[0, 1] =np.multiply(np.multiply(np.cos(a),np.sin(b)),np.sin(c))-np.multiply(np.sin(a),np.cos(c))
#     rot[0, 2] =np.multiply(np.multiply(np.cos(a),np.sin(b)),np.cos(c))+np.multiply(np.sin(a),np.sin(c))
#     rot[1, 0] =np.multiply(np.sin(a),np.cos(b))
#     rot[1, 1] =np.multiply(np.multiply(np.sin(a),np.sin(b)),np.sin(c))+np.multiply(np.cos(a),np.cos(c))
#     rot[1, 2] =np.multiply(np.multiply(np.sin(a),np.sin(b)),np.cos(c))-np.multiply(np.cos(a),np.sin(c))
#     rot[2, 0] =-np.sin(b)
#     rot[2, 1] =np.multiply(np.cos(b),np.sin(c))
#     rot[2, 2] =np.multiply(np.cos(b),np.cos(c))
#     return rot
#
# def vecNormorlize(x):
#     '''
#     :param x: (4, 1)
#     :return: (4, 1)
#     '''
#     y=x/np.linalg.norm(x,axis=0)
#     y[np.isnan(y) + np.isinf(y)] = 0
#     return y
#
# def quat2matrix(qq):
#     '''
#
#     :param q:(1, 4)
#     :return:(3, 3, )
#     '''
#     # q=q.reshape(-1,4)
#     q=qq.copy()
#     rot=np.zeros([3,3,len(q)])
#     q=vecNormorlize(q.T).T
#     rot[0,0]=(q[0,0]**2 + q[0,1]**2)-(q[0,2]**2 + q[0,3]**2)
#     rot[0, 1] =2*(np.multiply(q[0,1],q[0,2]) + np.multiply(q[0,0],q[0,3]))
#     rot[0, 2] =2*(np.multiply(q[0,1],q[0,3]) - np.multiply(q[0,0],q[0,2]))
#     rot[1, 0] =2*(np.multiply(q[0,1],q[0,2]) - np.multiply(q[0,0],q[0,3]))
#     rot[1, 1] =(q[0,0]**2 - q[0,1]**2)+(q[0,2]**2 - q[0,3]**2)
#     rot[1, 2] =2*(np.multiply(q[0,2],q[0,3]) + np.multiply(q[0,0],q[0,1]))
#     rot[2, 0] =2*(np.multiply(q[0,1],q[0,3]) + np.multiply(q[0,0],q[0,2]))
#     rot[2, 1] =2*(np.multiply(q[0,2],q[0,3]) - np.multiply(q[0,0],q[0,1]))
#     rot[2, 2] =(q[0,0]**2 - q[0,1]**2)-(q[0,2]**2 - q[0,3]**2)
#     return rot
#
# def quaternion_conjugate(q):
#     '''
#     :param q: (4, N)
#     :return: (4, N)
#     '''
#     qq=q.copy()
#     qq[1:,:]*=-1
#     return qq
#
#
# def quat2vec(qq):
#     '''
#     :param q: (4, n)
#     :return:(3, n)
#     '''
#     q=qq.copy()
#     q=vecNormorlize(q)
#     theta=np.arccos(q[0])*2
#     sin=np.sin(theta)
#     vec=(q[1:4]/sin)*sin
#     vec[np.isnan(vec)+np.isinf(vec)]=0
#     # return q[1:4]
#     return vec
#
#
#
# def rot2rpy(rot):
#     '''
#     :param rot: (3, 3, n)
#     :return: (3, n)
#     '''
#     rpy=np.zeros([3,rot.shape[2]])
#     rpy[2]=np.arctan2(rot[1,0,:],rot[0,0,:])
#     rpy[1] = np.arctan2(-rot[2, 0, :], np.sqrt(rot[2, 1, :]**2+rot[2,2,:]**2))
#     rpy[0] = np.arctan2(rot[2, 1, :], rot[2, 2, :])
#     return rpy
#
# # def plotRots(a_rots,w_rots,ukf_rots,v_rots,t_imu,tgt):
# #     ts_imu=t_imu.T
# #     rpy_a = rot2rpy(a_rots)
# #     rpy_w = rot2rpy(w_rots)
# #     rpy_ukf = rot2rpy(ukf_rots)
# #     rpy_v = rot2rpy(v_rots)
# #
# #     f, ax = plt.subplots(3, sharex=True)
# #     ax[0].scatter(ts_imu,rpy_a[0],s=0.5,c='y',label='a')
# #     ax[0] .scatter(ts_imu,rpy_w[0],s=0.5,c= 'g', label='w')
# #     ax[0].scatter(ts_imu, rpy_ukf[0], s=0.5,c='r', label='ukf')
# #     ax[0] .scatter(tgt,rpy_v[0], s=0.5,c='b', label='v')
# #     ax[0].set_title('Roll')
# #     ax[0].legend(loc='upper right')
# #     ax[1].scatter(ts_imu,rpy_a[1], s=0.5,c='y', label='a')
# #     ax[1].scatter(ts_imu,rpy_w[1], s=0.5,c='g', label='w')
# #     ax[1].scatter(ts_imu, rpy_ukf[1], s=0.5,c='r', label='ukf')
# #     ax[1].scatter(tgt,rpy_v[1], s=0.5,c='b', label='v')
# #     ax[1].set_title('Pitch')
# #     ax[1].legend(loc='upper right')
# #     ax[2].scatter(ts_imu,rpy_a[2], s=0.5,c='y', label='a')
# #     ax[2].scatter(ts_imu,rpy_w[2], s=0.5,c='g', label='w')
# #     ax[2].scatter(ts_imu, rpy_ukf[2], s=0.5,c='r', label='ukf')
# #     ax[2].scatter(tgt,rpy_v[2], s=0.5,c='b', label='v')
# #     ax[2].set_title('Yaw')
# #     ax[2].legend(loc='upper right')
# #     plt.show()
# #     print(1)
# # def plotRots_nonVicon(a_rots,w_rots,ukf_rots,t_imu):
# #     ts_imu=t_imu.T
# #     rpy_a = rot2rpy(a_rots)
# #     rpy_w = rot2rpy(w_rots)
# #     rpy_ukf = rot2rpy(ukf_rots)
# #     # rpy_v = rot2rpy(v_rots)
# #
# #     f, ax = plt.subplots(3, sharex=True)
# #     ax[0].scatter(ts_imu,rpy_a[0],s=0.5,c='y',label='a')
# #     ax[0] .scatter(ts_imu,rpy_w[0], s=0.5,c='g', label='w')
# #     ax[0].scatter(ts_imu, rpy_ukf[0], s=0.5,c='r', label='ukf')
# #     # ax[0] .scatter(tgt,rpy_v[0],s=0.5, c='b', label='v')
# #     ax[0].set_title('Roll')
# #     ax[0].legend(loc='upper right')
# #     ax[1].scatter(ts_imu,rpy_a[1], s=0.5,c='y', label='a')
# #     ax[1].scatter(ts_imu,rpy_w[1],s=0.5, c='g', label='w')
# #     ax[1].scatter(ts_imu, rpy_ukf[1], s=0.5,c='r', label='ukf')
# #     # ax[1].scatter(tgt,rpy_v[1], s=0.5,c='b', label='v')
# #     ax[1].set_title('Pitch')
# #     ax[1].legend(loc='upper right')
# #     ax[2].scatter(ts_imu,rpy_a[2], s=0.5,c='y', label='a')
# #     ax[2].scatter(ts_imu,rpy_w[2], s=0.5,c='g', label='w')
# #     ax[2].scatter(ts_imu, rpy_ukf[2], s=0.5,c='r', label='ukf')
# #     # ax[2].scatter(tgt,rpy_v[2], s=0.5,c='b', label='v')
# #     ax[2].set_title('Yaw')
# #     ax[2].legend(loc='upper right')
# #     plt.show()
# #     print(1)
#
