from scipy import io
import numpy as np
from helper_ukf import vec2quat, quatMulti,acc2rp,rpy2rot,vecNormorlize,quat2matrix,quaternion_conjugate,quat2vec,rot2rpy,rot2quat
from utils import averageQuaternions
from scipy.linalg import cholesky
# import matplotlib.pyplot as plt
from PIL import  Image
from scipy.ndimage.filters import gaussian_filter
# from rotplot import rotplot

def caldQ(w,t_i):
    '''
    :param w: 3,n,
    :param t_i: 1,n
    :return: dq: 4,1
    '''
    w_norm=np.linalg.norm(w,axis=0)
    dAngle = w_norm * t_i
    dAxis = w / w_norm
    dq=np.append(np.cos(dAngle / 2), np.multiply(dAxis, np.sin(dAngle / 2)),axis=0)
    dq[np.isnan(dq)+np.isinf(dq)]=0
    return dq


def processA(A):
    r,p,y=acc2rp(A)
    r=gaussian_filter(r,sigma=7)
    p = gaussian_filter(p, sigma=7)
    return rpy2rot(r,p,y) # return 3*3*n

def processW(W,ts):
    _,n_data=W.shape
    rots_W=np.zeros([3,3,n_data])
    ti= np.append([[0]], ts[:,1:] - ts[:,:-1],axis=1)
    dq=caldQ(W,ti)
    q0=np.array([1,0,0,0])
    for i in range(n_data):
        q0=vecNormorlize(quatMulti(q0.reshape(4,-1),dq[:,i].reshape(4,-1))).T
        rots_W[:,:,i]=quat2matrix(q0).reshape(3,3).T

    return rots_W








def impData():
    sensitivity_A = 0.33  # unit:mv/g
    scale_A = 3.3 / 1023 / sensitivity_A
    Az_addbias =-1/scale_A

    sensitivity_W = 3.3   # unit:mv/deg/s
    scale_W = 3300 / 1023 / (sensitivity_W*180/np.pi)


    data = io.loadmat('./imu/imuRaw%d.mat'%dataset)
    raw_vals, ts = data['vals'].astype(float), data['ts'].astype(float)
    A,W=raw_vals[:3,:],raw_vals[3:,:]
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


def gtData():


    data=io.loadmat('./vicon/viconRot%d.mat'%dataset)
    rots, ts_gt= data['rots'], data['ts']



    return rots,ts_gt



def iniPara():
    P = np.eye(6)  # State Cov

    Q = np.eye(6)
    Q[:3, :3] += np.ones(3)
    Q[3:, 3:] += np.ones(3)
    R = Q.copy()
    # Q *= 5e-8
    Q[:3, :3] *= 5e-8

    R[:3, :3] *= 2.8e-4
    R[3:, 3:] *= 10e-4
    return P,Q,R


def ukf(A,W_gyro,ts):
    #init para
    P,Q,R=iniPara()




    X=np.zeros([12,7])
    Y=X.copy()
    Z=np.zeros([12,6])
    Wp=np.zeros([12,6])





    n=6
    x=np.array([1,0,0,0,0,0,0],dtype=float)
    xk = x.copy()
    g = np.array([0, 0, 0, 1],dtype=float)

    t_interval=np.append([[0]], ts[:,1:] - ts[:,:-1],axis=1)

    _, n_measure=A.shape
    rot_q=np.zeros([n_measure,4])
    rots_ukf=np.zeros([3,3,n_measure])

    for i in range(n_measure):
        # print(i)

        #calulate dq
        omega=x[4:]
        dq=caldQ(omega.reshape(3,1),t_interval[0,i].reshape(1,-1))


        #extract sigma points
        # print(P)
        S=cholesky(P+Q) #S:(6, 6)

        W=np.sqrt(2*n)*S#W: (6, 6)
        W=np.append(W,-W,axis=1)#W=6*12
            #to Quater
        X[:,:4]=quatMulti(x[:4].reshape(4,-1),vec2quat(W[:3,:])).T
        X[:,4:]=W[3:,:].T  #X: (12, 7)
        # print(1)

        #Tansformation of sigma pts X
        Y[:,:4] = quatMulti(X[:, 0: 4].T, dq).T
        Y[:,4:] =X[:,4:]+omega
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(pos_sp[0], pos_sp[1], pos_sp[2], c='r', marker='o')

        # coor_projed+=1.

        qkk=averageQuaternions(Y[:,:4])
        # print(1)
        xk[:4],xk[4:]=qkk,np.mean(Y[:,4:],axis=0)

        Wp[:, 0: 3] = quat2vec(quatMulti(Y[:, 0: 4].T, quaternion_conjugate(xk[0: 4].reshape(4,-1)))).T
        Wp[:, 3: 6] =Y[:, 4:7]-xk[4:7].reshape(-1,3)
        Pk=Wp.T.dot(Wp)/(2*n)

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(pos_sp[0], pos_sp[1], pos_sp[2], c='r', marker='o')

        # coor_projed+=1.


        gp = quatMulti(Y[:, 0: 4].T, quatMulti(g.reshape(4,-1), quaternion_conjugate(Y[:, 0: 4].T))).T
        Z[:, :3] = gp[:, 1: 4]
        Z[:, 3:] = Y[:,4:7]
        z_mean=np.mean(Z,axis=0).reshape(1,-1)
        z=np.append(A[:, i],W_gyro[:, i]).reshape(1,-1)
        ve=z-z_mean


        Wz=Z-z_mean
        Pzz=Wz.T.dot(Wz)/(2*n)
        Pvv=Pzz+R


        Pxz=1/(2*n)*(Wp.T).dot(Wz)
        #kalman gain
        K_gain=Pxz.dot(np.linalg.inv(Pvv))
        #update P
        P=Pk-(K_gain.dot(Pvv)).dot(K_gain.T)

        #cal Kv
        K_vel=K_gain.dot(ve.T)

        #update state
        x[:4]=quatMulti(xk[:4].reshape(4,-1),vec2quat(K_vel[:3])).reshape(4)
        x[4:]=xk[4:]+K_vel[3:6].T
        #record q
        rot_q[i]=x[:4]
        rots_ukf[:,:,i]=quat2matrix(rot_q[i].reshape(-1,4)).reshape(3,3).T


    return rots_ukf

# def processGt(ts_imu,ts_gt):
#     ts_gt_ind=0
#     while(ts_gt[ts_gt_ind]<cam)

def impIm():
    data=io.loadmat('./cam/cam%d.mat'%dataset)
    Ims, ts = data['cam'], data['ts']
    return Ims,ts

def weightUKF(rots_W,rots_A,rots_ukf):
    rpy_ukf = rot2rpy(rots_ukf)
    rpy_ukf[-1] = rot2rpy(rots_W)[-1] * 0.6 + rpy_ukf[-1] * 0.3
    rpy_ukf[:2] = rot2rpy(rots_A)[:2] * 0.7 + rpy_ukf[:2] * 0.9 - rot2rpy(rots_W)[:2] * 0.6
    # rpy_ukf[:2] = rot2rpy(rots_A)[:2] * 1.3 + rpy_ukf[:2] * 0.3 - rot2rpy(rots_W)[:2] * 0.6
    rots_ukf = rpy2rot(rpy_ukf[0], rpy_ukf[1], rpy_ukf[2])
    return rots_ukf


def exportUKF(A,W,ts_imu):
    rots_A=processA(A)
    rots_W=processW(W,ts_imu)
    rots_ukf=ukf(A,W,ts_imu)
    rots_ukf=weightUKF(rots_W,rots_A,rots_ukf)

    ukf_q=rot2quat(rots_ukf)
    return ukf_q






