import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from math import sin, cos, atan2, pi
import math

s = 0.1
sd = 0.5

def plot_data(data_1, data_2, label_1, label_2, markersize_1=8, markersize_2=5):
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection ='3d')
    ax.axis('equal')
    if data_1 is not None:     
        x_p, y_p, z_p = data_1          #put z_p in this line and the next
        ax.plot(x_p, y_p, z_p, color='#336699', markersize=markersize_1, marker='o', linestyle=":", label=label_1)
    if data_2 is not None:
        x_q, y_q, z_p = data_2           #put z_p in this line and the next
        ax.plot(x_q, y_q, z_p, color='orangered', markersize=markersize_2, marker='o', linestyle=":", label=label_2)
    ax.legend()
    return ax

def draw_correspondences(P, Q, correspondences, ax):
    label_added = False
    for i, j in correspondences:
        x = [P[0, i], Q[0, j]]
        y = [P[1, i], Q[1, j]]
        z = [P[2, i], Q[2, j]]
        if not label_added:
            ax.plot(x, y, z, color='grey', label='correspondences')
            label_added = True
        else:
            ax.plot(x, y, z, color='grey')
    ax.legend()

########################### get correspondences #################################

def get_correspondence_indices(P, Q):
    psize = P.shape[1]
    qsize = Q.shape[1]
    correspondences = []
    for i in range(psize):
        p_point = P[:, i]
        mindist = sys.maxsize
        closest_q = -1
        for j in range(qsize):
            q_point = Q[:, j]
            dist = np.linalg.norm(q_point - p_point)
            if dist < mindist:
                mindist = dist
                closest_q = j
        correspondences.append((i, closest_q))
    return correspondences


def get_covariance(P,Q,correspondences,s):
    covar = np.zeros((3,3))
    sumdist = 0
    for i,j in correspondences:
        p_point = P[:, i]
        q_point = Q[:, j]
        sumdist += np.linalg.norm(q_point - p_point)
    m = sumdist/(P.shape[1])
    #print(m)
    for i,j in correspondences:
        p_point = P[:, [i]]
        q_point = Q[:, [j]]

        weight = weights(np.linalg.norm(Q[:, j] - P[:, i]),m,s)
        covar += weight * np.dot(q_point,p_point.T)
    # print("==========================")
    return covar

def weights(x,m,s):
    return (np.pi*s) * np.exp(-0.5*((x-m)/s)**2)
##############################################################

################### generate points ##########################
a = pi / 4
b = pi / 6
c = pi / 8
R_true = np.array([[cos(a)*cos(b), cos(a)*sin(b)*sin(c) - sin(a)*cos(c), cos(a)*sin(b)*cos(c) + sin(a)*sin(c)], 
                   [sin(a)*cos(b), sin(a)*sin(b)*sin(c) + cos(a)*cos(c), sin(a)*sin(b)*cos(c) - cos(a)*sin(c)],
                   [-sin(b),       cos(b)*sin(c),                        cos(b)*cos(c)                       ]])

t_true = np.array([[-20], [50], [30]])

num_points = 30
true_data = np.zeros((3, num_points))
true_data[0, :] = range(0, num_points)
true_data[1, :] = 0.2 * true_data[0, :] * np.cos(0.5 * true_data[0, :]) 
true_data[2, :] = 0.2 * true_data[0, :] * np.sin(0.5 * true_data[0, :]) 

moved_data = R_true.dot(true_data) + t_true

Q = true_data
P = moved_data

for i in range(P.shape[1]):
    P[:,i] += [np.random.normal(0,sd),np.random.normal(0,sd),np.random.normal(0,sd)]
    #print(P[:, i])

##############################################################
for i in range(20):
    centerp, centerq = np.array([P.mean(axis=1)]).T, np.array([Q.mean(axis=1)]).T
    P, Q = P-centerp, Q-centerq
    centerp, centerq = np.array([P.mean(axis=1)]).T, np.array([Q.mean(axis=1)]).T

    covar = get_covariance(P, Q, get_correspondence_indices(P,Q),s)

    print(covar)

    U, S, Vt = np.linalg.svd(covar)
    R = np.dot(U,Vt)
    t = centerq - np.dot(R,centerp)

    P = np.dot(R,P)+t



#############################


ax = plot_data(P, Q, "P: moved data", "Q: true data")
draw_correspondences(P, Q, get_correspondence_indices(P, Q), ax)
plt.show()