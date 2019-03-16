import numpy as np
import cv2
import random
import torch
from numpy import cos
from numpy import sin
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import os
# from math  import log
from numpy  import log2
from numpy  import pi
from numpy  import exp
from numpy  import sqrt

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
plt.rcParams.update({'font.size':13})#   xy axis size


args = (2, 3, 7, 8, 9, 10)
flag_for=1
# ----------------------------f1-------------------------
FN_INDEX = 0  # select which function to draw. Start from 0!
# H_INDEX=0
opt_index = 4  # set parameters---------------------------
x_init, y_init = -2, 1  # start point on the x-y plane
start_point = [x_init, y_init]
step = 0.012
discount = 0.9
epoch = 100
epsilon = 1e-5
yend = [0, 0]


# -----------------------------------------------------


# ---------------------f_trigonometric--------------------------------

#
# FN_INDEX = 1  # select which function to draw. Start from 0!
# # H_INDEX=0
# opt_index=4#set parameters---------------------------
# x_init, y_init = -2, 1 # start point on the x-y plane
# start_point = [x_init,y_init]
# step = 0.012
# discount = 0.9
# epoch=100
# epsilon=1e-5
# yend=[0,0]


# -----------------------------------------------------

# --------------------------f_rosenbrock---------------------------

# FN_INDEX =2  # select which function to draw. Start from 0!
# # H_INDEX=0
# opt_index=4#set parameters---------------------------
# x_init, y_init = 4, -1.5 # start point on the x-y plane
# start_point = [x_init,y_init]
# step = 0.00006
# discount = 0.7
# epoch=100
# epsilon=1e-2
# yend=[0,0]


# --------------------------f---------------------------


# --------------------------f_gold---------------------------

# FN_INDEX =3  # select which function to draw. Start from 0!
# # H_INDEX=0
# opt_index=4#set parameters---------------------------
# x_init, y_init = -4, 4.5 # start point on the x-y plane
# start_point = [x_init,y_init]
# step = 0.00000005
# discount = 0.9
# epoch=100
# epsilon=1e-2
# yend=[0,-1]
#


# --------------------------f-------------------------


def lossfn(yend, x):  # l2

    loss = np.linalg.norm(x - yend)
    return loss


# def lossfn(yend,x):#log_l2
#    loss= np.linalg.norm(x - yend)
#    loss=log2(loss)
#    return loss

# -------------------------------------------------
def fn(x_start, index=FN_INDEX):
    x = x_start[0]
    y = x_start[1]
    fn_list = [
        x * x + 50 * y * y,  # formula 1-------
        -(np.cos(x) + 1) * (np.cos(2 * y) + 1),  # formula tra-------------

        (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x),  # rob

        (1 + (x + y + 1) * (x + y + 1) * (19 - 14 * x + 3 * x * x - 14 * y + 6 * x * y + 3 * y * y)) * (
                    30 + (2 * x - 3 * y) * (2 * x - 3 * y) * (
                        18 - 32 * x + 12 * x * x + 48 * y - 36 * x * y + 27 * y * y)),
        0.26 * (x * x + y * y) - 0.48 * x * y,  # gold

    ]
    return fn_list[index]


def g(x0, index=FN_INDEX):
    x = x0[0]
    y = x0[1]
    grad_list = [
        np.array([2 * x, 100 * y]),  # formula 1 's grad--------

        np.array([np.sin(x) * (np.cos(2 * y) + 1), 2 * np.sin(2 * y) * (np.cos(x) + 1)]),  # formula tra 's grad--------

        np.array([2 * x - 2 * x * (- 100 * x ** 2 + 100 * y) - 200 * x * (- x ** 2 + y) - 2, - 200 * x ** 2 + 200 * y]),
        # rob

        np.array([((6 * x + 6 * y - 14) * (x + y + 1) ** 2 + (2 * x + 2 * y + 2) * (
                    3 * x ** 2 + 6 * x * y - 14 * x + 3 * y ** 2 - 14 * y + 19)) * ((2 * x - 3 * y) ** 2 * (
                    12 * x ** 2 - 36 * x * y - 32 * x + 27 * y ** 2 + 48 * y + 18) + 30) + (
                              (x + y + 1) ** 2 * (3 * x ** 2 + 6 * x * y - 14 * x + 3 * y ** 2 - 14 * y + 19) + 1) * (
                              (8 * x - 12 * y) * (12 * x ** 2 - 36 * x * y - 32 * x + 27 * y ** 2 + 48 * y + 18) - (
                                  2 * x - 3 * y) ** 2 * (36 * y - 24 * x + 32)), (
                              (6 * x + 6 * y - 14) * (x + y + 1) ** 2 + (2 * x + 2 * y + 2) * (
                                  3 * x ** 2 + 6 * x * y - 14 * x + 3 * y ** 2 - 14 * y + 19)) * (
                              (2 * x - 3 * y) ** 2 * (
                                  12 * x ** 2 - 36 * x * y - 32 * x + 27 * y ** 2 + 48 * y + 18) + 30) - (
                              (x + y + 1) ** 2 * (3 * x ** 2 + 6 * x * y - 14 * x + 3 * y ** 2 - 14 * y + 19) + 1) * (
                              (12 * x - 18 * y) * (12 * x ** 2 - 36 * x * y - 32 * x + 27 * y ** 2 + 48 * y + 18) - (
                                  2 * x - 3 * y) ** 2 * (54 * y - 36 * x + 48))]),

    ]
    return grad_list[index]


def momentum(x_start,step, g, discount = 0.9):   #escent
#     x0 = np.array(x_init, dtype='float64')
#     y0 = np.array(y_init, dtype='float64')
#     x=[x0,y0]

    x = np.array(x_start, dtype='float64')
    grad = np.zeros_like(x)
    grad_dot = [grad.copy()]
    passing0_dot= [x.copy()]
    passing_dot = [x.copy()]
    pre_grad = np.zeros_like(x)
    pre_grad_dot = [pre_grad.copy()]
    pre_grad2 = np.zeros_like(x)
    pre_grad2_dot = [pre_grad2.copy()]
    loss_dot=[]
    for i in range(epoch):
        x0=x
        passing0_dot.append(x0.copy())
        grad = g(x)
        pre_grad = pre_grad2 * discount + grad* step   
        x -= pre_grad     
        passing_dot.append(x.copy())
        pre_grad2_dot.append(pre_grad2.copy())
        grad_dot.append(grad.copy())
        pre_grad_dot.append(pre_grad.copy())
#         print( '[ Epoch {0} ]  x0 = {1},grad = {2}, pre_grad = {3}, pre_grad2 = {4}, x = {5}'.format(i,x0, -step*grad,-pre_grad,-discount*pre_grad2, x))
        pre_grad2=pre_grad 
        
#         loss_MSE=(x-[1,1])*(x-[1,1])
        loss_MSE= lossfn(yend, x)     
        loss_dot.append(loss_MSE)
        if abs(sum(grad)) < epsilon:
            print("m:"+str(i))
            break;       
    x_end=x
#    print(x_end)
#     plt.plot(np.array((range(len(loss_dot)))),loss_dot,"-",label='momentum') 
    return x, passing_dot,loss_dot,grad_dot,pre_grad_dot,pre_grad2_dot,passing0_dot

def sgd(x_start,step, g):   #   #for function1 120~150
#     x0 = np.array(x_init, dtype='float64')
#     y0 = np.array(y_init, dtype='float64')
#     x=[x0,y0]
    x = np.array(x_start, dtype='float64')
    grad = np.zeros_like(x)
    grad_dot = [grad.copy()]
    passing0_dot= [x.copy()]
    passing_dot = [x.copy()]
    pre_grad = np.zeros_like(x)
    pre_grad_dot = [pre_grad.copy()]
    pre_grad2 = np.zeros_like(x)
    pre_grad2_dot = [pre_grad2.copy()]
    loss_dot=[]
    for i in range(epoch):
        x0=x  
        passing0_dot.append(x0.copy())
        grad = g(x)   
        x -= grad  * step   
        passing_dot.append(x.copy())
        pre_grad2_dot.append(pre_grad2.copy())
        grad_dot.append(grad.copy())
        pre_grad_dot.append(pre_grad.copy())
#         print( '[ Epoch {0} ]  x0 = {1},grad = {2}, pre_grad = {3}, pre_grad2 = {4}, x = {5}'.format(i,x0, -step*grad,-pre_grad,-discount*pre_grad2, x))
        pre_grad2=pre_grad 
        passing_dot.append(x.copy())
        loss_MSE= lossfn(yend, x)
        loss_dot.append(loss_MSE)
        #print( '[ Epoch {0} ] grad = {1}, pre_grad = {2}, pre_grad2 = {3}, x = {4}'.format(i, grad,pre_grad,pre_grad2, x))     
        if abs(loss_dot[i]) < epsilon:
            if abs(loss_dot[i-1]) < epsilon:
                print("sgd:"+str(i))
#            break;
    x_end=x
    print(x_end)
#     plt.plot(np.array((range(len(loss_dot)))),loss_dot,"-",label='sgd') 
    return x, passing_dot,loss_dot,grad_dot,pre_grad_dot,pre_grad2_dot,passing0_dot
def nesterov(x_start, step, g, discount = 0.7):
    x = np.array(x_start, dtype='float64')
    passing_dot = [x.copy()]
    pre_grad = np.zeros_like(x)
    loss_dot=[]
    for i in range(epoch):
        x_future = x - step * discount * pre_grad
        grad = g(x_future)
        pre_grad = pre_grad * discount + grad 
        x -= pre_grad * step
        passing_dot.append(x.copy())
        loss_MSE= lossfn(yend, x)
        loss_dot.append(loss_MSE)
        #print '[ Epoch {0} ] grad = {1}, x = {2}'.format(i, grad, x)
        if abs(sum(grad)) < epsilon:
            print("nag:"+str(i))
            break;
    x_end=x
    print(x_end)
#     plt.plot(np.array((range(len(loss_dot)))),loss_dot,"-",label='nesterov') 
    return x, passing_dot,loss_dot
def m99(x_start,step, g, discount = 0.7):   #
    x = np.array(x_start, dtype='float64')
    passing_dot = [x.copy()]
    pre_grad = np.zeros_like(x)
    pre_grad2 = np.zeros_like(x)
    loss_dot=[]
    for i in range(epoch):
        grad = g(x)
        sign=np.sign(grad) * np.sign(pre_grad2)
        sign=np.clip(sign,-1,0)
        pre_grad2=pre_grad2+(1*sign*pre_grad2)
        pre_grad = pre_grad2 * discount + grad
        x -= pre_grad * step        
        passing_dot.append(x.copy())
        loss_MSE= lossfn(yend, x)
        loss_dot.append(loss_MSE)
        #print( '[ Epoch {0} ] grad = {1}, pre_grad = {2}, pre_grad2 = {3}, x = {4}'.format(i, grad,pre_grad,pre_grad2, x))
        pre_grad2=pre_grad      
        if abs(sum(grad)) < epsilon:
            print("m99:"+str(i))
            break;
    x_end=x
    print(x_end)
#     plt.plot(np.array((range(len(loss_dot)))),loss_dot,"-",label='m99') 
    return x, passing_dot,loss_dot







i=-1
loss4=[]
xa,x_arr0,loss0,grad_dot,pre_grad_dot,pre_grad2_dot, passing0  = sgd(start_point, step, g)#sgd


passing0=np.array(passing0)
x_arr0 = np.array(x_arr0)
grad_dot = np.array(grad_dot)
pre_grad_dot = np.array(pre_grad_dot)
pre_grad2_dot = np.array(pre_grad2_dot)
x0=passing0[:,0]
y0=passing0[:,1]
x=x_arr0[:,0]
y=x_arr0[:,1]
x_grad=-step*grad_dot[:,0]
x_pre_grad=-pre_grad_dot[:,0]
x_pre_grad2=-discount*pre_grad2_dot[:,0]
y_grad=-step*grad_dot[:,1]
y_pre_grad=-pre_grad_dot[:,1]
y_pre_grad2=-discount *pre_grad2_dot[:,1]



#xx = np.linspace(0, 50, 50)  
#yy = 0*xx

fig=plt.figure()
#plt.legend()
#plt.subplots_adjust(top=2)  




#ax1 = plt.subplot(211)
#line1=plt.plot(np.array((range(len(x_grad)))),x_grad,"-",color='red',label='MOM_-r*gradt_x')
#line2=plt.plot(np.array((range(len(x_pre_grad2)))),x_pre_grad2,"-",color='green',label='MOM_-a*vt-1_x')
#line3=plt.plot(np.array((range(len(x_pre_grad)))),x_pre_grad,"-",label='MOM_vt_x')
#plt.plot(xx, yy, linestyle='--',linewidth=1,color="black")
#plt.plot([34,34], [-1,1], linestyle='-.',linewidth=1,color="darkorange")
#plt.plot([13,13], [-1,1], linestyle='-.',linewidth=1,color="darkgreen")
##plt.plot([21,21], [-5,1], linestyle='-',linewidth=1,color="black")
#plt.ylim(-0.1,0.25 )
#plt.axvline(83,ls="--",linewidth=1,color="darkviolet")
#plt.setp(ax1.get_xticklabels(), fontsize=6)

# plt.title('')

ax2= plt.subplot()
l=plt.plot(np.array((range(len(x0)))),x0,"-",color='darkblue',label='MOM_xt-1')

ax2.set_yticks([-2,-1.5,-1,-0.5,0,0.5,1,1.5,2])
ax2.set_yticklabels([-2,-1.5,-1,-0.5,0,0.5,1,1.5,2])

plt.axhline(0,ls="--",linewidth=1,color="black")
'''
ax2.set_xticks([0,20,40,60,80,100])
ax2.set_xticklabels([0,20,40,60,80,100])

'''

#legend1=ax2.legend(['MOM'],loc=1)
font1 = {'size': 20}
font2 = {'size': 15}
legend1=ax2.legend(['GD'],loc=1,prop = font1)
#legend1=ax2.legend(['NAG'],loc=1)
#legend1=ax2.legend(['ISP-Optimizer'],loc=1)


#plt.legend([line1, line2, line3],["MOM", "SGD",  "NAG"], loc=4,ncol=2)
#ax2.axis["left"].label.set_text("Long Label Left")
# legend(loc='upper left')
# make these tick labels invisible
# plt.setp(ax3.get_xticklabels(), visible=False)

plt.ylim(-2.1,1.8 )
plt.xlim(0,epoch)
# plt.yticks([-1,-0.5,0,0.5,1])
# plt.xlim([0, 50])
# ll = plt.legend([line1, line2, line3],["MOM", "SGD",  "NAG"], loc=4,ncol=2)
# plt.arrow(0, 0, 1, 1, head_width=0.05, head_length=0.1, fc='k', ec='k')
# #jiantou
# import matplotlib.pyplot as plt

#for a in range(len(x0)):
#    if abs(x0[a]) <= 0.1:
#        print(a)


# ax = plt.axes()
# fruits = ['banana', 'apple',  'mango']
#ax2.label_outer()




#ax2.axis["left"].label.set_text("Long Label Left")
for t in range(len(x0)):        
    sy=x0[t]
#     if sy ==0:
#         print(t)
    grad=plt.arrow(t-0.1, sy, 0,x_grad[t]*10 ,head_width=0.5, head_length=0.05, fc='r', ec='r')

#for t in range(len(x0)):        
#    sy=x0[t]
#    pregrad=plt.arrow(t+0.1, sy, 0,x_pre_grad2[t]*10,head_width=0.5, head_length=0.05, fc='g', ec='g')
#plt.show
plt.tight_layout()
plt.subplots_adjust(hspace=0.015)
#plt.legend([])
#plt.legend([grad],['gradient'])
ax2.legend([l,grad],['SGD','gradient'],loc=4,prop = font2)
#bbox_to_anchor=(0.84, 0.9)
plt.gca().add_artist(legend1)
plt.savefig('f'+str(FN_INDEX)+'oversgd.png', dpi=150, quality=100)