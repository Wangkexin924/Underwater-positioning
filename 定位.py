import math
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False # 步骤二（解决坐标轴负数的负号显示问题）

#只计算了该程序运行CPU的时间
import timeit
start=timeit.default_timer()
#中间写代码


def Vector_DOA(Vx,Vy,p):#p是向量吗
    c=1500;
    phou=1000;
    length_p=len(p);

    phi_angle=np.arange(0,360-1,0.1);
    phi=[phi_angle[i]*math.pi/180 for i in range(len(phi_angle))]

    A_Vc=[0]*len(phi)

    for i in range(len(phi)):
        Vc=np.multiply(Vx,math.cos(phi[i]))+np.multiply(Vy,math.sin(phi[i]))
        Vc=[a + b for a,b in zip((Vc+p), Vc)]
        A_Vc[i]=sum(list(map(abs,Vc)))/length_p;
    #[ma,I]=max(A_Vc);#ma没用到
    #ma记录每列的最大值
    #I记录每列最大值的行号
    I=A_Vc.index(max(A_Vc))
    I=phi_angle[I];
    return I


c=1500
phou=1000


#数据仿真
x0=-5;
y0=2;
v=2;
T=5;
gama_angle=45;
gama=gama_angle/180*math.pi;

f0=1000;
w=2*math.pi*f0;
k=w/c;
fs=10*f0;

t=np.arange(0,T,1/fs)
#[0.0000e+00 1.0000e-04 2.0000e-04 ... 4.9997e+00 4.9998e+00 4.9999e+00]

v_x=v*math.cos(gama);#目标运动速度在x方向的分量
v_y=v*math.sin(gama);#目标运动速度在y方向的分量
x=v_x*t+x0;#目标实时位置
y=v_y*t+y0;#目标实时位置


#矢量水听器位置
d=10;#两个矢量水听器之间的距离
x1=-d/2;
x2=d/2;
y1=0;
y2=0;


#plt.figure(num = 1)
plt.figure()
plt.plot(x,y)
plt.plot(x1,y1,'*',x2,y2,'*')
plt.title('目标运动轨迹 仿真')
#plt.show()


#信噪比
Pref=pow(10,-6);
Iref=6.67*pow(10,-19);
NL=80;
SL=90;
SNR=SL-NL;

As=Pref*pow(10,SL/20);
An=Pref*pow(10,NL/20);

#1号矢量水听器接收数据仿真
r1=np.sqrt(pow((x-x1),2)+pow((y-y1),2));
phi1 = [math.atan2((y-y1)[i],(x-x1)[i]) for i in range(len(y))]
#matlab length 和 python len函数返回的结果差一个？
#print(phi1)
#这俩怎么都是cos？
#*k到底是扩大列数还是每项乘
kx1=[k*np.cos(phi11) for phi11 in phi1]
ky1=[k*np.cos(phi11) for phi11 in phi1]
p1=As/r1*np.exp(1j*(w*t-k*r1));
#外加噪声
Vx1=(p1)*np.cos(phi1)+An*np.sqrt(1/3)*np.random.randn(t.shape[0]);
Vy1=(p1)*np.sin(phi1)+An*np.sqrt(1/3)*np.random.randn(t.shape[0]);
p1=p1+An*np.random.randn(t.shape[0]);


#2号矢量水听器接收数据仿真
r2=np.sqrt((x-x2)**2+(y-y2)**2);
phi2=[math.atan2((y-y2)[i],(x-x2)[i]) for i in range(len(y))]
kx2=[k*np.cos(phi22) for phi22 in phi2]
ky2=[k*np.cos(phi22) for phi22 in phi2]
p2=As/r2*np.exp(1j*(w*t-k*r2));#虚数i？
#外加噪声
Vx2=(p2)*np.cos(phi2)+An*np.sqrt(1/3)*np.random.randn(t.shape[0]);
Vy2=(p2)*np.sin(phi2)+An*np.sqrt(1/3)*np.random.randn(t.shape[0]);
p2=p2+An*np.random.randn(t.shape[0]);


#数据处理

#———————滤波————————

n=20;#滤波器阶数
Wn=[990/(fs/2),1010/(fs/2)];#滤波器截止频率


'''
Matlab的fir1函数不需要指定滤波器的类型(低通, 高通, 带通或带阻), 因为已经在kaiserord已经指定了滤波器类型. 而python中的firwin函数需要通过pass_zero参数指定滤波器类型.
fir1函数使用滤波器的阶数n, 而firwin函数使用窗长度numtaps
'''

b=sp.signal.firwin(n+1,Wn);#fir滤波器系数
#n+1吗不确定
#带通？因为Wn是二元组
a=1
p1=sp.signal.lfilter(b,a,p1);
Vx1=sp.signal.lfilter(b,a,Vx1);
Vy1=sp.signal.lfilter(b,a,Vy1);
p2=sp.signal.lfilter(b,a,p2);
Vx2=sp.signal.lfilter(b,a,Vx2);
Vy2=sp.signal.lfilter(b,a,Vy2);


#—————————————————
'''
for循环也经常和range搭配使用，range有三种用法
1.range(start,stop)
其中start是序列的起始值，stop是序列的结束至，但不包括该值，即[start,stop)
'''
#单矢量水听器方位估计
deltT=0.5;
T0=int(T/deltT);
fs=int(fs*deltT);

#A_Vc=[0]*len(phi)
e_phi1=[0]*T0
e_phi2=[0]*T0

for i in range(T0):#0-T0不包括T0
    temp_p1=p1[i*fs:(i+1)*fs];
    temp_Vx1=Vx1[i*fs:(i+1)*fs];
    temp_Vy1=Vy1[i*fs:(i+1)*fs];
    e_phi1[i]=Vector_DOA(temp_Vx1, temp_Vy1, temp_p1);#调用了T0次
    temp_p2=p2[i*fs:(i+1)*fs];
    temp_Vx2=Vx2[i*fs:(i+1)*fs];
    temp_Vy2=Vy2[i*fs:(i+1)*fs];
    e_phi2[i]=Vector_DOA(temp_Vx2, temp_Vy2, temp_p2);


#a=[1 2 3 4 5 6 7 8 9]；则 a(1:4)=1 2 3 4
#python数组等索引默认从0开始的。
'''
a =np.array([1,2,3,4,5,6,7,8,9])
b =a[3-1:5]
print(b,a[0])

[3 4 5] 1
'''


phi1_angle=[phi1[i]*180/math.pi for i in range(len(phi1))];
phi2_angle=[phi2[i]*180/math.pi for i in range(len(phi1))];


plt.figure()
x025=np.arange(deltT/2,T-deltT/2+deltT,deltT)

plt.subplot(2,1,1)
plt.plot(t,phi1_angle,'b',label='phi1 angle')
plt.plot(x025,e_phi1,'r*',label='e phi1')
plt.legend()
plt.xlabel('时间/s')
plt.ylabel('目标方位/°')
plt.title('1号矢量水听器')

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)


plt.subplot(2,1,2)
plt.plot(t,phi2_angle,'b',label='phi2 angle')
plt.plot(x025,e_phi2,'r*',label='e phi2')
plt.legend()
plt.xlabel('时间/s')
plt.ylabel('目标方位/°')
plt.title('2号矢量水听器')


end=timeit.default_timer()
print('Running time: %s Seconds'%(end-start))
#返回值是浮点数

