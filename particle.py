import numpy as np
import matplotlib.pyplot as plt
import random
from math import *


sensor = [10.0, 0.0]


class auv(object):

	def __init__(self):
		self.x = random.gauss(0.0, 1.0)
		self.y = random.gauss(0.0, 2.0)
		self.a = random.gauss(0.0, 5.0)
		self.u_noise = 0.0
		self.v_noise = 0.0
		self.r_noise = 0.0
		self.u = 1.0
		self.v = 0.0
		self.r = 0.0

	def set_speed(self, new_u, new_v, new_r):
		self.u = float(new_u)
		self.v = float(new_v)
		self.r = float(new_r)

	def set_noise(self, u_n, v_n, r_n):
		self.u_noise = float(u_n)
		self.v_noise = float(v_n)
		self.r_noise = float(r_n)

	def set_position(self, new_x, new_y, new_a):
		self.x = float(new_x)
		self.y = float(new_y)
		self.a = float(new_a)


	def sense(self):
		dist = sqrt((self.x - sensor[0]) ** 2 + (self.y - sensor[1]) ** 2)  
		#dist+=random.gauss(0.0, sense_noise)
		return dist	



	def likihood(self,sigma):
		 return exp(- ((self.sense() - 10.0) ** 2) / (sigma ** 2) / 2.0) / sqrt(2.0 * pi * (sigma ** 2))  

	def move(self):
		u = float(self.u) + random.gauss(0.0, self.u_noise)
		v = float(self.v) + random.gauss(0.0, self.v_noise)
		sina = np.sin(self.a * np.pi / 180)
		cosa = np.cos(self.a * np.pi / 180)
		d = np.array([float(self.x), float(self.y)])
		s = np.array([u, v])
		R = np.array([[cosa, -sina], [sina, cosa]])
		d = np.dot(R, s)
		x = d[0]+self.x
		y = d[1]+self.y
		a = float(self.a) + float(self.r) + random.gauss(0.0, self.r_noise)
		av = auv()
		av.set_position(x, y, a)
		av.set_noise(self.u_noise, self.v_noise, self.r_noise)
		av.set_speed(self.u, self.v, self.r)
		return av

	def Gaussian(self, mu, sigma, x):
		 return exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / sqrt(2.0 * pi * (sigma ** 2))
"""

	def measurement_prob(self, measurement):

        # calculates how likely a measurement should be
        # 计算出的距离相对于正确正确的概率　离得近肯定大　离得远就小
                prob = 1.0;

            dist = sqrt((self.x - landmarks[0])
                        ** 2 + (self.y - landmarks[1]) ** 2)
            prob *= self.Gaussian(dist, self.sense_noise, measurement[i])
        return prob
"""
N = 500
colors=['r','c','m','g','y','gold','mediumpurple','darkgreen','hotpink','darkred','orangered']
fig, (ax,ax2)= plt.subplots(1,2, figsize=(18, 6))  # 航跡プロットの準備
ax.set_xlabel('Y [m]')  # X軸を上向きに描画するため、XY軸を入れ替える
ax.set_ylabel('X [m]')
ax.grid(True)  # グリッドを表示
ax.axis('equal')  # 軸のスケールを揃える（重要！）
ax2.set_xlabel('Y [m]')  # X軸を上向きに描画するため、XY軸を入れ替える
ax2.set_ylabel('X [m]')
ax2.grid(True)  # グリッドを表示
ax2.axis('equal')  # 軸のスケールを揃える（重要！）




p=[] 
X=[]
Y=[]

for i in range(N):
	n = auv()
	n.set_noise(0.15,0.2,1.0)
	#ax.plot(n.y,n.x,'.b',size=20)
	p.append(n)
	X.append(n.x)
	Y.append(n.y)

ax.scatter(Y,X,s=5,c='b')

for t in range(100):
	q=[]
	X=[]
	Y=[]
	for au in p:
		m=au.move()
		print(m.x,m.y)
		q.append(m)
		X.append(m.x)
		Y.append(m.y)
		#ax.plot(m.y,m.x,'ok')
	p=q
	if(t%10!=9):continue
	ax.scatter(Y,X,c=colors[t//10],s=5)
	plt.pause(0.01)


p=[] 
X=[]
Y=[]


for i in range(N):
	n = auv()
	#n.set_position
	n.x = random.gauss(0.0, 1.2)
	n.set_noise(1.5,0.8,2.0)
	n.set_speed(0.0, 0.0, 0.0)
	#ax.plot(n.y,n.x,'.b',size=20)
	p.append(n)
	X.append(n.x)
	Y.append(n.y)

ax2.scatter(Y,X,s=5,c='b')

for t in range(100):
	q=[]
	X=[]
	Y=[]
	W=[]
	for au in p:
		m=au.move()
		w=m.likihood(2.0)
		W.append(w)
        #print(m.x,m.y)

		q.append(m)

	sum_w=sum(W)
	for i in range(N):
		W[i]/=sum_w
	samples = np.random.multinomial(N, W)
	p=[]
	for i,s in enumerate(samples):
		for j in range(s):
			p.append(q[i])
			X.append(q[i].x)
			Y.append(q[i].y)
	#p=q
	if(t%10!=9):continue
	ax2.scatter(Y,X,s=5,c=colors[t//10])
	plt.pause(0.01)



#f.close()  # ファイルを閉じる
plt.show()  # グラフウィンドウを開いたままにする

