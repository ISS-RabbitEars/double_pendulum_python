import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation


def a1a2(to12, t, p):
	theta1,omega1,theta2,omega2 = to12
	m1,m2,l1,l2,g = p
	ms=m1+m2
	mu=m2/ms
	s1=np.sin(theta1)
	s2=np.sin(theta2)
	th21=theta2-theta1
	s21=np.sin(th21)
	c21=np.cos(th21)
	A=mu*c21*s21
	B=(l2/l1)*mu*s21
	C=(s1-mu*s2*c21)*(g/l1)
	D=1-mu*c21**2
	return [omega1,((A*omega1**2)+(B*omega2**2)-C)/D,omega2,(l1/l2)*(-c21*(((A*omega1**2)+(B*omega2**2)-C)/D)-s21*(omega1**2))-(g/l2)*s2]

m1=1
m2=1
l1=0.5
l2=1
g=9.8

theta1=90
theta1=(theta1/180)*np.pi
theta2=180
theta2=(theta2/180)*np.pi
omega1=0
omega2=0

p=[m1,m2,l1,l2,g]
to12=[theta1,omega1,theta2,omega2]

tf = 240
nfps = 60
nframes = tf * nfps
t = np.linspace(0, tf, nframes)
dt=1/nfps

aw = odeint(a1a2, to12, t, args = (p,))

th1=aw[:,0]
th2=aw[:,2]

x1=l1*np.sin(th1)
y1=-l1*np.cos(th1)
x2=l2*np.sin(th2)
y2=-l2*np.cos(th2)


xmax=max(x1)+max(x2)+0.2
xmin=min(x1)+min(x2)-0.2
ymax=abs(max(y1))+abs(max(y2))+0.2
ymin=min(y1)+min(y2)-0.2

pe1=m1*g*y1
pe2=m2*g*(y1+y2)

w1=aw[:,1]
w2=aw[:,3]

ke1=0.5*m1*(l1**2)*(w1**2)
ke2=0.5*m2*((l1**2)*(w1**2)+(l2**2)*(w2**2)+(2*l1*l2*w1*w2*np.cos(th1-th2)))

E1=ke1+pe1
E2=ke2+pe2
E=E1+E2

ke=ke1+ke2
pe=pe1+pe2

Emax=max(E)

pe1/=Emax
pe2/=Emax
ke1/=Emax
ke2/=Emax
E1/=Emax
E2/=Emax
E/=Emax
ke/=Emax
pe/=Emax

fig, a=plt.subplots()

def run(frame):
	plt.clf()
	plt.subplot(211)
	plt.arrow(0,0,x1[frame],y1[frame],head_width=None,color='b')
	circle=plt.Circle((x1[frame],y1[frame]),radius=0.05,fc='r')
	plt.gca().add_patch(circle)
	plt.arrow(x1[frame],y1[frame],x2[frame],y2[frame],head_width=None,color='b')
	circle=plt.Circle((x1[frame]+x2[frame],y1[frame]+y2[frame]),radius=0.05,fc='r')
	plt.gca().add_patch(circle)
	plt.title("double pendulum")
	ax=plt.gca()
	ax.set_aspect(1)
	plt.xlim([xmin,xmax])
	plt.ylim([ymin,ymax])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_facecolor('xkcd:black')
	plt.subplot(212)
	plt.plot(t[0:frame],ke[0:frame],'r',lw=0.5)
	plt.plot(t[0:frame],pe[0:frame],'b',lw=0.5)
	plt.plot(t[0:frame],E[0:frame],'g',lw=1.0)
	plt.xlim([0,tf])
	plt.title("Energy (Rescaled)")
	ax=plt.gca()
	ax.legend(['T','V','E'],labelcolor='w',frameon=False)
	ax.set_facecolor('xkcd:black')


ani=animation.FuncAnimation(fig,run,frames=nframes)
writervideo = animation.FFMpegWriter(fps=nfps)
ani.save('doubpendode.mp4', writer=writervideo)

plt.show()




