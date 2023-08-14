from __future__ import division, print_function, unicode_literals
import math
import numpy as np 
import matplotlib.pyplot as plt

plt.figure(figsize=(12,8), dpi= 50, facecolor='w', edgecolor='k')
x0 = np.linspace(-8, 8, 500)
y0 = x0 ** 2 + 5 * np.sin(x0)

plt.plot(x0, y0, 'b')


def grad(x):
    return 2*x+ 5*np.cos(x)

def cost(x):
    return x**2 + 5*np.sin(x)

def myGD1( x0):
    x = [x0]
   
    for it in range(200):
        
        x_new = -(grad(x[-1]))/(2-5*np.sin(x[-1]))+x[-1]
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x, it)
    
(x1, it1) = myGD1( -5)
(x2, it2) = myGD1( 5)
print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), it2))
plt.figure(figsize=(12,8), dpi= 100, facecolor='w', edgecolor='k')

x0 = np.linspace(-7, 8, 500)
y0 = x0 ** 2 + 5 * np.sin(x0)

plt.plot(x0, y0, 'b')

local_y = cost(np.asarray(x1))
global_y = cost(np.asarray(x2))

plt.plot(8, cost(8), 'ro')

plt.plot(x1, local_y, 'r', label = 'pure GD')
plt.plot(x1, local_y, 'rs', markersize=10)

plt.plot(x2, global_y, 'g', label = 'GD with momentum')
plt.plot(x2, global_y, 'go', markersize=3)

plt.legend()
plt.show()








