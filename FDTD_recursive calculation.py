import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime
from numba import njit

@njit
def FDTD(N,epsilon):
    x = np.linspace(-0.00001,0.00001,N)
    y = np.linspace(-0.00001,0.00001,N)
    
    dx = x[2]-x[1]
    dy = y[2]-y[1]
    
    mu0 = np.pi * 4e-7
    eps0 = 8.854187817e-12
    
    c = 1 / np.sqrt (mu0 * eps0)
    dt = (1 / (np.sqrt((1 / (dx ** 2)) + (1 / (dy ** 2))))) / c
    
    sigE = np.zeros((N, N))
    sigH = np.zeros((N, N))
    
    chxH = np.zeros((N, N))
    chxE = np.zeros((N, N))
    
    chyH = np.zeros((N, N))
    chyE = np.zeros((N, N))
    
    cEz = np.zeros((N, N))
    cHy = np.zeros((N, N))
    cHx = np.zeros((N, N))
    
    mu = np.ones((N, N))

    for i in range(0, N):
        for j in range(0, N):
            chxH[i, j]  = ((1 - (dt * sigH[i,j]) / (mu[i,j] * mu0 * 2)) / 
                           (1 + (dt * sigH[i,j]) / (mu[i,j] * mu0 * 2)))
            chxE[i, j]  = ((dt / (dy * mu[i,j] * mu0)) /
                           (1 + ((dt * sigH[i,j]) / (mu[i,j] * mu0 * 2))))
            chyH[i, j] = chxH[i,j]
            chyE[i, j] = ((dt / (dx * mu[i,j] * mu0)) / 
                          (1 + ((dt * sigH[i,j]) / (mu[i,j] * mu0 * 2))))   
            cEz[i, j] = ((1 - (dt * sigE[i,j]) / (epsilon[i,j] * eps0 * 2)) / 
                         (1 + (dt * sigE[i,j]) / (epsilon[i,j] * eps0 * 2)))
            cHy[i, j] = ((dt / (dx * epsilon[i,j] * eps0)) /
                         (1 + ((dt * sigE[i,j]) / (epsilon[i,j] * eps0 * 2))))
            cHx[i, j] = ((dt / (dy * epsilon[i,j] * eps0)) /
                         (1 + ((dt * sigE[i,j]) / (epsilon[i,j] * eps0 * 2))))
    return chxH, chxE, chyH, chyE, cEz, cHy, cHx, dt
    

@njit
def FDTD_t(N, index, chxH, chxE, chyH, chyE, cEz, cHy, cHx, dt, Hx0, Hy0, Ez0):

    siz = N
    Hx = np.zeros((N, N, 2))
    Hy = np.zeros((N, N, 2))
    Ez = np.zeros((N, N, 2))
    # for t in range(0, T-1):
    # if index != 0:
    Hx[:,:, 0] = Hx0
    Hy[:,:, 0] = Hy0
    Ez[:,:, 0] = Ez0
    
    source = np.zeros((N, N))
    source[10:N-10,50:52] = 4*np.sin(16*599584916021005.8*(dt*(index+1)))
       
    for i in range(1, N-1):
        for j in range(0, N-2):
            Hx[i, j+1, 1] = (chxH[i,j] * Hx[i, j+1, 0] - chxE[i,j] * 
                                   (Ez[i, j+1, 0] - Ez[i, j, 0]))
    for i in range(0, N-2):
        for j in range(1, N-1):
            Hy[i+1, j, 1] = (chyH[i,j] * Hy[i+1, j , 0] + chyE[i,j] * 
                                  (Ez[i+1, j , 0] - Ez[i, j , 0]))  
    for i in range(1, N-2):
        for j in range(1, N-2):                
            Ez[i, j, 1] = (cEz[i,j] * Ez[i, j, 0] + cHy[i,j] * 
                                (Hy[i+1, j, 1] - Hy[i, j, 1]) - 
                                cHx[i,j] * (Hx[i, j+1, 1] - Hx[i, j, 1]) + 
                                source[i,j])
    
    Hx[:, 2, 1] = Hx[:, 1, 1]
    Hx[1, :, 1] = Hx[0, :, 1]

    Hy[:, 1, 1] = Hy[:, 0, 1]
    Hy[2, :, 1] = Hy[1, :, 1]

    Ez[:, 1, 1] = Ez[:, 0,  1]
    Ez[1, :, 1] = Ez[0, :,  1]

    Hx[:, siz-2,  1] = Hx[:, siz-3,  1]
    Hx[siz-2, :,  1] = Hx[siz-3, :,  1]

    Hy[:, siz-2,  1] = Hy[:,siz-3,  1]
    Hy[siz-2, :,  1] = Hy[siz-3, :,  1]

    Ez[:, siz-2,  1] = Ez[:, siz-3,  1]
    Ez[siz-2, :,  1] = Ez[siz-3, :,  1]

    return Hx[:,:, 1], Hy[:,:, 1], Ez[:,:, 1]


@njit
def circle_set(N, L, r):
    
    space = np.ones((N, N))
    x = np.linspace(-L/2, L/2, N)
    y = x
    for i in range(0, N):
        for j in range(0, N):
            if ((x[i]/1.6)**2 + (y[j]+0.3)**2*22) <= r**2:
                space[i, j] = 4

                            
        
    return space


T = 1800
N=650
r = 0.3
L = 1

set_circ = circle_set(N, L, r)
# plt.imshow(set_circ)

Hx_0 = np.zeros((N, N))
Hy_0 = np.zeros((N, N))
Ez_0 = np.zeros((N, N))
# Ez_0[100:110, 100:110] = 4
const = FDTD(N, set_circ) 


start = datetime.datetime.now()
print('Время старта: ' + str(start))
fig1, ax = plt.subplots()
cnt = (Hx_0, Hy_0, Ez_0) 
ims = []
for i in range(T):
      
    cnt = FDTD_t(N, i, *const, *cnt)
    
    im = ax.imshow(cnt[2] , animated=True, cmap='plasma', aspect='equal', vmin=-1, vmax=1)
    ims.append([im])
ani = animation.ArtistAnimation(fig1, ims, interval=1, blit=True,
                                repeat_delay=1000)
fig1.colorbar(im)
ani.save("linza4.gif")
plt.show()
# plt.imshow(cnt[2], cmap='plasma', aspect='equal', vmin=-1, vmax=1)
finish = datetime.datetime.now()
print('Время окончания: ' + str(finish))
print('Время работы: ' + str(finish - start)) 