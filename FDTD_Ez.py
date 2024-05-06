import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime
from numba import int64, float64, complex128
from numba.experimental import jitclass
from numba import njit

"""Указание структуры для си интерпритатора"""
struct = [
    ('N', int64),
    ('T', int64),
    ('dt', float64),
    ('dx', float64),
    ('dy', float64),
    ('mu0', float64),
    ('eps0', float64),
    ('c', float64),
    ('time', float64[:]),
    ('source', float64[:,:,:]),
    ('init_c', float64[:,:]),
    ('Hx', float64[:,:,:]),
    ('Hy', float64[:,:,:]),
    ('Ez', float64[:,:,:]),
    ('x', float64[:]),
    ('y', float64[:]),
    ('sigE', float64[:,:]),
    ('sigH', float64[:,:]),
    ('chxH', float64[:,:]),
    ('chxE', float64[:,:]),
    ('chyH', float64[:,:]),
    ('chyE', float64[:,:]),
    ('cEz', float64[:,:]),
    ('cHy', float64[:,:]),
    ('cHx', float64[:,:]),
    ('mu', float64[:,:]),
    ('epsilon', float64[:,:]),
    ]

@jitclass(struct)
class FDTD():
    """"""
    def __init__(self, N = 100, T = 100, C=1):
        
        self.x = np.linspace(-0.00001,0.00001,N)
        self.y = np.linspace(-0.00001,0.00001,N)
        self.T = T
        self.N = N
        self.dx = self.x[2]-self.x[1]
        self.dy = self.y[2]-self.y[1]
        
        self.mu0 = np.pi * 4e-7
        self.eps0 = 8.854187817e-12
        
        self.c = 1 / np.sqrt (self.mu0 * self.eps0)
        self.dt = C * (1 / (np.sqrt((1 / (self.dx ** 2)) + (1 / (self.dy ** 2))))) / self.c
        self.time = np.linspace(0, self.dt*self.T, self.T)
        
        self.Hx = np.zeros((N, N, T))
        self.Hy = np.zeros((N, N, T))
        self.Ez = np.zeros((N, N, T))
        
        self.source = np.zeros((N, N, T))
        self.init_c = np.zeros((N, N))
        
        self.sigE = np.zeros((N, N))
        self.sigH = np.zeros((N, N))
        
        self.chxH = np.zeros((N, N))
        self.chxE = np.zeros((N, N))
        
        self.chyH = np.zeros((N, N))
        self.chyE = np.zeros((N, N))
        
        self.cEz = np.zeros((N, N))
        self.cHy = np.zeros((N, N))
        self.cHx = np.zeros((N, N))
        
        self.mu = np.ones((N, N))
        self.epsilon = np.ones((N, N))
        
    def set_source(self, source):
        self.source = source
       
    def set_init(self, init_c):
        self.init_c = init_c
        self.Ez[:,:,0] = self.init_c[:,:]
          
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon  
   
    def set_mu(self, mu):  
        self.mu = mu    
        
    def set_sigma(self, sigE, sigH):  
        self.sigE = sigE
        self.sigH = sigH
        
    def get_c(self):
        return self.c

    def calc_field(self, absorption = False):
        
        for i in range(0, self.N):
            for j in range(0, self.N):
                self.chxH[i, j]  = ((1 - (self.dt * self.sigH[i,j]) / (self.mu[i,j] * self.mu0 * 2)) / 
                               (1 + (self.dt * self.sigH[i,j]) / (self.mu[i,j] * self.mu0 * 2)))
                self.chxE[i, j]  = ((self.dt / (self.dy * self.mu[i,j] * self.mu0)) /
                               (1 + ((self.dt * self.sigH[i,j]) / (self.mu[i,j] *self.mu0 * 2))))
                self.chyH[i, j] = self.chxH[i,j]
                self.chyE[i, j] = ((self.dt / (self.dx * self.mu[i,j] * self.mu0)) / 
                              (1 + ((self.dt * self.sigH[i,j]) / (self.mu[i,j] * self.mu0 * 2))))   
                self.cEz[i, j] = ((1 - (self.dt * self.sigE[i,j]) / (self.epsilon[i,j] * self.eps0 * 2)) / 
                             (1 + (self.dt * self.sigE[i,j]) / (self.epsilon[i,j] * self.eps0 * 2)))
                self.cHy[i, j] = ((self.dt / (self.dx * self.epsilon[i,j] * self.eps0)) /
                             (1 + ((self.dt * self.sigE[i,j]) / (self.epsilon[i,j] * self.eps0 * 2))))
                self.cHx[i, j] = ((self.dt / (self.dy * self.epsilon[i,j] * self.eps0)) /
                             (1 + ((self.dt * self.sigE[i,j]) / (self.epsilon[i,j] * self.eps0 * 2))))
        
        
        for t in range(0, self.T-1):    
            for i in range(1, self.N-1):
                for j in range(0, self.N-2):
                    self.Hx[i, j+1, t+1] = (self.chxH[i,j] * self.Hx[i, j+1, t] - self.chxE[i,j] * 
                                           (self.Ez[i, j+1, t] - self.Ez[i, j, t]))
            for i in range(0, self.N-2):
                for j in range(1, self.N-1):
                    self.Hy[i+1, j, t+1] = (self.chyH[i,j] * self.Hy[i+1, j , t] + self.chyE[i,j] * 
                                          (self.Ez[i+1, j , t] - self.Ez[i, j , t]))  
            for i in range(1, self.N-2):
                for j in range(1, self.N-2):                
                    self.Ez[i, j, t+1] = (self.cEz[i,j] * self.Ez[i, j, t] + self.cHy[i,j] * 
                                        (self.Hy[i+1, j, t + 1] - self.Hy[i, j, t + 1]) - 
                                        self.cHx[i,j] * (self.Hx[i, j+1, t + 1] - self.Hx[i, j, t + 1]) + 
                                        self.source[i,j,t+1])
                    
            if absorption == True:
            
                self.Hx[:, 2, t + 1] = self.Hx[:, 1, t + 1]
                self.Hx[1, :, t + 1] = self.Hx[0, :, t + 1]
            
                self.Hy[:, 1, t + 1] = self.Hy[:, 0, t + 1]
                self.Hy[2, :, t + 1] = self.Hy[1, :, t + 1]
            
                self.Ez[:, 1, t + 1] = self.Ez[:, 0, t + 1]
                self.Ez[1, :, t + 1] = self.Ez[0, :, t + 1]
            
                self.Hx[:, self.N-2, t + 1] = self.Hx[:, self.N-3, t + 1]
                self.Hx[self.N-2, :, t + 1] = self.Hx[self.N-3, :, t + 1]
            
                self.Hy[:, self.N-2, t + 1] = self.Hy[:,self.N-3, t + 1]
                self.Hy[self.N-2, :, t + 1] = self.Hy[self.N-3, :, t + 1]
            
                self.Ez[:, self.N-2, t + 1] = self.Ez[:, self.N-3, t + 1]
                self.Ez[self.N-2, :, t + 1] = self.Ez[self.N-3, :, t + 1]
        