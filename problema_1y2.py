#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Tue Mar 31 22:02:09 2026

@author: antonia

'''
import time
import matplotlib as mpl
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt 
from matplotlib import rc
from scipy.optimize import curve_fit

plt.rcParams['text.usetex'] = False

plt.rcParams.update({
    "mathtext.fontset": "cm",      
    "font.family": "cm",        
    "font.serif": ["STIXGeneral"],
})
mpl.rcParams['axes.linewidth'] = 1
rc('font',size=13)
rc('font',family='serif')
rc('axes',labelsize=14) 

plt.figure(figsize=(8, 5))
sx = np.array([[0, 1], [1, 0]])
sz = np.array([[1, 0], [0, -1]])
id2 = np.eye(2)

def operador (s, j, N): # j es la posicion de la matriz de pauli en el producto tensorial
    resultado = 1
    for i in range(N):
        resultado = np.kron(resultado, s if i == j else id2)
    return resultado

def Hamiltoniano(N,J,B):
    
    H = np.zeros((2**N,2**N))
    
    for i in range(N-1):  
        H += J * operador(sx, i, N) @ operador(sx, i+1, N)
    
    for i in range(N):
        H +=  B * operador(sz, i, N)
    return H
        

# parte c

def evolución_temporal(H,t):
    
     return expm(-1j * H * t)
    
def psi0(n):
    psi0 = np.array([0,1])
    for i in range (n-1):
        psi0 = np.kron(psi0, [0,1])
    return psi0

J = 1
N = 3
p0 = psi0(N)


t_array = np.linspace(0,10,100)
J = 1
casosB= [0.1,1,10]

for B in casosB:
    probabilidades=[] 
    H = Hamiltoniano(N, J, B)
    for t in t_array:
        
        U = evolución_temporal(H, t)
        #actual = 
        psi_t = U @ p0
        
        prob = np.abs(np.vdot(p0, psi_t))**2
        probabilidades.append(prob)

    plt.plot(t_array, probabilidades, color = "pink")
    plt.ylabel("probabilidad")
    plt.title(rf"J/B={J/B}")
    plt.xlabel("t (s)")
    plt.ylim(0, 1.1)
    plt.show()
    
    
#parte d

N = np.array([4,5,6,7,8,10,11])

tiempos_prom =[]
tiempos =[]
desviaciones = []
mediciones_n = []

for n in N:
    mediciones_n = []
    for i in range(5):
            
        inicio = time.time()
        H = Hamiltoniano(n, 1, 1)
        fin = time.time()
        mediciones_n.append(fin - inicio)
        #print (f"Duración: {fin-inicio} segundos")
        promedio = np.mean(mediciones_n)
    std_dev = np.std(mediciones_n) #Desviación estándar (ruido)
    
    tiempos_prom.append(promedio)
    desviaciones.append(std_dev) 
    
    print(f"Promedio: {promedio:.6f} s (std: {std_dev:.6e}, {n})")

tiempo = np.array(tiempos_prom)
N_array = np.linspace(4, 20,100)


def exponencial(t,a,b,):
    return a* np.exp(b*t)

p,c = curve_fit(exponencial, N , tiempo ,p0=[1e-4, 1.0])
N_smooth = np.linspace(4, 11, 100)
plt.plot(N_smooth, exponencial(N_smooth, *p), label='Ajuste Exponencial', color='pink')
plt.scatter(N, tiempo)

print('tiempo estimado para n=20:', p[0]*np.exp(p[1]*20 ))
print('tiempo estimado para n=50:', p[0]*np.exp(p[1]*50 ))
print('tiempo estimado para n=100:', p[0]*np.exp(p[1]*100 ))

t_pred = exponencial(tiempo, *p)
rmse = np.sqrt(np.mean((tiempo - t_pred)**2))

# problema 2 

import numpy as np
import matplotlib as mpl
from matplotlib import rc
import random
import matplotlib.pyplot as plt 
plt.rcParams['text.usetex'] = False

plt.rcParams.update({
    "mathtext.fontset": "cm",      
    "font.family": "cm",        
    "font.serif": ["STIXGeneral"],
})
mpl.rcParams['axes.linewidth'] = 1
rc('font',size=13)
rc('font',family='serif')
rc('axes',labelsize=14) 


def generar_señal_aleatoria(tn):
    
    f1, f2 = random.randint(1, 15), random.randint(1, 15)
    A, B = random.uniform(0.5, 1.5), random.uniform(0.5, 1.5)
    print(f"Frecuencias generadas: f1={f1}Hz, f2={f2}Hz")
    return A * np.sin(2 * np.pi * f1 * tn) + B * np.sin(2 * np.pi * f2 * tn)

tiempo = np.linspace(0,10, 200)

def dft_manual(x): 
    N = len(x)
   
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        suma = 0
        for n in range(N):
            fase = -2j * np.pi * k * n / N
            suma += x[n] * np.exp(fase)
        X[k] = suma
    return X

Fs = 50
T = 1/Fs
N_v = 200 
t_v = np.arange(N_v) * T
señal_v = generar_señal_aleatoria(t_v)

dft_v = dft_manual(señal_v)
fft_v = np.fft.fft(señal_v)

mag_dft_v = (np.abs(dft_v) / N_v)[:N_v//2] * 2
mag_fft_v = (np.abs(fft_v) / N_v)[:N_v//2] * 2
frec_v = np.fft.fftfreq(N_v, T)[:N_v//2]

casosN= [100, 10e2, 20e2 ,10e3]
tiempos_dft = []
tiempos_fft = []
tiempo = np.linspace(0,10,200)

for N in casosN:
    
    N = int(N) 
    t_array = np.arange(N) * T

    señal_n = np.sin(2 * np.pi * 5 * t_array)
    t = np.arange(N) * T   

    import time
    i_dft = time.time()
    dft =  dft_manual(señal_n) ;magdft = (np.abs(dft) / N)[:N//2]*2
    f_dft =  time.time()

    tiempos_dft.append(f_dft - i_dft)
    
    i_fft = time.time()
    fft = np.fft.fft(señal_n) ;  magfft = (np.abs(fft) / N)[:N //2]*2
    frec = np.fft.fftfreq(N, T)[:N//2]
    f_fft = time.time()
    
    tiempos_fft.append(f_fft - i_fft)
    print(f'{N} t dft= {f_dft - i_dft}, t fft = {f_fft - i_fft} ')
 

        
plt.figure(figsize=(10, 5))
plt.plot(casosN, tiempos_dft, 'o-', linewidth=2, c='magenta')
plt.plot(casosN, tiempos_fft, 's-', linewidth=2, c='blue')
plt.yscale('log')
plt.xscale('log') 
plt.xlabel("tamaño de la muestra ($N$)", fontsize=12)
plt.ylabel("tiempo (segundos) - Escala Log", fontsize=12)
plt.show()

plt.figure(figsize=(10, 10))
plt.subplot(3, 1, 1)


plt.subplot(3, 1, 1)
plt.plot(t_v, señal_v, color='gray')  
plt.title("Señal Aleatoria en el Tiempo")
plt.ylabel("Amplitud")

plt.subplot(3, 1, 2)
plt.plot(frec_v, mag_dft_v, color="#FFB5F0")
plt.title("Espectro: DFT Manual (Picos detectados)")
plt.ylabel("Magnitud")

plt.subplot(3, 1, 3)
plt.plot(frec_v, mag_fft_v, color="#B5FF8A")
plt.title('Espectro: FFT de NumPy')
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
log_N = np.log10(casosN)
log_T_dft = np.log10(tiempos_dft)
log_T_fft = np.log10(tiempos_fft)

p_dft, V_dft = np.polyfit(log_N, log_T_dft, 1, cov=True)
p_fft, V_fft = np.polyfit(log_N, log_T_fft, 1, cov=True)


pendiente_dft = p_dft[0]
pendiente_fft = p_fft[0]
err_pendiente_dft = np.sqrt(V_dft[0, 0])
err_pendiente_fft = np.sqrt(V_fft[0, 0])

print(f"Pendiente DFT: {pendiente_dft:.4f} ± {err_pendiente_dft:.4f}")
print(f"Pendiente FFT: {pendiente_fft:.4f} ± {err_pendiente_fft:.4f}")
