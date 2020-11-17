import numpy as np

from matplotlib import pyplot as plt
import IPython.display as ipd

from scipy.io.wavfile import read as wavread, write as wavwrite
from scipy.signal import spectrogram
from scipy.signal import get_window
from scipy.linalg import solve_toeplitz

class Signal:
    
    def __init__(self, x=np.array([]),Fs=1.,t0=0.):
        self.x = x.astype(np.float64) # Valores de la señal
        self.L = np.size(x) #  Largo de la señal
        self.Fs = Fs # Frecuencia de muestreo
        self.t = np.arange(self.L) / self.Fs + t0 # Tiempo sobre el cual se definen los valores

        
        
    """
        Operaciones para crear una señal:
    """
    
    @classmethod
    def from_wav(cls, filename):
        """
            Lee un wav que puede estar en formato int16, int32, float32, float64 o uint8 y devuelve
            una instancia de Signal() cuyo vector x siempre es de tipo np.float64 y tiene valores 
            entre -1 y 1. 
        """
        
        Fs, x = wavread(filename)
        
        if x.dtype == 'int16':
            nb_bits = np.float(16)
        elif x.dtype == 'int32':
            nb_bits = np.float(32)
        elif x.dtype == 'float32':
            return cls(x.astype(np.float),Fs,0)
        elif x.dtype == 'float64':
            return cls(x.astype(np.float),Fs,0)
        elif x.dtype == 'uint8':
            x = x / 2**8
            return cls(x.astype(np.float),Fs,0)
        else:
            print('No se puede leer el archivo WAV. El formato de cuantización no está soportado por la función scipy.io.wavfile.read')
            return        
        
        x = x / 2**(nb_bits-1)
        return cls(x.astype(np.float),Fs,0)
    
    
    @classmethod
    def cos(cls, f=440.,A=1.,L=44100.,Fs=44100.,t0=0.):
        """
            Función para crear una señal compuesta de una suma de cosenos
            de diferentes amplitudes y frecuencias.
        """
        
        n = np.arange(L)
        
        if np.size(f) == 1:
            return cls(A * np.cos(2 * np.pi * n * f / Fs), Fs, t0)
        
        x = np.zeros(L)
        for i in range(np.size(f)):
            x += A[i] * np.cos(2 * np.pi * n * f[i] / Fs)

        return cls(x,Fs,t0)

    
    @classmethod
    def glotic(cls, F0=125, Fs=44100, t_max=1.,mode='Delta Train'):
        """
            Crea una señal que representa la fuente glótica. 
        """
        if mode == 'Delta Train':
            n = np.arange(t_max * Fs, dtype='int')
            x = np.zeros(int(t_max * Fs))
            x[np.where(np.mod(n,Fs//F0) == 0)] = 1
            return cls(x,Fs=Fs)
        
    
    
    """
        Operaciones básicas:
    """
        
    def fft(self,Nfft=0):
        """
            Función para calcular la DFT normalizada de la señal.
        """
        if Nfft <= 0:
            Nfft = self.L
        return np.fft.fft(self.x,Nfft)

    
    def ifft(self,Nfft=0):
        """
            Función para calcular la IDFT normalizada de la señal.
        """
        if Nfft <= 0:
            Nfft = self.L
        return np.fft.ifft(self.x,Nfft)

    
    def xcorr(self,mode='full'):
        """
            Calcula la autocorrelación de la señal.
        """
        return np.correlate(self.x,self.x,mode='full')
    
        
    def LPC_coef(self, N=20):
        """
            Calcula los N coeficientes LPC de la señal.
        """
        r = self.xcorr() # np.size(r) = self.L * 2 - 1
        ro = r[self.L-1:self.L+N] # np.size(ro) = N + 1
        b = ro[1:N+1]
        a = solve_toeplitz(ro[:N],b)
        G = np.sqrt(ro[0] - a.dot(b))
        
        return a, G

    
    def ceps_coef(self,N=20,Nfft=2**14,window='boxcar'):
        """
            Calcula los N primeros coeficientes cepstrum de la señal.
        """
        x_hat = np.real(np.fft.ifft(np.log(np.abs(self.fft(Nfft))),Nfft))       
        w = get_window(window,2*N) # ventana para hacer el liftering
        return x_hat[:N] * w[-N:]
    

    
    """
        Funciones de display:
    """
    
    def plot(self,ax=None, **kwargs):
        """
            Función para graficar la señal en el tiempo.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            
        ax.plot(self.t,self.x, **kwargs)
        ax.grid(True)
        ax.set_xlabel('Tiempo [seg]')
        ax.set_ylabel('Amplitud')
        
        return ax

    
    def fft_plot(self,Nfft=0,ax=None, **kwargs):
        """
            Función para graficar el espectro de la señal. 
        """
        if Nfft == 0:
            Nfft = self.L * 8
        X = self.fft(Nfft)
        X_plot = np.abs(X)[:int(np.floor(Nfft/2))]
        w = (np.arange(Nfft) / Nfft * self.Fs)[:int(np.floor(Nfft/2))]
        
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)        

        ax.plot(w,X_plot, **kwargs)
        ax.set_xlabel('Frecuencia [Hz]')
        ax.set_ylabel('Amplitud')
        ax.grid(True)

        return ax

    
    def spectrogram(self, ax=None, window='hanning', t_window=.025, t_overlap=0., 
                    Nfft=None, detrend='constant', return_onesided=True, scaling='density', 
                    axis=-1, mode='psd'):
        
        n_perseg = int(t_window * self.Fs)
        n_overlap = int(t_overlap * self.Fs)
        if Nfft is None:
            Nfft = n_perseg * 8
        
        f, t, Sxx = spectrogram(self.x, window=window, fs=self.Fs, nperseg=n_perseg, 
                                noverlap=n_overlap, nfft=Nfft, detrend=detrend, 
                                return_onesided=return_onesided, scaling=scaling, 
                                axis=axis, mode=mode)
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
        ax.pcolormesh(t,f,np.log(np.abs(Sxx)))
        ax.set_xlabel('t [seg]')
        ax.set_ylabel('F [Hz]')
        
        return ax        
        
        
    def play(self):
        """
            Muestra la barra de audio para escuchar la señal.
        """
        return ipd.Audio(self.x, rate=self.Fs)

    
    def to_wav(self,filename):
        """
            Escribe la señal en un archivo .wav
        """
                
        wavwrite(filename,self.Fs,self.x)

        
    
    """
        Funciones para manipulación de la señal:
    """
    
    def copy(self):
        """
            Devuelve una copia de la señal original.
        """
        sig_copy = Signal(self.x.copy(), Fs=self.Fs, t0=self.t[0].copy())
        return sig_copy
    
    
    def crop(self,t0=None,tf=None,M0=None,Mf=None):
        """
            Devuelve una copia de la señal recortada. La señal original no se modifica.
        """
        sig_crop = self.copy()
        if t0 is None:
            if tf is None:
                if M0 is None:
                    if Mf is None:
                        print('Especifique el tiempo o las muestras que desea recortar')
                        return
                    else:
                        sig_crop.x = sig_crop.x[:Mf]
                        sig_crop.t = sig_crop.t[:Mf]
                        sig_crop.L = np.size(sig_crop.x)
                        return sig_crop
                elif Mf is None:
                    sig_crop.x = sig_crop.x[M0:]
                    sig_crop.t = sig_crop.t[M0:]
                    sig_crop.L = np.size(sig_crop.x)
                    return sig_crop
                else:
                    sig_crop.x = sig_crop.x[M0:Mf]
                    sig_crop.t = self.t[M0:Mf]
                    sig_crop.L = np.size(sig_crop.x)
                    return sig_crop
            else:
                sig_crop.x = sig_crop.x[self.t < tf]
                sig_crop.t = sig_crop.t[self.t < tf]
                sig_crop.L = np.size(sig_crop.x)
                return sig_crop
        elif tf is None:
            sig_crop.x = sig_crop.x[self.t >= t0]
            sig_crop.t = sig_crop.t[self.t >= t0]
            sig_crop.L = np.size(sig_crop.x)
            return sig_crop
        else:
            sig_crop.x = sig_crop.x[np.logical_and(self.t >= t0, self.t < tf)]
            sig_crop.t = sig_crop.t[np.logical_and(self.t >= t0, self.t < tf)]
            sig_crop.L = np.size(sig_crop.x)
            return sig_crop
                   
            
    def append(self,sig1,where='end'):
        """
            Recibe sig1 que puede ser un ndarray o un Signal y lo agrega a la señal
        """
        
        if isinstance(sig1, np.ndarray):
            x_append = sig1.copy()
        elif isinstance(sig1, Signal):
            x_append = sig1.x.copy()
        else:
            print('La señal anexada tiene que ser un objeto Signal o ndarray')
            return
        
        if np.size(self.x) == 0:
            t0 = 0
        else:
            t0 = self.t[0]
        
        if where == 'end':
            self.x = np.hstack((self.x,x_append))
        elif where == 'beginning':
            self.x = np.hstack((x_append,self.x))
        else:
            print('La señal sólo se puede anexar al principio o al final')
            return

        self.L = np.size(self.x)
        self.t = np.arange(self.L) / self.Fs + t0
        
    
    """
        Métodos "mágicos":
    """
    
    def __repr__(self):
        return \
        'Signal(' + '\n' +\
        'x = ' + repr(self.x) + '\n' +\
        't = ' + repr(self.t) + '\n' +\
        'L = ' + repr(self.L) + '\n' +\
        'Fs = ' + repr(self.Fs) + ')'
        
    
    def __str__(self):
        return \
        'x: ' + str(self.x) + '\n' +\
        't: ' + str(self.t) + '\n' +\
        'L: ' + str(self.L) + '\n' +\
        'Fs: ' + str(self.Fs)
    
    
    
    # Falta completar!!!!
    def __add__(self, sig2):
        
        if self.Fs != sig2.Fs:
            print('Error: Las señales sumadas deben tener la misma Fs')
            return
        
        if np.isclose(self.t[0] % sig2.t[0], 0):
            pass
        
        return
            
            
    
# ---------------------------------------------------------------------        
# Esto está medio verde todavía, pero sería un ejemplo de una subclase:
# ---------------------------------------------------------------------
class fonema(Signal):
    
    def __init__(self, x, Fs, t0, fon='a'):
        super().__init__(x, Fs, t0) # Esto copia los atributos x, Fs y t0 de Signal a la clase fonema
        self.fon = fon
    
    
                 
                 
                
        
        
    
    
    
    