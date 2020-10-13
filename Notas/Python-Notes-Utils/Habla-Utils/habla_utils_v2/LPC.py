import numpy as np
from scipy.signal import lfilter

def LPC(sig, t_window, delta_t, N_lpc, Nbits=16):
    """
        Función para calcular la compresión LPC.
        Devuelve la matriz LPC_matrix y la señal de error, que son las cosas que 
        se van a transmitir.
    """

    # Agregamos ceros al final de la señal para que 
    # todas las ventanas tengan el mismo ancho.
    N_window = int(t_window * sig.Fs) # Cantidad de muestras por ventana
    N_delta_t = int(delta_t * sig.Fs) # Cantidad de muestras que me corro por iteración
    N_zeros = N_window - (sig.L - (sig.L // N_delta_t) * N_delta_t) # Cantidad de ceros agregados
    sig_completa = sig.copy()
    sig_completa.append(np.zeros(N_zeros))

    # Definimos una matriz LPC_matrix para guardar el espectro
    # del filtro obtenido con los coeficientes LPC
    N_windows = (sig_completa.L-N_zeros)//N_delta_t + 1 # Cantidad de iteraciones
    LPC_matrix = np.zeros((N_lpc+1, N_windows))
    sig_error = Signal(Fs=sig_completa.Fs)
    zf = np.zeros(N_lpc)
        
    for i in range(N_windows-1):
        window = sig_completa.crop(M0=i*N_delta_t, Mf=i*N_delta_t+N_window)
        a, b = window.LPC_coef(N_lpc)
        window = sig_completa.crop(M0=i*N_delta_t, Mf=(i+1)*N_delta_t)
        y, zf = lfilter(np.hstack((np.array([0]),a)),1,window.x,zi=zf)
        sig_error.append((window.x - y)/b)
            
        LPC_matrix[:-1,i] = a
        LPC_matrix[-1,i] = b
        
    # Última iteración con una ventana de largo t_window para que queden del 
    # mismo tamaño las señales sig_error y sig_completa
    window = sig_completa.crop(M0=(N_windows-1)*N_delta_t, Mf=(N_windows-1)*N_delta_t+N_window)
    a, b = window.LPC_coef(N_lpc)
    y, zf = lfilter(np.hstack((np.array([0]),a)),1,window.x,zi=zf)
    sig_error.append((window.x - y)/b)
    LPC_matrix[:-1,N_windows-1] = a
    LPC_matrix[-1,N_windows-1] = b
    
    # Cuantización para Nbits bits
    min_err = np.min(sig_error.x)
    max_err = np.max(sig_error.x)
    bins = np.linspace(min_err, max_err, 2**Nbits)
    ints = np.digitize(sig_error.x, bins)
    ints[ints == 2**Nbits] = 2**Nbits - 1
    sig_error_cuant = Signal(bins[ints], Fs=sig_error.Fs)
    
    return LPC_matrix, sig_error_cuant
    
    
def ReconstruirSignal(LPC_matrix, sig_error,delta_t):
    
    N_delta_t = int(delta_t * sig_error.Fs)
    N_windows = sig_error.L//N_delta_t
    sig_reconstruida = Signal(Fs=sig_completa.Fs)
    zf = np.zeros(LPC_matrix.shape[0]-1)
    
    for i in range(N_windows):
        window = sig_error.crop(M0=i*N_delta_t, Mf=(i+1)*N_delta_t)
        y, zf = lfilter(LPC_matrix[-1,i].reshape(1,),np.hstack((1,-LPC_matrix[:-1,i])),window.x,zi=zf)
        sig_reconstruida.append(y)
    
    return sig_reconstruida