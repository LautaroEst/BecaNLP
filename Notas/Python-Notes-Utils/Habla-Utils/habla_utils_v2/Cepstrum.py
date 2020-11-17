def Cepstrum(sig=None, t_window=.01, delta_t=.01, N_ceps=20, Nfft=0, window_ceps='boxcar'):
    """
        Función para calcular los coeficientes cepstrum para una señal completa.
        Devuelve la matriz ceps_matrix, que contiene en cada columna los N_ceps
        primeros coeficientes cepstrum de la ventana.
    """

    # Agregamos ceros al final de la señal para que 
    # todas las ventanas tengan el mismo ancho.
    N_window = int(t_window * sig.Fs) # Cantidad de muestras por ventana
    N_delta_t = int(delta_t * sig.Fs) # Cantidad de muestras que me corro por iteración
    N_zeros = N_window - (sig.L - (sig.L // N_delta_t) * N_delta_t) # Cantidad de ceros agregados    
    sig_completa = sig.copy()
    sig_completa.append(np.zeros(N_zeros))
    
    # Definimos una matriz ceps_matrix para guardar el espectro
    # del filtro obtenido con los coeficientes cepstrum
    N_windows = (sig_completa.L-N_zeros)//N_delta_t + 1 # Cantidad de iteraciones
    ceps_matrix = np.zeros((N_ceps, N_windows))
    if Nfft == 0:
        Nfft = N_window * 8
        
    for i in range(N_windows):
        window = sig_completa.crop(M0=i*N_delta_t, Mf=i*N_delta_t+N_window)
        ceps_matrix[:,i] = window.ceps_coef(N_ceps,Nfft,window_ceps)            
        
    return ceps_matrix
