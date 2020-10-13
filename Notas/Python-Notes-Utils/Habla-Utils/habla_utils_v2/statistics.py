import numpy as np
from matplotlib.patches import Ellipse
from matplotlib import pyplot as plt

def covariance_ellipse(mu, sigma, ax=None, color="k"):
    
    """
        Función para graficar la elispe asociada a una distribución gaussiana
        de media mu y matriz de covarianza sigma, en 2D.
        
    """

    # Cálculo de los autovalores:
    vals, vecs = np.linalg.eigh(sigma)
    
    # Inclinación de la elipse:
    x, y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y, x))

    # Semiejes de la elipse:
    w, h = 2 * np.sqrt(vals)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
                
    ax.tick_params(axis='both', which='major', labelsize=20)
    ellipse = Ellipse(mu, w, h, theta, color=color)
    ellipse.set_clip_box(ax.bbox)
    ellipse.set_alpha(0.2)
    ax.add_artist(ellipse)
    
    return ax
    
    
# class KMeansClassifier:
        
#     def train(self, X_train, epsilon=1e-2, N_classes=3, mu0=None):
        
#         D_old = np.inf # Distorsión de la iteración anterior.
#         mu = mu0 # Media inicial. mu[j,:] = media estimada para la clase j
        
#         # Iteración para encontrar las medias de las clases. El lazo se rompe
#         # cuando la distorisón anterior y la actual no cambia (la diferencia 
#         # es menor a un epsilon).
#         while True:
            
#             # Calculo para cada muestra el índice de la media que más cerca está 
#             # de la muestra y guardo los resultados en el vector c.
#             c = np.argmin(np.sum((X_train[:,np.newaxis] - mu[np.newaxis])**2,axis=2),axis=1)
            
#             # Matriz auxiliar para calcular la media.
#             M = (c[:,None] == np.arange(N_classes)).astype(np.float)
            
#             # Calculo la media.
#             mu = np.array([M.astype(np.float).T.dot(X_train)[i] / M.sum(axis=0)[i] 
#                            if ((M.sum(axis=0) == 0)[i]==0) 
#                            else np.array([0., 0.]) 
#                            for i in range(N_classes)])
            
#             # Calculo la nueva distorisón.
#             D_new = np.sum(np.sum(l,axis=0)/X_train.shape[0])

#             # Si la distorisón no varió, me voy del lazo.
#             if np.abs(D_old - D_new) < epsilon:
#                 break
            
#             # Si no, sigo.
#             D_old = D_new
            
#         # Me guardo las N_classes medias encontradas.
#         self.mu = mu
    
#     def classify(self, X_test):
#         pass
    
        
#     def cross_validation(self, num_folds=1, k_choices=[1, 3]):
        
#         if self.X_train is None:
#             print('No se recibieron las muestras de entrenamiento aún.')
#             return
            
#         X_train_folds = np.array_split(self.X_train, num_folds)
#         y_train_folds = np.array_split(self.y_train, num_folds)
        
#         accuracy = []
#         for fold in range(num_folds):
#             X_train_new = np.concatenate([x for i,x in enumerate(X_train_folds) if i!=fold])
#             X_validate = X_train_folds[fold]
#             y_train_new = np.concatenate([x for i,x in enumerate(y_train_folds) if i!=fold])
#             y_validate = y_train_folds[fold]
#             self.train(X_train_new,y_train_new)
#             y_predict = self.predict(X_validate,k=k_choices)
#             num_correct = np.sum(np.equal(y_predict,y_validate))
#             accuracy.append(float(num_correct) / num_test)

