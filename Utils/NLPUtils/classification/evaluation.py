import numpy as np
import torch
import re


def get_train_dev_idx(N,dev_size=.2,random_state=0):

    if random_state is None:
        rand_idx = np.random.permutation(N)
    else:
        rs = np.random.RandomState(random_state)
        rand_idx = rs.permutation(N)

    if dev_size == 0:
        return rand_idx

    N_train = int(N * (1-dev_size))
    if N_train == N:
        print('Warning: dev_size too small!')
        N_train = N-1
    
    return rand_idx[:N_train], rand_idx[N_train:]


def get_kfolds_idx(N,k_folds=5,random_state=0):

    if random_state is None:
        rand_idx = np.random.permutation(N)
    else:
        rs = np.random.RandomState(random_state)
        rand_idx = rs.permutation(N)

    indeces = []
    splitted_arrays = np.array_split(rand_idx,k_folds)
    for i in range(1,k_folds+1):
        train_idx = np.hstack(splitted_arrays[:i-1] + splitted_arrays[i:])
        dev_idx = splitted_arrays[i-1]
        indeces.append((train_idx, dev_idx))

    return indeces

    


def train_dev_validation(model,train_dataset,vectorizer,**kwargs):
    """
    Función para validar un modelo. El entrenamiento es realizado sobre el dataset train_dataset y se 
    mide la performance sobre el dev_dataset, que puede ser dado como argumento de la función, o calculado
    a partir de un valor dev_size. 
    """

    # Argumentos opcionales de la función:
    metric = kwargs.pop('metric','accuracy') # Métrica para medir la performance
    dev_size = kwargs.pop('dev_size',.1) # Tamaño del dev en caso de querer dividir el dataset
    dev_dataset = kwargs.pop('dev_dataset',None) # Dev dataset en caso de que ya exista un dataset para dev
    random_state = kwargs.pop('random_state',None) # Semilla para hacer el split en caso de que no exista un dev dataset

    # Divido el dataframe en train y dev
    if dev_dataset is None:
        N_data = len(train_dataset)
        train_idx, dev_idx = get_train_dev_idx(N_data,dev_size,random_state)
        dev_dataset = train_dataset.iloc[dev_idx,:]
        train_dataset = train_dataset.iloc[train_idx,:]

    # Entrenamos el modelo sobre los datos de train
    vectorized_train_dataset = vectorizer.fit_transform(train_dataset)
    model.train(vectorized_train_dataset,**kwargs)
    
    # Predecimos las nuevas muestras y medimos la performance
    vectorized_dev_dataset = vectorizer.transform(dev_dataset)
    y_dev, y_predict = model.predict(vectorized_dev_dataset)
    score = get_score(y_dev,y_predict,metric)

    return score


def k_fold_validation(model,dataset,vectorizer,k_folds=5,random_state=0,metrics='accuracy',**kwargs):

    N_data = len(dataset)
    indices = get_kfolds_idx(N_data,k_folds,random_state)
    scores = []
    
    for train_idx, dev_idx in indices:
        
        train_dataset = dataset.iloc[train_idx,:]
        dev_dataset = dataset.iloc[dev_idx,:]

        vectorized_train_dataset = vectorizer.fit_transform(train_dataset)
        model.train(vectorized_train_dataset,**kwargs)
        
        vectorized_dev_dataset = vectorizer.transform(dev_dataset)
        y_dev, y_predict = model.predict(vectorized_dev_dataset)
        score = get_score(y_dev,y_predict,metrics)
        scores.append(score)

    return scores


def get_score(y_test,y_predict,metrics):
    if isinstance(metrics,str):
        return check_performance(y_test,y_predict,metrics)

    return {metric:check_performance(y_test,y_predict,metric) for metric in metrics}



def check_performance(y_test,y_predict,metric):

    beta = re.findall(r'f(\d+)_score',metric)
    beta_macro = re.findall(r'f(\d+)_macro',metric)
    beta_micro = re.findall(r'f(\d+)_micro',metric)

    if metric == 'confusion_matrix':
        return confusion_matrix(y_test,y_predict)

    elif metric == 'accuracy':
        return accuracy(y_test,y_predict)

    elif metric == 'balanced_accuracy':
        return balanced_accuracy(y_test,y_predict)

    elif metric == 'precision':
        return precision(y_test,y_predict)

    elif metric == 'recall':
        return recall(y_test,y_predict)

    elif len(beta) > 0:
        return f_beta_score(y_test,y_predict,int(beta[0]))

    elif len(beta_macro) > 0:
        return f_beta_macro(y_test,y_predict,int(beta_macro[0]))

    elif len(beta_micro) > 0:
        return f_beta_micro(y_test,y_predict,int(beta_micro[0]))        

    else:
        raise TypeError('Not supported {} metric'.format(metric))



def confusion_matrix(y_test,y_predict):

    if isinstance(y_test,torch.Tensor):
        y_test = y_test.numpy()
    if isinstance(y_predict,torch.Tensor):
        y_predict = y_predict.numpy()

    classes = np.unique(y_test)
    n_classes = len(classes)
    cm = np.zeros((n_classes,n_classes))
    for i, c in enumerate(classes):
        cm[i,:] = (y_predict[y_test == c].reshape(-1,1) == classes).sum(axis=0)
    return cm


def accuracy(y_test,y_predict):

    cm = confusion_matrix(y_test,y_predict)

    return np.diag(cm).sum() / cm.sum()

def precision(y_test,y_predict):

    cm = confusion_matrix(y_test,y_predict)
    out = np.zeros(cm.shape[0])
    x1 = np.diag(cm)
    x2 = cm.sum(axis=0)

    return np.divide(x1,x2,where=(x2 != 0),out=out)


def recall(y_test,y_predict):

    cm = confusion_matrix(y_test,y_predict)
    out = np.zeros(cm.shape[0])
    x1 = np.diag(cm)
    x2 = cm.sum(axis=1)

    return np.divide(x1,x2,where=(x2 != 0),out=out)


def balanced_accuracy(y_test,y_predict):

    rec = recall(y_test,y_predict)
    return rec.mean()


def f_beta_score(y_test,y_predict,beta):
    
    p = precision(y_test,y_predict)
    r = recall(y_test,y_predict)
    result = np.zeros_like(p)
    x1 = p * r
    x2 = beta**2 * p + r
    np.divide(x1,x2,where=x2!=0,out=result)
    return (1+beta**2) * result


def f_beta_macro(y_test,y_predict,beta):
    return f_beta_score(y_test,y_predict,beta).mean()


def f_beta_weighted(y_test,y_predict,beta):
    weights = np.array([(y_test == c).sum() for c in np.unique(y_test)])
    return f_beta_score(y_test,y_predict,beta).dot(weights)


def f_beta_micro(y_test,y_predict,beta):
    pass



