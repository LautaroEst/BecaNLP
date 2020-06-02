from .utils import get_train_dev_idx, get_kfolds_idx


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
    score = check_performance(y_dev,y_predict,metric)

    return score


def k_fold_validation(model,dataset,vectorizer,k_folds=5,random_state=0,metric='accuracy',**kwargs):

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
        score = check_performance(y_dev,y_predict,metric)
        scores.append(score)

    return scores


def check_performance(y_test,y_predict,metric):
    if metric == 'accuracy':
        perf = accuracy(y_test,y_predict)

    return perf


def accuracy(y_test,y_predict):

    num_samples = len(y_predict)
    num_correct = (y_test == y_predict).sum()
    acc = float(num_correct) / num_samples

    print('Total accuracy: {}/{} ({:.2f}%)'\
          .format(num_correct, num_samples, 100*acc))

    return acc
