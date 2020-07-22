import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfTransformer


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


def k_fold_validation(model,dataset,vectorizer,reweight=None,k_folds=5,random_state=0,metrics='accuracy'):
    
    N_data = len(dataset)
    indices = get_kfolds_idx(N_data,k_folds,random_state)
    scores = []
    
    for train_idx, dev_idx in indices:
        
        ds_train, y_train = dataset.iloc[train_idx,0], dataset.iloc[train_idx,1].values
        ds_dev, y_dev = dataset.iloc[dev_idx,0], dataset.iloc[dev_idx,1].values

        X_train = vectorizer.fit_transform(ds_train)
        X_dev = vectorizer.transform(ds_dev)
        
        if reweight is not None:
            X_train, X_dev = do_reweight(X_train,X_dev,method='tfidf')
        
        model.fit(X_train,y_train)
        y_predict = model.predict(X_dev)
        score = get_score(y_dev,y_predict,metrics)
        scores.append(score)

    mean_scores = {metric: [] for metric in scores[0].keys()}
    for score_dict in scores:
        for metric, score in score_dict.items():
            mean_scores[metric].append(score)
    for metric in mean_scores.keys():
        mean_scores[metric] = np.mean(mean_scores[metric])
        
    return mean_scores

def do_reweight(X_train,X_dev,method='tfidf'):
    if method == 'tfidf':
        transformer = TfidfTransformer(smooth_idf=False)
        X_train = transformer.fit_transform(X_train)
        X_dev = transformer.transform(X_dev)
        return X_train, X_dev
    else:
        raise TypeError
        
        

def get_score(y_test,y_pred,metrics):
    
    scores = {}
    
    if isinstance(metrics,str):
        if 'accuracy' == metrics:
            scores['accuracy'] = accuracy_score(y_test,y_pred)
        elif 'balanced_accuracy' == metrics:
            scores['balanced_accuracy'] = balanced_accuracy_score(y_test,y_pred)
        elif 'f1_macro' == metrics:
            scores['f1_macro'] = f1_score(y_test,y_pred,average='macro')
    else:

        if 'accuracy' in metrics:
            scores['accuracy'] = accuracy_score(y_test,y_pred)
        if 'balanced_accuracy' in metrics:
            scores['balanced_accuracy'] = balanced_accuracy_score(y_test,y_pred)
        if 'f1_macro' in metrics:
            scores['f1_macro'] = f1_score(y_test,y_pred,average='macro')
            
    return scores