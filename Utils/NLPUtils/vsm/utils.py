from sklearn.decomposition import TruncatedSVD
import numpy as np



def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurence count matrix of dimensionality (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    
        Params:
            M (numpy matrix of shape (number of corpus words, number of corpus words)): co-occurence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)): matrix of k-dimensioal word embeddings.
                    In terms of the SVD from math class, this actually returns U * S
    """    
    n_iters = 10     # Use this parameter in your call to `TruncatedSVD`
    TSVD = TruncatedSVD(n_components=k, n_iter=n_iters)
    M_reduced = TSVD.fit_transform(M)
    return M_reduced


# Reweight methods:

def observed_over_expected(df):
    col_totals = df.sum(axis=0)
    total = col_totals.sum()
    row_totals = df.sum(axis=1)
    expected = np.outer(row_totals, col_totals) / total
    oe = np.zeros_like(df)
    oe[expected != 0] = df[expected != 0] / expected[expected != 0]
    return oe


def pmi(df, positive=True):
    df = observed_over_expected(df)
    # Silence distracting warnings about log(0):
    with np.errstate(divide='ignore'):
        df = np.log(df)
    df[np.isinf(df)] = 0.0  # log(0) = 0
    if positive:
        df[df < 0] = 0.0
    return df


def tfidf(df):
    # Inverse document frequencies:
    doccount = float(df.shape[1])
    freqs = df.astype(bool).sum(axis=1)
    idfs = np.log(doccount / freqs)
    idfs[np.isinf(idfs)] = 0.0  # log(0) = 0
    # Term frequencies:
    col_totals = df.sum(axis=0)
    tfs = np.zeros_like(df)
    tfs[:,col_totals!=0] = df[:,col_totals!=0] / col_totals[col_totals != 0]
    return (tfs.T * idfs).T