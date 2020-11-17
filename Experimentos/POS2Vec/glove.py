import pandas as pd

def loadGloveModel(gloveFile):
    glove = pd.read_csv(gloveFile, sep=' ', header=None, encoding='utf-8', 
                        index_col=0, na_values=None, keep_default_na=False, quoting=3)
    return glove  # (word, embedding), 400k*dim



df_glove = loadGloveModel('/home/lestien/Documents/BecaNLP/Utils/Datasets/GloVe/glove.6B.300d.txt')
