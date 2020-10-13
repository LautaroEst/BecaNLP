import pandas as pd
import string
import re
from textblob import TextBlob


def filter_by_rates(df,max_comments,n,random_state):
    
    # Limitamos la cantidad de comentarios por producto:
    scores = -(df['review_rate']-1)/4 + \
    .5 * (df['review_likes']-min(df['review_likes']))/(max(df['review_likes'])-min(df['review_likes'])) + \
    .1 * (df['review_dislikes']-min(df['review_dislikes']))/(max(df['review_dislikes'])-min(df['review_dislikes']))
    indices = scores.sort_values(ascending=False).index
    df = df.loc[indices,:].groupby('prod_id').head(max_comments)
    
    # Muestreamos aleatoriamente y mezclamos:
    random_state = np.random.RandomState(random_state)
    df = df.iloc[np.hstack([random_state.choice(arr, n, replace=False) 
                           for arr in df.groupby('review_rate').indices.values()]),:]
    df = df.reset_index(drop=True)
    df = df.iloc[random_state.permutation(len(df)),:]
    df = df.reset_index(drop=True)
    return df


def filter_reviews():
    
    # Leo los datos:
    df = pd.read_csv('data.csv')
    
    # Defino el vocabulario de caracteres:
    non_ascii = 'áàâãäÁÀÂÃÄéèêëÉÈÊẼËíìîĩïÍÌÎĨÏóòôõöÓÒÔÕÖúùûũüÚÙÛŨÜñÑçÇ'
    chars_vocab = string.digits + string.ascii_letters + string.punctuation + string.whitespace + non_ascii
    chars_vocab_regex = '|'.join(sorted([re.escape(c) for c in chars_vocab]))
    
    # Elimino los comentarios con caracteres raros:
    df = df[-df['review_content'].str.contains(chars_vocab_regex,regex=True,na=True)].reset_index(drop=True)
    
    # Separo por país:
    df_por = df[df['country'] == 'MLB'].reset_index(drop=True)
    df_esp = df[df['country'] != 'MLB'].reset_index(drop=True)
    
    # Elimino comentarios en otros idiomas:
    df_por = df_por[df_por['review_content'].apply(lambda s: TextBlob(s).detect_language()) == 'pt'].reset_index(drop=True)
    df_esp = df_esp[df_esp['review_content'].apply(lambda s: TextBlob(s).detect_language()) == 'es'].reset_index(drop=True)
    
    # Filtramos los comentarios por valorización:
    max_comments = 20
    random_state = 1234
    n = 25000
    df_por = filter_by_rates(df_por,max_comments,n,random_state)
    df_esp = filter_by_rates(df_esp,max_comments,n,random_state)
    
    df_esp.to_csv('data_esp.csv',index=False)
    df_por.to_csv('data_por.csv',index=False)
    
    
    