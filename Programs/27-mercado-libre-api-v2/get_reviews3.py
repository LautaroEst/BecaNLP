import sys
import os
sys.path.append('python-sdk/lib/')
from meli import Meli
import pandas as pd
import re
import numpy as np

countries_ids = {'MLA':'Argentina', 'MCO':'Colombia', 'MPE':'Perú ', 'MLU':'Uruguay',
                 'MLC':'Chile', 'MLM':'Mexico', 'MLV':'Venezuela', 'MLB':'Brasil'}


def process_data():
    df_all = pd.read_csv('data_all2.csv')
    # Nos quedamos con las filas que no contienen caracteres basura
    char_vocab = df_all['review_content'].str.split('').explode().value_counts()
    chars_to_keep = set('abcçdefghijklmnñopqrstuvwxyzABCÇDEFGHIJKLMNÑOPQRSTUVWXYZ' + \
                        'áàâãäÁÀÂÃÄéèêëÉÈÊẼËíìîĩïÍÌÎĨÏóòôõöÓÒÔÕÖúùûũüÚÙÛŨÜ' + \
                        '.,;:%&$-!¡()?¿/\\"\'' + \
                        '0123456789' + \
                        ' \n\t')# VER los caracteres de IMDb para elegir con cual quedarme
    chars_to_remove = sorted(list(set(char_vocab.index) - chars_to_keep - {''}))
    df_all = df_all[-df_all['review_content'].str.contains('|'.join([re.escape(c) 
                                                                     for c in chars_to_remove]),regex=True,na=True)]
    

    # Fijamos una cantidad de máxima de comentarios por producto y sampleamos 
    # los reviews de los productos que tienen más de esa cantidad como un 
    # promedio ponderado del rate, cantidad de likes y de dislikes
    max_comments = 20
    indices = (-1 * df_all['review_rate'] + \
    .1 * df_all['review_likes'] + \
    .01 * df_all['review_dislikes']).sort_values(ascending=False).index
    df_sampled = df_all.loc[indices,:].groupby('prod_id').head(max_comments)
    
    # Separamos los datos por idioma y definimos los dos df separados
    df_por = df_sampled[df_sampled['country'] == 'MLB']
    df_esp = df_sampled[df_sampled['country'] != 'MLB']

    # Nos quedamos con n muestras al azar por categoría
    random_state = np.random.RandomState(1234)
    n = 25000
    df_esp = df_esp.iloc[np.hstack([random_state.choice(arr, n, replace=False) 
                           for arr in df_esp.groupby('review_rate').indices.values()]),:]
    df_esp = df_esp.reset_index(drop=True)
    df_esp = df_esp.iloc[random_state.permutation(len(df_esp)),:]
    df_esp = df_esp.reset_index(drop=True)
    
    df_por = df_por.iloc[np.hstack([random_state.choice(arr, n, replace=False) 
                           for arr in df_por.groupby('review_rate').indices.values()]),:]
    df_por = df_por.reset_index(drop=True)
    df_por = df_por.iloc[random_state.permutation(len(df_por)),:]
    df_por = df_por.reset_index(drop=True)
    
    df_esp.to_csv('reviews_esp.csv',index=False)
    df_por.to_csv('reviews_por.csv',index=False)


def merge_parts_and_drop_duplicated_comments():
    filenames = os.listdir('./parts/')
    df_all = pd.concat([pd.read_csv('./parts/' + file) for file in filenames], ignore_index=True)
    #df_all.to_csv('data_all.csv')
    df_all = df_all.drop_duplicates(subset=['review_content'])
    df_all.to_csv('data_all2.csv',index=False)


def get_reviews_from_products_list():
    products_df = pd.read_csv('products_list.csv') # la lista ya tiene los productos únicos (no duplicados)
    meli = Meli(client_id=1212334,client_secret='a secret')
    reviews_dict = {'prod_id': [], 'cat_id': [], 'review_id': [], 'country': [],
                    'prod_title': [], 'reviewer_id': [], 'review_date': [],
                    'review_status': [], 'review_title': [], 'review_content': [],
                    'review_rate': [], 'review_likes': [], 'review_dislikes': []}
    
    prod_len = len(products_df)
    for idx, (prod_id, category_id) in products_df.iterrows():
        print('{}/{} ({:.1f}%)'.format(idx,prod_len,idx/prod_len*100))
        prod_title = meli.get('items/{}'.format(prod_id)).json()['title']
        country = prod_id[:3]
        for i in range(5):
            try:
                reviews = meli.get('/reviews/search?item_id={}&limit=100&order_criteria=valorization'.format(prod_id)).json()['results']
                break
            except KeyError:
                print('Error')
        for review in reviews:
            reviews_dict['prod_id'].append(prod_id)
            reviews_dict['cat_id'].append(category_id)
            reviews_dict['country'].append(country)
            reviews_dict['prod_title'].append(prod_title)
            reviews_dict['review_id'].append(review['id'])
            reviews_dict['review_date'].append(review['date_created'])
            reviews_dict['review_status'].append(review['status'])
            reviews_dict['review_title'].append(review['title'])
            reviews_dict['review_content'].append(review['content'])
            reviews_dict['review_rate'].append(review['rate'])
            reviews_dict['review_likes'].append(review['likes'])
            reviews_dict['review_dislikes'].append(review['dislikes'])
            reviews_dict['reviewer_id'].append(review['reviewer_id'])
        if idx % 200 == 199:
            df_reviews = pd.DataFrame(reviews_dict)
            df_reviews.to_csv('./parts/reviews_data_part{:06}.csv'.format(idx+1),index=False)
            reviews_dict = {'prod_id': [], 'cat_id': [], 'review_id': [], 'country': [],
                            'prod_title': [], 'reviewer_id': [], 'review_date': [],
                            'review_status': [], 'review_title': [], 'review_content': [],
                            'review_rate': [], 'review_likes': [], 'review_dislikes': []}
    
def generate_products_list():
    data = ['{}.csv'.format(count) for count in countries_ids.values()]

    # Elegimos por valorización los productos:
    val_threshold = 1
    df = pd.concat([pd.read_csv('./data/' + file) for file in data],ignore_index=True)
    df['valorization'] = abs(df['valorization'])
    df = df[df['valorization'] >= val_threshold]
    df = df.reset_index(drop=True)
    df = df.iloc[df['valorization'].sort_values(ascending=False).index,:]
    df.loc[:,['product_id','category_id']].to_csv('products_list.csv',index=False)
    # Descartamos los repetidos en la lista:
    products_df = pd.read_csv('products_list.csv').drop_duplicates(subset=['product_id'])
    products_df = products_df.reset_index(drop=True)
    products_df.to_csv('products_list.csv')

    
if __name__ == '__main__':
    # Previamente se bajaron hasta 100 comentarios por producto
    # entre todos los productos de todas las categorías de todos 
    # los países disponibles en la base de ML. Ver get_reviews.py
    
    # generamos una lista de productos con los que más valorización tienen:
    #generate_products_list()
    
    # Volvemos a bajar hasta 100 comentarios de cada producto pero ahora 
    # lo hacemos a partir de una lista de productos que fue generada anteriormente:
    #get_reviews_from_products_list()
    
    # Creamos un único archivo con los comentarios:
    #merge_parts_and_drop_duplicated_comments()
    
    # Procesamos los datos y nos quedamos con los más relevantes (ver ml-api-v4.ipynb):
    process_data()



