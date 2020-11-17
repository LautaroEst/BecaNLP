import sys
import os
sys.path.append('python-sdk/lib/')
from meli import Meli
import pandas as pd
from tqdm import tqdm
import json

countries_ids = {'MLA':'Argentina', 'MCO':'Colombia', 'MPE':'Perú', 'MLU':'Uruguay',
                 'MLC':'Chile', 'MLM':'Mexico', 'MLV':'Venezuela', 'MLB':'Brasil'}

n_tries = 5

def generate_products_list():
    meli = Meli(client_id=1234, client_secret="a secret")
    products_and_categories = {'prod_id': [], 'cat_id': []}
    for country_id, country in countries_ids.items():
        try:
            print('País:',country)
            # Obtengo las categorías del país:
            for i in range(n_tries):
                try:
                    categories = meli.get('sites/{}/categories/all'.format(country_id)).json()
                    break
                except json.decoder.JSONDecodeError:
                    print('Error 1')
            # para cada categoría, obtengo sus productos
            for category_id in tqdm(categories.keys()):
                for i in range(n_tries):
                    try:
                        products = meli.get('sites/{}/search?category={}'.format(country_id,category_id)).json()['results']
                        break
                    except (KeyError, json.decoder.JSONDecodeError) as e:
                        print('Error 2')
                products_and_categories['prod_id'].extend([product['id'] for product in products])
                products_and_categories['cat_id'].extend([category_id] * len(products))
        
        except KeyboardInterrupt:
            pass
            
        pd.DataFrame(products_and_categories).to_csv('./products/{}.csv'.format(country),index=False)
        
        
if __name__ == '__main__':
    generate_products_list()
