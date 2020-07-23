import sys
sys.path.append('python-sdk/lib/')
from meli import Meli
import pandas as pd
from tqdm import tqdm


def main():
    
    meli = Meli(client_id=1234, client_secret="a secret")
    countries_ids = {d['name']:d['id'] for d in meli.get('sites/').json()}
    for name, country_id in countries_ids.items():
        reviews_dict = {'product_id': [], 'category_id': [], 'content': [], 'rate': [], 'valorization': []}
        categories = meli.get('sites/{}/categories/all'.format(country_id)).json()
        print('País:',name)
        for category_id in tqdm(categories.keys()):
            products = meli.get('sites/{}/search?category={}'.format(country_id,category_id)).json()['results']
            for product in products:
                product_id = product['id']
                reviews = meli.get('/reviews/search?item_id={}&limit=5&order_criteria=valorization'.format(product_id)).json()['results']
                for review in reviews:
                    reviews_dict['product_id'].append(product_id)
                    reviews_dict['category_id'].append(category_id)
                    reviews_dict['content'].append(review['content'])
                    reviews_dict['rate'].append(review['rate'])
                    reviews_dict['valorization'].append(review['valorization'])
        df = pd.DataFrame(reviews_dict)
        df.to_csv('./{}.csv'.format(name),index=False)
        
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
