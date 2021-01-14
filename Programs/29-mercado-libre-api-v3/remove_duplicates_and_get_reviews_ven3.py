import sys
sys.path.append('python-sdk/lib/')
from meli import Meli
import pandas as pd

countries_ids = {'MLV':'Venezuela'}

n_tries = 5

def remove_duplicates_and_get_reviews():
    products_df = pd.concat([pd.read_csv('products/{}2.csv'.format(country))
                             for country in countries_ids.values()]).drop_duplicates(subset=['prod_id'])
    products_df = products_df.reset_index(drop=True)
    offset = 582600
    prod_len = len(products_df)
    products_df = products_df.iloc[offset:,:]
    meli = Meli(client_id=1212334,client_secret='a secret')
    reviews_dict = {'prod_id': [], 'cat_id': [], 'review_id': [], 'country': [],
                    'prod_title': [], 'reviewer_id': [], 'review_date': [],
                    'review_status': [], 'review_title': [], 'review_content': [],
                    'review_rate': [], 'review_likes': [], 'review_dislikes': []}
    
    for idx, (prod_id, category_id) in products_df.iterrows():
        print('{}/{} ({:.1f}%)'.format(idx,prod_len,idx/prod_len*100))
        prod_title = meli.get('items/{}'.format(prod_id)).json()['title']
        country = prod_id[:3]
        for i in range(n_tries):
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
            df_reviews.to_csv('./parts/reviews_ven_part{:06}.csv'.format(idx+1),index=False)
            reviews_dict = {'prod_id': [], 'cat_id': [], 'review_id': [], 'country': [],
                            'prod_title': [], 'reviewer_id': [], 'review_date': [],
                            'review_status': [], 'review_title': [], 'review_content': [],
                            'review_rate': [], 'review_likes': [], 'review_dislikes': []}
    


if __name__ == '__main__':
    remove_duplicates_and_get_reviews()
