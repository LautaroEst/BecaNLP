# DESPUÉS DE GET_PRODUCTS_LIST SE ELIMINAN LOS PRODUCTOS DE LOS 
# QUE YA SE TIENEN REVIEWS.

import pandas as pd

countries = ['Argentina', 'Colombia', 'Perú', 'Uruguay', 'Chile', 
			'Mexico', 'Venezuela', 'Brasil']


def main():
	for country in countries:
		uniq_prod = pd.read_csv('./products/{}.csv'.format(country)).drop_duplicates(subset=['prod_id'])
		uniq_prod = uniq_prod.reset_index(drop=True)

		prods_orig = pd.read_csv('../27-mercado-libre-api-v2/products/{}.csv'.format(country))['prod_id'].unique().tolist()

		uniq_prod = uniq_prod[-(uniq_prod['prod_id'].isin(prods_orig))].reset_index(drop=True)
		uniq_prod.to_csv('result.csv',index=False)

def print_all():
	for country in countries:
		df = pd.read_csv('./products/{}2.csv'.format(country))
		print(country+':',len(df))


if __name__ == "__main__":
	main()
	print_all()
	


