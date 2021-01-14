import numpy as np
import pandas as pd
import os

root_path = './28-melisa/melisa/'

def filenames_generator(parts_directories):

	rs = np.random.RandomState(127361824)
	for directory in parts_directories:
		print('Extracting from {}'.format(directory))
		list_parts = np.array(os.listdir(directory))
		num_parts = len(list_parts)
		for filename in rs.choice(list_parts,num_parts,replace=False):
			yield '{}{}'.format(directory,filename)

def drop_dups(df):
	# Elimino los duplicados y los que tienen valores faltantes:
	df = df.drop_duplicates(subset=['review_id'])\
		.reset_index(drop=True).dropna()

	assert df['prod_id'].apply(type).eq(str).all()
	assert df['cat_id'].apply(type).eq(str).all()
	assert df['review_id'].apply(type).eq(int).all()
	assert df['country'].isin(['MLB','MLA','MLM',
	'MLU','MCO','MLC','MLV','MPE']).all()
	assert df['prod_title'].apply(type).eq(str).all()
	assert df['reviewer_id'].apply(type).eq(int).all()
	assert df['review_date'].apply(type).eq(str).all()
	assert df['review_status'].apply(type).eq(str).all()
	df['review_title'] = df['review_title'].apply(str)
	assert df['review_title'].apply(type).eq(str).all()
	assert df['review_content'].apply(type).eq(str).all()
	assert df['review_rate'].isin([1, 2, 3, 4, 5]).all()
	assert df['review_likes'].apply(type).eq(int).all()
	assert df['review_dislikes'].apply(type).eq(int).all()

	print('Cantidad de reviews únicos descargados:',len(df))

	# Cambio todos los espacios por espacios simples 
	# y vuelvo a eliminar duplicados:
	df['review_content'] = df['review_content']\
		.str.replace(r'\s+',' ',regex=True)
	df['review_title'] = df['review_title'].str.replace(r'\s+',' ',regex=True)
	df = df.drop_duplicates(subset=['review_content',
				'review_title','review_rate']).reset_index(drop=True)
	print('Cantidad de reviews con contenido, título y rate únicos:',len(df))

	return df


def get_csv(filenames_gen,csv_filename):
	df = pd.concat([pd.read_csv(filename,lineterminator='\n',sep=',') \
		for filename in filenames_gen], ignore_index=True)
	df = drop_dups(df)

	# Guardo en un csv los campos más importantes:
	df.to_csv(root_path + csv_filename,index=False)
	print('Guardado OK.')


def merge_dfs(csvs_lists):
	df = pd.concat([pd.read_csv(csv_filename,lineterminator='\n',sep=',') \
		for csv_filename in csvs_lists], ignore_index=True)
	df = drop_dups(df)
	return df


def get_parts_in_csv():

	csvs = {'orig.csv': ['parts/', 'parts-2/'],
			'ven.csv': ['ven_parts/'],
			'per.csv': ['peru_parts/']}

	generators = {csvfilename: filenames_generator([root_path + 'all_parts/' + part for part in directories])\
					for csvfilename,directories in csvs.items()}

	for filename,gen in generators.items():
		print('Generando archivo {}...'.format(filename))
		get_csv(gen,filename)
		print()
				
	print('Mergeando todos en un único csv...')
	df = merge_dfs([root_path + csvfiles for csvfiles in csvs.keys()])
	filename = root_path + 'reviews_all.csv'
	df.to_csv(filename,index=False)


if __name__ == "__main__":
	get_parts_in_csv()