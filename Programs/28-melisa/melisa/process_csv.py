import sys
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np


root_path = './28-melisa/melisa/'
MeLi_path = './29-mercado-libre-api-v3/python-sdk/lib/'
sys.path.append(MeLi_path)
from meli import Meli
from get_parts_in_csv import drop_dups


# Se ejecutó --> dict(df['category'].value_counts())
# Se obtuvo el diccionario de ocurrencias de las categorías.
# Se usa eso para generar un conversor de categorías menos específico:

generalize_categories = {

	'Hogar, Muebles y Jardín': 'Hogar / Casa',
	'Casa, Móveis e Decoração': 'Hogar / Casa',
	'Herramientas y Construcción': 'Hogar / Casa',
	'Industrias y Oficinas': 'Hogar / Casa',
	'Ferramentas e Construção': 'Hogar / Casa',
	'Bebés': 'Hogar / Casa',
	'Animales y Mascotas': 'Hogar / Casa',
	'Hogar y Muebles': 'Hogar / Casa',
	'Bebês': 'Hogar / Casa',
	'Animais': 'Hogar / Casa',
	'Indústria e Comércio': 'Hogar / Casa',
	'Industrias': 'Hogar / Casa',

	'Computación': 'Tecnología y electrónica / Tecnologia e electronica',
	'Accesorios para Vehículos': 'Tecnología y electrónica / Tecnologia e electronica',
	'Acessórios para Veículos': 'Tecnología y electrónica / Tecnologia e electronica',
	'Electrónica, Audio y Video': 'Tecnología y electrónica / Tecnologia e electronica',
	'Electrodomésticos y Aires Ac.': 'Tecnología y electrónica / Tecnologia e electronica',
	'Celulares y Telefonía': 'Tecnología y electrónica / Tecnologia e electronica',
	'Informática': 'Tecnología y electrónica / Tecnologia e electronica',
	'Eletrônicos, Áudio e Vídeo': 'Tecnología y electrónica / Tecnologia e electronica',
	'Electrodomésticos': 'Tecnología y electrónica / Tecnologia e electronica',
	'Eletrodomésticos': 'Tecnología y electrónica / Tecnologia e electronica',
	'Celulares y Teléfonos': 'Tecnología y electrónica / Tecnologia e electronica',
	'Cámaras y Accesorios': 'Tecnología y electrónica / Tecnologia e electronica',
	'Consolas y Videojuegos': 'Tecnología y electrónica / Tecnologia e electronica',
	'Celulares e Telefones': 'Tecnología y electrónica / Tecnologia e electronica',
	'Câmeras e Acessórios': 'Tecnología y electrónica / Tecnologia e electronica',

	'Deportes y Fitness': 'Salud, ropa y cuidado personal / Saúde, roupas e cuidado pessoal',
	'Belleza y Cuidado Personal': 'Salud, ropa y cuidado personal / Saúde, roupas e cuidado pessoal',  
	'Calçados, Roupas e Bolsas': 'Salud, ropa y cuidado personal / Saúde, roupas e cuidado pessoal',
	'Esportes e Fitness': 'Salud, ropa y cuidado personal / Saúde, roupas e cuidado pessoal',
	'Ropa y Accesorios': 'Salud, ropa y cuidado personal / Saúde, roupas e cuidado pessoal',
	'Salud y Equipamiento Médico': 'Salud, ropa y cuidado personal / Saúde, roupas e cuidado pessoal',
	'Beleza e Cuidado Pessoal': 'Salud, ropa y cuidado personal / Saúde, roupas e cuidado pessoal',
	'Ropa, Bolsas y Calzado': 'Salud, ropa y cuidado personal / Saúde, roupas e cuidado pessoal',
	'Saúde': 'Salud, ropa y cuidado personal / Saúde, roupas e cuidado pessoal',
	'Vestuario y Calzado': 'Salud, ropa y cuidado personal / Saúde, roupas e cuidado pessoal',
	'Ropa, Calzados y Accesorios': 'Salud, ropa y cuidado personal / Saúde, roupas e cuidado pessoal',
	'Ropa, Zapatos y Accesorios': 'Salud, ropa y cuidado personal / Saúde, roupas e cuidado pessoal',
	'Estética y Belleza': 'Salud, ropa y cuidado personal / Saúde, roupas e cuidado pessoal',

	'Juegos y Juguetes': 'Arte y entretenimiento / Arte e Entretenimiento',
	'Brinquedos e Hobbies': 'Arte y entretenimiento / Arte e Entretenimiento',
	'Arte, Librería y Mercería': 'Arte y entretenimiento / Arte e Entretenimiento',
	'Instrumentos Musicales': 'Arte y entretenimiento / Arte e Entretenimiento',
	'Arte, Papelaria e Armarinho': 'Arte y entretenimiento / Arte e Entretenimiento',
	'Arte, Papelería y Mercería': 'Arte y entretenimiento / Arte e Entretenimiento',
	'Joyas y Relojes': 'Arte y entretenimiento / Arte e Entretenimiento',
	'Instrumentos Musicais': 'Arte y entretenimiento / Arte e Entretenimiento',
	'Games': 'Arte y entretenimiento / Arte e Entretenimiento',
	'Joias e Relógios': 'Arte y entretenimiento / Arte e Entretenimiento',
	'Souvenirs, Cotillón y Fiestas': 'Arte y entretenimiento / Arte e Entretenimiento',
	'Festas e Lembrancinhas': 'Arte y entretenimiento / Arte e Entretenimiento',
	'Recuerdos, Cotillón y Fiestas': 'Arte y entretenimiento / Arte e Entretenimiento',
	'Antigüedades y Colecciones': 'Arte y entretenimiento / Arte e Entretenimiento',
	'Libros, Revistas y Comics': 'Arte y entretenimiento / Arte e Entretenimiento',
	'Relojes y Joyas': 'Arte y entretenimiento / Arte e Entretenimiento',
	'Antiguidades e Coleções': 'Arte y entretenimiento / Arte e Entretenimiento',
	'Arte, Librería y Cordonería': 'Arte y entretenimiento / Arte e Entretenimiento',
	'Recuerdos, Piñatería y Fiestas': 'Arte y entretenimiento / Arte e Entretenimiento',
	'Música, Películas y Series': 'Arte y entretenimiento / Arte e Entretenimiento',
	'Música, Filmes e Seriados': 'Arte y entretenimiento / Arte e Entretenimiento',
	'Relojes, Joyas y Bisutería': 'Arte y entretenimiento / Arte e Entretenimiento',
	'Música y Películas': 'Arte y entretenimiento / Arte e Entretenimiento',
	'Livros, Revistas e Comics': 'Arte y entretenimiento / Arte e Entretenimiento',

	'Alimentos y Bebidas': 'Alimentos y Bebidas / Alimentos e Bebidas',
	'Alimentos e Bebidas': 'Alimentos y Bebidas / Alimentos e Bebidas',

	'Servicios': np.nan,
	'Serviços': np.nan,
	'Agro': np.nan,
	'Otras categorías': np.nan,
	'Mais Categorias': np.nan,
	'Otras Categorías': np.nan,
	'Ingressos': np.nan,
	'Entradas para Eventos': np.nan,
	'Boletas para Espectáculos': np.nan,
	'Autos, Motos y Otros': np.nan

}

countries = ['MLB','MLA','MLM','MLU','MCO','MLC','MLV','MPE']

rates = [1, 2, 3, 4, 5]

cat2n = {

	'Tecnología y electrónica / Tecnologia e electronica': [14396,11189,9350,1473,3237,3188,656,316],
	'Salud, ropa y cuidado personal / Saúde, roupas e cuidado pessoal': [16032,11399,9522,1228,3260,2295,173,211],
	'Hogar / Casa': [19477,16601,8613,2175,2321,3188,182,157],
	'Arte y entretenimiento / Arte e Entretenimiento': [4572,2710,1789,253,487,380,72,30],
	'Alimentos y Bebidas / Alimentos e Bebidas': [386,462,467,29,22,86,2,2] 

}


def get_orig_cat_name():
	filename = root_path + 'reviews_all.csv'
	df = pd.read_csv(filename)

	all_categories = pd.read_csv(root_path + 'catid2catname.csv',index_col='cat_id').to_dict()['category']
	df['category'] = df['cat_id'].map(all_categories)
	return df.dropna().reset_index(drop=True)


def sort_by_score(df):
    score_map = {2:1., 1:.75, 3:.5, 4:.25, 5:0.}
    df['rate_score'] = df['review_rate'].map(score_map)
    diff = df['review_likes'] - df['review_dislikes']
    vals = np.log(diff - diff.min() + 1)
    vals_norm = (vals - vals.min()) / (vals.max()-vals.min())
    lenghts = np.maximum(0,df['review_content'].str.len()-50)
    lenghts_norm = (lenghts - lenghts.min()) / (lenghts.max() - lenghts.min())
    df['val_score'] = vals_norm + .5 * lenghts_norm
    df = df.sort_values(by=['rate_score','val_score'],ascending=[False,False])
    return df.drop(['rate_score','val_score'], axis=1)


def mask(df,ctry,cat,rate):
    return ((df['country'] == ctry) & (df['category'] == cat) & (df['review_rate'] == rate))


def sample_and_get_indices(df,n):
    diff = df['review_likes'] - df['review_dislikes']
    vals = np.log(diff - diff.min() + 1)
    vals_norm = (vals - vals.min()) / (vals.max()-vals.min())
    lenghts = np.maximum(0,df['review_content'].str.len()-50)
    lenghts_norm = (lenghts - lenghts.min()) / (lenghts.max() - lenghts.min())
    df['score'] = vals_norm + .5 * lenghts_norm
    df = df.sort_values(by=['score'],ascending=False)
    return df.iloc[:n,:].index.tolist()


def process_csv():
	print('Generalizando categorías...')
	df = get_orig_cat_name()
	df['category'] = df['category'].map(generalize_categories)
	df = df.dropna().reset_index(drop=True)
	print('Limitando por cantidad de productos...')
	df = sort_by_score(df)
	df = df.groupby(['prod_id']).head(30)
	df = df.reset_index(drop=True)

	print('Sampleando los reviews por categoría, país y rate...')
	indices = []
	for cat, n in cat2n.items():
		for rate in rates:
			for i, country in enumerate(countries):
				df_new = df[mask(df,country,cat,rate)]
				idx = sample_and_get_indices(df_new,n[i])
				indices.extend(idx)
		print('Cantidad de índices hasta ahora:',len(indices))
	df = df.loc[indices,:].reset_index(drop=True)
	
	#df.to_csv(root_path + 'reviews_sampled_full.csv',index=False)
	df = drop_dups(df)

	df_esp = df[df['country']!='MLB']
	df_esp.to_csv(root_path + 'reviews_esp_full.csv',index=False)

	df_por = df[df['country']=='MLB']
	df_por.to_csv(root_path + 'reviews_por_full.csv',index=False)


if __name__ == "__main__":
	process_csv()
