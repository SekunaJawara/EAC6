"""
@ IOC - CE IABD
"""
import os
import logging
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_dataset(path):
    """
	Carrega el dataset de registres dels ciclistes

	arguments:
		path -- dataset

	Returns: dataframe
	"""
    df = pd.read_csv(path)
    return df

def EDA(df):
    """
	Exploratory Data Analysis del dataframe

	arguments:
		df -- dataframe

	Returns: None
	"""
# Imprimir título y luego DataFrame y otras funciones, con títulos para cada sección

    print('EDA\n')  # Título general
    print('Información del DataFrame:\n')
    df.info()  # df.info() imprime la información del DataFrame

    print('\nDescripción estadística del DataFrame:\n')
    print(df.describe())  # Estadísticas descriptivas

    print('\nPrimeras filas del DataFrame:\n')
    print(df.head())  # Primeras 5 filas del DataFrame

    print('\nValores nulos por columna:\n')
    print(df.isnull().sum())  # Cantidad de valores nulos por columna

    print('\nTipos de datos de las columnas:\n')
    print(df.dtypes)  # Tipos de datos de cada columna

    return None

def clean(df):
    """
	Elimina les columnes que no són necessàries per a l'anàlisi dels clústers

	arguments:
		df -- dataframe

	Returns: dataframe
	"""
    df = df.drop('id', axis=1)
    print('Columna id eliminada')
    print(df.head())
    return df

def extract_true_labels(df):
    """
	Guardem les etiquetes dels ciclistes (BEBB, ...)

	arguments:
		df -- dataframe

	Returns: numpy ndarray (true labels)
	"""
    true_labels = df['categoria'].values
    return true_labels


def visualitzar_pairplot(df):
    """
	Genera una imatge combinant entre sí tots els parells d'atributs.
	Serveix per apreciar si es podran trobar clústers.

	arguments:
		df -- dataframe

	Returns: None
	"""
    sns.pairplot(df)
    plt.savefig('../img/output_plot.png')
    return None

def clustering_kmeans(data, n_clusters=4):
    """
	Crea el model KMeans de sk-learn, amb 4 clusters (estem cercant 4 agrupacions)
	Entrena el model

	arguments:
		data -- les dades: tp i tb

	Returns: model (objecte KMeans)
	"""
    clustering_model = KMeans(n_clusters=n_clusters)
    clustering_model.fit(data)
    return clustering_model
    return None

def visualitzar_clusters(data, labels):
    """
    Visualitza els clusters en diferents colors. Provem diferents combinacions de parells d'atributs

    arguments:
        data -- el dataset sobre el qual hem entrenat
        labels -- l'array d'etiquetes a què pertanyen les dades (hem assignat les dades a un dels 4 clústers)
    
    Returns: None
    """
    colormap = ['r', 'g', 'b', 'y']
    plt.figure()
    for cluster_num in range(4):
        cluster_points = data[labels == cluster_num]
    plt.scatter(
		cluster_points.iloc[:, 0], 
		cluster_points.iloc[:, 1], 
		color=colormap[cluster_num], 
		label=f'Cluster {cluster_num}'
	)
    plt.legend()
    plt.savefig('../img/clusters_plot.png')
    plt.close()

    return None

def associar_clusters_patrons(tipus, model):
	"""
	Associa els clústers (labels 0, 1, 2, 3) als patrons de comportament (BEBB, BEMB, MEBB, MEMB).
	S'han trobat 4 clústers però aquesta associació encara no s'ha fet.

	arguments:
	tipus -- un array de tipus de patrons que volem actualitzar associant els labels
	model -- model KMeans entrenat

	Returns: array de diccionaris amb l'assignació dels tipus als labels
	"""
	# proposta de solució

	dicc = {'tp':0, 'tb': 1}

	logging.info('Centres:')
	for j in range(len(tipus)):
		logging.info('{:d}:\t(tp: {:.1f}\ttb: {:.1f})'.format(j, model.cluster_centers_[j][dicc['tp']], model.cluster_centers_[j][dicc['tb']]))

	# Procés d'assignació
	ind_label_0 = -1
	ind_label_1 = -1
	ind_label_2 = -1
	ind_label_3 = -1

	suma_max = 0
	suma_min = 50000

	for j, center in enumerate(clustering_model.cluster_centers_):
		suma = round(center[dicc['tp']], 1) + round(center[dicc['tb']], 1)
		if suma_max < suma:
			suma_max = suma
			ind_label_3 = j
		if suma_min > suma:
			suma_min = suma
			ind_label_0 = j

	tipus[0].update({'label': ind_label_0})
	tipus[3].update({'label': ind_label_3})

	lst = [0, 1, 2, 3]
	lst.remove(ind_label_0)
	lst.remove(ind_label_3)

	if clustering_model.cluster_centers_[lst[0]][0] < clustering_model.cluster_centers_[lst[1]][0]:
		ind_label_1 = lst[0]
		ind_label_2 = lst[1]
	else:
		ind_label_1 = lst[1]
		ind_label_2 = lst[0]

	tipus[1].update({'label': ind_label_1})
	tipus[2].update({'label': ind_label_2})

	logging.info('\nHem fet l\'associació')
	logging.info('\nTipus i labels:\n%s', tipus)
	return tipus

import os

def generar_informes(df, tipus):
    """
    Generació dels informes a la carpeta informes/. Tenim un dataset de ciclistes i 4 clústers, i generem
    4 fitxers de ciclistes per cadascun dels clústers

    arguments:
        df -- dataframe
        tipus -- objecte que associa els patrons de comportament amb els labels dels clústers

    Returns: None
    """
    # Comprovar que la carpeta "informes/" existeix, si no, crear-la
    informes_dir = '../informes/'
    os.makedirs(informes_dir, exist_ok=True)

    # Generar un fitxer per a cada clúster
    for tipus_item in tipus:
        label = tipus_item['label']
        nom_patro = tipus_item['name']
        
        # Filtrar el dataframe per a aquest clúster
        cluster_df = df[df['label'] == label]
        
        # Nom del fitxer per al clúster actual
        fitxer_cluster = os.path.join(informes_dir, f"{nom_patro}_cluster_{label}.csv")
        
        # Guardar el dataframe filtrat en un fitxer CSV
        cluster_df.to_csv(fitxer_cluster, index=False)
        logging.info('Informe generat per al clúster %s: %s', nom_patro, fitxer_cluster)

    logging.info('S\'han generat els informes en la carpeta informes/\n')
    return None


def nova_prediccio(dades, model):
    """
    Passem nous valors de ciclistes, per tal d'assignar aquests valors a un dels 4 clústers

    arguments:
        dades -- llista de llistes, que segueix l'estructura 'id', 'tp', 'tb', 'tt'
        model -- clustering model
    Returns: (dades agrupades, prediccions del model)
    """
    # Convertir les dades a un DataFrame
    df_nous_ciclistes = pd.DataFrame(dades, columns=['id', 'temps_pujada', 'temps_baixada', 'tt'])

    # Fer la predicció amb el model KMeans
    prediccions = model.predict(df_nous_ciclistes[['temps_pujada', 'temps_baixada']])

    # Afegir les prediccions al DataFrame
    df_nous_ciclistes['cluster'] = prediccions

    return df_nous_ciclistes, prediccions

# ----------------------------------------------

if __name__ == "__main__":

	path_dataset = '../data/ciclistes.csv'
	"""
	TODO:
	load_dataset
	EDA
	clean
	extract_true_labels
	eliminem el tipus, ja no interessa .drop('tipus', axis=1)
	visualitzar_pairplot
	clustering_kmeans
	pickle.dump(...) guardar el model
	mostrar scores i guardar scores
	visualitzar_clusters
	"""
	df = load_dataset(path_dataset)
	EDA(df)
	df = clean(df)
	true_labels = extract_true_labels(df)
	df = df.drop('categoria', axis=1)
	visualitzar_pairplot(df)
	clustering_model = clustering_kmeans(df)
	with open('../model/clustering_model.pkl', 'wb') as f:
		pickle.dump(clustering_model, f)
	# scores
	hom_score = homogeneity_score(true_labels, clustering_model.labels_)
	com_score = completeness_score(true_labels, clustering_model.labels_)
	v_score = v_measure_score(true_labels, clustering_model.labels_)
	logging.info('homogeneity_score: %s', hom_score)
	logging.info('completeness_score: %s', com_score)
	logging.info('v_measure_score: %s', v_score)

	# Crear el diccionari de mètriques
	scores = {
		"homogeneity_score": hom_score,
		"completeness_score": com_score,
		"v_measure_score": v_score
	}

	# Guardar el diccionari en un fitxer scores.pkl
	with open('../model/scores.pkl', 'wb') as f:
		pickle.dump(scores, f)

	logging.info('Les mètriques s\'han guardat a model/scores.pkl')
	visualitzar_clusters(df, clustering_model.labels_)
	


	


	# array de diccionaris que assignarà els tipus als labels
	tipus = [{'name': 'BEBB'}, {'name': 'BEMB'}, {'name': 'MEBB'}, {'name': 'MEMB'}]

	"""
	afegim la columna label al dataframe
	associar_clusters_patrons(tipus, clustering_model)
	guardem la variable tipus a model/tipus_dict.pkl
	generar_informes
	"""
	df['label'] = clustering_model.labels_
	tipus = associar_clusters_patrons(tipus, clustering_model)
	with open('../model/tipus_dict.pkl', 'wb') as f:
		pickle.dump(tipus, f)
	generar_informes(df, tipus)

	
	# Classificació de nous valors
	nous_ciclistes = [
		[500, 3230, 1430, 4670], # BEBB
		[501, 3300, 2120, 5420], # BEMB
		[502, 4010, 1510, 5520], # MEBB
		[503, 4350, 2200, 6550] # MEMB
	]

	"""
	nova_prediccio

	#Assignació dels nous valors als tipus
	for i, p in enumerate(pred):
		t = [t for t in tipus if t['label'] == p]
		logging.info('tipus %s (%s) - classe %s', df_nous_ciclistes.index[i], t[0]['name'], p)
	"""

	#Asignar los valores devueltos a dos variables
	df_nous_ciclistes, prediccions = nova_prediccio(nous_ciclistes, clustering_model)

	# Imprimir los resultados
	print(df_nous_ciclistes)
	print(prediccions)

