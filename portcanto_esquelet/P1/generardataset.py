import os
import logging
import numpy as np
import csv

# Configuració de logging
logging.basicConfig(level=logging.INFO)

def generar_dataset(num, ind, dicc):
    """
    Genera un dataset de ciclistes amb temps de pujada i baixada.
    
    Args:
        num (int): Nombre de ciclistes a generar.
        ind (int): Índex inicial per als ciclistes.
        dicc (list): Llista de diccionaris amb informació sobre les categories.
        
    Returns:
        list: Una llista de diccionaris amb els temps generats per cada ciclista.
    """
    dataset = []
    for i in range(num):
        # Assignar categoria aleatòria
        categoria = np.random.choice(dicc)
        
        # Generar temps de pujada i baixada
        temps_pujada = np.random.normal(loc=categoria["mu_p"], scale=categoria["sigma"])
        temps_baixada = np.random.normal(loc=categoria["mu_b"], scale=categoria["sigma"])
        
        # Afegir dades al dataset
        dataset.append({
            "id": ind + i,
            "categoria": categoria["name"],
            "temps_pujada": max(0, int(temps_pujada)),  # Temps en segons (no negatiu)
            "temps_baixada": max(0, int(temps_baixada)) # Temps en segons (no negatiu)
        })
    return dataset

if __name__ == "__main__":
    str_ciclistes = '../data/ciclistes.csv'

    # Crear directori si no existeix
    os.makedirs(os.path.dirname(str_ciclistes), exist_ok=True)

    # Definició de categories
    mu_p_be = 3240  # Mitjana temps pujada bons escaladors
    mu_p_me = 4268  # Mitjana temps pujada mals escaladors
    mu_b_bb = 1440  # Mitjana temps baixada bons baixadors
    mu_b_mb = 2160  # Mitjana temps baixada mals baixadors
    sigma = 240      # Desviació estàndard (4 min)

    dicc = [
        {"name": "BEBB", "mu_p": mu_p_be, "mu_b": mu_b_bb, "sigma": sigma},
        {"name": "BEMB", "mu_p": mu_p_be, "mu_b": mu_b_mb, "sigma": sigma},
        {"name": "MEBB", "mu_p": mu_p_me, "mu_b": mu_b_bb, "sigma": sigma},
        {"name": "MEMB", "mu_p": mu_p_me, "mu_b": mu_b_mb, "sigma": sigma}
    ]

    # Generar dataset
    num_ciclistes = 100
    ind_inicial = 1
    dataset = generar_dataset(num_ciclistes, ind_inicial, dicc)

    # Escriure el dataset al fitxer CSV
    with open(str_ciclistes, mode='w', newline='') as csvfile:
        fieldnames = ["id", "categoria", "temps_pujada", "temps_baixada"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(dataset)

    logging.info("S'ha generat el fitxer '%s' amb èxit.", str_ciclistes)
