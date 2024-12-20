"""
Module for generating dataset.
"""

import os
import csv
import logging
import numpy as np

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
    STR_CICLISTES = 'data/ciclistes.csv'

    # Crear directori si no existeix
    os.makedirs(os.path.dirname(STR_CICLISTES), exist_ok=True)

    # Definició de categories
    MU_P_BE = 3240  # Mitjana temps pujada bons escaladors
    MU_P_ME = 4268  # Mitjana temps pujada mals escaladors
    MU_B_BB = 1440  # Mitjana temps baixada bons baixadors
    MU_B_MB = 2160  # Mitjana temps baixada mals baixadors
    SIGMA = 240     # Desviació estàndard (4 min)

    DICC = [
        {"name": "BEBB", "mu_p": MU_P_BE, "mu_b": MU_B_BB, "sigma": SIGMA},
        {"name": "BEMB", "mu_p": MU_P_BE, "mu_b": MU_B_MB, "sigma": SIGMA},
        {"name": "MEBB", "mu_p": MU_P_ME, "mu_b": MU_B_BB, "sigma": SIGMA},
        {"name": "MEMB", "mu_p": MU_P_ME, "mu_b": MU_B_MB, "sigma": SIGMA}
    ]

    # Generar dataset
    NUM_CICLISTES = 100
    IND_INICIAL = 1
    generated_dataset = generar_dataset(NUM_CICLISTES, IND_INICIAL, DICC)

    # Escriure el dataset al fitxer CSV
    with open(STR_CICLISTES, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["id", "categoria", "temps_pujada", "temps_baixada"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(generated_dataset)

    logging.info("S'ha generat el fitxer '%s' amb èxit.", STR_CICLISTES)
    