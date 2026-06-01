
# Apprentissage automatique non supervisé

## Présentation
Ce projet explore des techniques de clustering non supervisé appliquées à un jeu de données d’hymnes nationaux. L’objectif est de regrouper les hymnes selon leurs caractéristiques textuelles et d’en visualiser les résultats.

## Contenu du projet
- **Mini_projet_apprentissage_non_supervisé.ipynb** : Notebook principal contenant tout le code, les analyses et les visualisations.
- **Mini_projet_apprentissage_non_supervise.ipynb** : Copie du notebook avec nom ASCII (compatibilité).
- **Mini_projet_apprentissage_non_supervise_v2.ipynb** : Deuxième copie canonique du notebook (secours).
- **Mini_projet_apprentissage_non_supervise.md** : Export Markdown du notebook, lisible directement sur GitHub.
- **anthems.csv** : Jeu de données utilisé pour l’étude.
- **requirements.txt** : Liste des dépendances Python nécessaires.
- **.gitignore** : Fichiers/dossiers à exclure du versionnement.

## Lecture sur GitHub
Si la prévisualisation des fichiers .ipynb affiche "An error occurred" sur GitHub, utilisez la version Markdown ci-dessous :

- Mini_projet_apprentissage_non_supervise.md

Le notebook source reste disponible pour exécution locale :

- Mini_projet_apprentissage_non_supervise.ipynb
- Mini_projet_apprentissage_non_supervise_v2.ipynb

Accès alternatif (si la prévisualisation GitHub des ipynb est en erreur) :

- nbviewer : https://nbviewer.org/github/Elie224/MLUP---Apprentissage-automatique-non-supervis-/blob/main/Mini_projet_apprentissage_non_supervise.ipynb
- Colab : https://colab.research.google.com/github/Elie224/MLUP---Apprentissage-automatique-non-supervis-/blob/main/Mini_projet_apprentissage_non_supervise.ipynb

## Méthodologie
- Prétraitement des données textuelles
- Application de deux méthodes de clustering :
	- KMeans
	- Clustering hiérarchique
- Réduction de dimension (PCA) pour la visualisation
- Analyse et interprétation des clusters

## Installation
Assurez-vous d’avoir Python 3.7 ou plus récent. Installez les dépendances avec :

```bash
pip install -r requirements.txt
```

## Utilisation
Ouvrez le notebook dans Jupyter ou VS Code, puis exécutez les cellules dans l’ordre pour reproduire l’analyse et les visualisations.

## Exemples de visualisations
- Dendrogrammes
- Nuages de mots
- Graphiques de clusters (PCA)

## Auteurs
- [Votre nom ici]

## Licence
Ce projet est open-source, sous licence MIT.