```python
%pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud plotly scipy

```

# Import des bibliothèques


```python
# Manipulation de données
import pandas as pd
import numpy as np

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# NLP (traitement de texte)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Hiérarchique
from scipy.cluster.hierarchy import dendrogram, linkage

# Nuage de mots
from wordcloud import WordCloud

# Carte interactive
import plotly.express as px
import plotly.io as pio

# Force explicit renderers in VS Code/Jupyter outputs
%matplotlib inline
pio.renderers.default = "vscode"

# Regex
import re
```


```python
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
```


```python
from pathlib import Path

# Load the dataset from the workspace regardless of machine-specific absolute paths
df = pd.read_csv(Path("anthems.csv"))
```


```python
# Aperçu des données
df.head()
```


```python
df.info()
```


```python
df.isnull().sum()
```


```python
df = df.dropna()

```


```python
df.columns
```

# 1) Pré-traitement des textes (nettoyage + lemmatisation + tokenisation)

### a-) Nettoyage superficiel


```python
def clean_text(text):
    text = str(text).lower()  # minuscules
    text = re.sub(r"\d+", "", text)  # chiffres
    text = re.sub(r"[^\w\s]", " ", text)  # ponctuation
    text = re.sub(r"\s+", " ", text).strip()  # espaces multiples
    return text
```


```python
# Application sur le dataset
df["anthem_clean"] = df["Anthem"].apply(clean_text)
df[["Anthem", "anthem_clean"]].head()
```

### Interprétation
La colonne anthem_clean montre une version simplifiée et uniformisée des hymnes : toute la ponctuation a été retirée, les majuscules ont été transformées en minuscules et les espaces ont été nettoyés. Le sens général du texte reste intact, mais il est désormais beaucoup plus propre et prêt pour les étapes suivantes du traitement automatique du langage, comme la tokenisation, la lemmatisation ou le clustering.

### b-) Nettoyage sémantique (lemmatisation).


```python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

df["anthem_lemma"] = df["anthem_clean"].apply(lemmatize_text)
df[["anthem_clean", "anthem_lemma"]].head()
```

### Interprétation
Nous avons appliqué une lemmatisation à l’aide du WordNetLemmatizer. Cette méthode ramène chaque mot à sa forme de base réelle (lemme), ce qui permet d’obtenir un texte plus homogène et linguistiquement cohérent, idéal pour la vectorisation TF‑IDF et le clustering.

La colonne anthem_lemma montre la version lemmatisée des hymnes : les mots ont été ramenés à leur forme de base (singulier, forme canonique), ce qui rend le texte plus homogène et plus facile à analyser, tout en conservant le sens général.


```python
import nltk
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("omw-1.4")

def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_text_pos(text):
    words = text.split()
    return " ".join([
        lemmatizer.lemmatize(word, get_wordnet_pos(word))
        for word in words
    ])

df["anthem_lemma"] = df["anthem_clean"].apply(lemmatize_text_pos)
```


```python
print(get_wordnet_pos("running"))
```

### Interpretation 
Nous avons appliqué une lemmatisation avancée basée sur l’étiquetage grammatical (POS tagging). Cette méthode identifie la nature de chaque mot (verbe, nom, adjectif, adverbe) avant de le ramener à sa forme de base. Le résultat est un texte plus cohérent, plus précis et mieux adapté à l’analyse sémantique et au clustering.

### c-) Tokenisation


```python
from nltk.tokenize import word_tokenize

def tokenize_text(text):
    return word_tokenize(text)

# Création de la colonne tokens
df["tokens"] = df["anthem_lemma"].apply(tokenize_text)

# Vérification
df[["anthem_lemma", "tokens"]].head()
```

### Interpretation
La tokenisation découpe chaque hymne en une liste de mots.
Au lieu d’avoir un texte continu, on obtient une suite de tokens (mots séparés), ce qui permet d’analyser le texte mot par mot.
C’est une étape essentielle pour la vectorisation TF‑IDF, le comptage de fréquences et le clustering.

###  Vectorisation.


```python
df["final_text"] = df["tokens"].apply(lambda x: " ".join(x))
df[["tokens", "final_text"]].head()
```

### Interpretation 
La colonne final_text correspond à la reconstruction du texte après tokenisation. Les tokens (mots individuels) sont réassemblés pour obtenir une version propre, normalisée et lemmatisée des hymnes, prête pour la vectorisation TF‑IDF et le clustering.


```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(
    max_features=3000,
    stop_words="english"
)

X = vectorizer.fit_transform(df["final_text"])
print(X.shape)
```

### Interpretation
La matrice TF‑IDF obtenue a une dimension de 189 × 2916 : chaque hymne est représenté par un vecteur de 2916 poids numériques correspondant aux mots les plus significatifs du corpus. Cette représentation vectorielle permet d’appliquer des méthodes de clustering et d’analyse sémantique.


```python
print(vectorizer.get_feature_names_out()[:20])

```

### Interpretation
Ce sont les premiers mots du vocabulaire construit par TF‑IDF à partir des hymnes.
Ils représentent les termes importants du corpus, triés dans l’ordre alphabétique.
Chaque mot correspond à une colonne de la matrice TF‑IDF, et TF‑IDF mesurera son importance dans chaque hymne.


```python
tfidf_df = pd.DataFrame(
    X.toarray(),
    columns=vectorizer.get_feature_names_out()
)

tfidf_df.head()
```

### Interpretation
Le DataFrame TF‑IDF contient 189 lignes (hymnes) et 2916 colonnes (mots significatifs). Chaque cellule représente le poids TF‑IDF d’un mot dans un hymne, c’est‑à‑dire son importance relative dans le corpus. Cette représentation vectorielle constitue la base du clustering et de l’analyse sémantique.

# 2) Partitionnement en K-Means

### a -) Application de cet algorithme pour différentes valeurs de k.


```python

from sklearn.cluster import KMeans

K = range(2, 15)

models = {}
labels_dict = {}

for k in K:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    
    labels = kmeans.fit_predict(X)
    
    models[k] = kmeans
    labels_dict[k] = labels
    
    print(f"k = {k} terminé")
```

### Interpretation
L’algorithme K-Means a été exécuté pour différentes valeurs de k allant de 2 à 14 afin d’explorer les différentes structures de partitionnement possibles des hymnes nationaux. Cette étape constitue une phase préparatoire nécessaire à la détermination du nombre optimal de clusters.

 ### Clustering hiérarchique


```python
from sklearn.cluster import AgglomerativeClustering

# Clustering hiérarchique k=4
hier_k4 = AgglomerativeClustering(
    n_clusters=4,
    linkage="ward"
)

df["hier_cluster_k4"] = hier_k4.fit_predict(X.toarray())

# Clustering hiérarchique k=12
hier_k12 = AgglomerativeClustering(
    n_clusters=12,
    linkage="ward"
)

df["hier_cluster_k12"] = hier_k12.fit_predict(X.toarray())
```


```python
df[[
    "Country",
    "hier_cluster_k4",
    "hier_cluster_k12"
]].head()
```

### Interpretation
Le clustering hiérarchique avec linkage de Ward permet de structurer les hymnes en groupes homogènes. Avec k=4, on obtient une segmentation globale des styles d’hymnes nationaux. En augmentant à k=12, on observe une subdivision plus fine révélant des similarités lexicales et structurelles entre certains pays au sein d’un même cluster principal.

### b-) Déterminer le nombre optimal de clusters par la méthode du coude.


```python
from sklearn.cluster import KMeans

K = range(2, 15)
inertias = []

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
```


```python
print(len(K))
print(len(inertias))
```


```python
plt.figure(figsize=(8,5))
plt.plot(K, inertias, marker='o')
plt.title("Méthode du coude")
plt.xlabel("Nombre de clusters k")
plt.ylabel("Inertie")
plt.show()
```

### Interpretation 
La méthode du coude consiste à analyser l’évolution de l’inertie en fonction du nombre de clusters k. On observe un changement  de l’inertie jusqu’à k = 4, puis une stabilisation progressive. Le point d’inflexion (coude) est donc atteint pour k = 4, ce qui indique que ce nombre de clusters représente un bon compromis entre compacité et simplicité du modèle.

### c-) Déterminer le nombre optimal de clusters par la méthode silhouette.


```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

K = range(2, 15)  #  silhouette commence à 2 clusters minimum
sil_scores = []

for k in K:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    
    score = silhouette_score(X, labels)
    sil_scores.append(score)
```


```python
plt.figure(figsize=(8,5))
plt.plot(K, sil_scores, marker='o')
plt.title("Score silhouette")
plt.xlabel("k")
plt.ylabel("Silhouette score")
plt.show()
```

### Interpretation
La méthode du score silhouette permet d’évaluer la qualité de la séparation des clusters en mesurant la cohésion intra-classe et la séparation inter-classe. Le score silhouette atteint son maximum pour k = 12, ce qui indique que cette valeur fournit la meilleure structuration des données en termes de distinction entre les groupes d’hymnes nationaux.

### d -) Ajouter au dataset original une variable “cluster“ indiquant pour chaque observation le cluster auquel elle appartient.


```python
from sklearn.metrics import silhouette_score

sil_scores = []

for k in K:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    sil_scores.append(silhouette_score(X, labels))
```


```python
final_k = 4

kmeans_final = KMeans(
    n_clusters=final_k,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans_final.fit_predict(X)
df[["Anthem", "cluster"]].head()
```

### Interpretation
Le modèle K‑means a regroupé les hymnes en 4 familles textuelles.
Les hymnes du cluster 2, par exemple, partagent des mots, des thèmes ou des structures linguistiques similaires.
Les hymnes du cluster 1 forment un autre groupe cohérent, différent des autres.
Chaque cluster représente donc un style ou une thématique dominante dans les hymnes nationaux.

Le cluster 2 contient plusieurs hymnes → ils ont probablement des thèmes communs (patriotisme, unité, liberté…).
Le cluster 1 regroupe des hymnes avec un autre style (nature, paysages, héritage culturel…).
Le cluster 3 contient des hymnes plus spécifiques (peut‑être plus courts, plus directs, plus militaires…).
Le cluster 0 (non affiché ici) représente encore un autre style.


```python
final_k = 12

kmeans_final = KMeans(
    n_clusters=final_k,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans_final.fit_predict(X)
df[["Anthem", "cluster"]].head()
```

### Interpretation
Avec k = 12, le modèle crée des clusters beaucoup plus fins.
Chaque cluster regroupe des hymnes très proches lexicalement, tandis que les clusters différents représentent des styles, des thèmes ou des structures linguistiques distinctes.

Par exemple :
Les hymnes 0 et 1 sont tous les deux dans le cluster 8 → ils partagent probablement un vocabulaire patriotique similaire (unité, liberté, nation…).
L’hymne 2 est dans le cluster 1 → il a un style différent (peut‑être plus descriptif, nature, paysages…).
L’hymne 3 est dans le cluster 4 → style plus martial ou solennel.
L’hymne 4 est dans le cluster 6 → encore un autre style.

### e-) Représenter les pays dans le plan engendré par les deux premières composantes principales (on pourra utiliser la librairie Yellowbrick) avec une couleur par cluster.


```python
%pip install yellowbrick

```

###  PCA + visualisation avec Yellowbrick


```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X.toarray())

```

### PCA pour k=4


```python
plt.figure(figsize=(10,6))

scatter = plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=df["hier_cluster_k4"],
    cmap="viridis"
)

plt.title("PCA - Clustering hiérarchique (k=4)")
plt.xlabel("CP1")
plt.ylabel("CP2")

plt.legend(*scatter.legend_elements(), title="Clusters")

plt.show()
```

### Interpretation 
La projection PCA montre que le clustering hiérarchique avec k=4 permet d’identifier des groupes relativement cohérents. On observe un cluster central regroupant la majorité des hymnes similaires, ainsi que des clusters plus distincts indiquant des différences lexicales et structurelles. La séparation partielle entre les groupes confirme l’existence de styles d’hymnes nationaux différenciés.

Le clustering hiérarchique combiné à la PCA révèle une structure naturelle des hymnes en 4 familles principales, avec un groupe dominant et quelques groupes plus spécifiques. Cela confirme l’efficacité de la méthode pour détecter des similarités textuelles.

### PCA pour k=12


```python
plt.figure(figsize=(10,6))

scatter = plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=df["hier_cluster_k12"],
    cmap="tab20"
)

plt.title("PCA - Clustering hiérarchique (k=12)")
plt.xlabel("CP1")
plt.ylabel("CP2")

plt.legend(*scatter.legend_elements(), title="Clusters")

plt.show()
```

### Interpretation
Le clustering hiérarchique avec k=12 permet d’obtenir une segmentation plus fine des hymnes, mais la projection PCA montre une forte superposition des clusters, indiquant une séparation moins nette entre les groupes. Cela suggère que les différences entre hymnes sont progressives plutôt que strictement discrètes. En comparaison, k=4 offre une meilleure interprétabilité globale.

###  f-) Représenter les pays sur une carte géographique avec une couleur par cluster.


```python
kmeans_k4 = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster_k4"] = kmeans_k4.fit_predict(X)
```


```python
import plotly.express as px
from sklearn.cluster import AgglomerativeClustering

if "hier_cluster_k4" not in df.columns:
    if "X" not in globals():
        raise ValueError("La variable X est introuvable. Exécutez d'abord la cellule TF-IDF.")
    df["hier_cluster_k4"] = AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(X.toarray())

fig = px.choropleth(
    df,
    locations="Country",
    locationmode="country names",
    color="hier_cluster_k4",
    title="Clustering hiérarchique (k=4)"
)

fig.show()
```

### Interpretation de la carte k = 4
La carte choroplèthe représente la répartition géographique des clusters obtenus avec k = 4. Chaque couleur correspond à un groupe d’hymnes partageant des similarités lexicales et thématiques. On observe que les pays appartenant à un même cluster ne sont pas nécessairement voisins géographiquement, ce qui montre que la proximité textuelle des hymnes dépend davantage de facteurs historiques, culturels ou stylistiques que de la localisation. Cette visualisation confirme la cohérence du partitionnement obtenu par K‑means.


```python
kmeans_k12 = KMeans(n_clusters=12, random_state=42, n_init=10)
df["cluster_k12"] = kmeans_k12.fit_predict(X)
```


```python
from sklearn.cluster import AgglomerativeClustering

if "hier_cluster_k12" not in df.columns:
    if "X" not in globals():
        raise ValueError("La variable X est introuvable. Exécutez d'abord la cellule TF-IDF.")
    df["hier_cluster_k12"] = AgglomerativeClustering(n_clusters=12, linkage="ward").fit_predict(X.toarray())

fig = px.choropleth(
    df,
    locations="Country",
    locationmode="country names",
    color="hier_cluster_k12",
    title="Clustering hiérarchique (k=12)"
)

fig.show()
```

### Interpretation de la carte k = 12
La carte choroplèthe montre la répartition géographique des 12 clusters obtenus par K‑means. Chaque couleur correspond à un groupe d’hymnes présentant des similarités lexicales et thématiques fines. On observe que les clusters ne suivent pas les frontières géographiques : des pays éloignés peuvent partager un style d’hymne similaire, tandis que des pays voisins peuvent appartenir à des clusters différents. Cela indique que la proximité textuelle des hymnes dépend davantage de facteurs historiques, culturels ou stylistiques que de la localisation géographique. Cette visualisation confirme la cohérence du partitionnement obtenu avec k = 12.

### g-) Représenter les mots les plus fréquents dans chaque cluster avec un nuage de mots.

### WordCloud hiérarchique


```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def plot_wordcloud(df, cluster_col, cluster_value):

    text = " ".join(
        df[df[cluster_col] == cluster_value]["final_text"]
    )

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(text)

    plt.figure(figsize=(10,5))
    plt.imshow(wc)
    plt.axis("off")
    plt.title(f"Cluster {cluster_value}")
    plt.show()
```

###  pour k=4


```python
for c in sorted(df["hier_cluster_k4"].unique()):
    plot_wordcloud(df, "hier_cluster_k4", c)
```

### Interpretation
Les nuages de mots révèlent des thématiques distinctes entre les clusters.
Le cluster 0 met en avant la patrie et la bénédiction divine, le cluster 1 insiste sur le peuple et l’unité nationale, le cluster 2 valorise la liberté et la défense du pays, tandis que le cluster 3 adopte un ton plus religieux et poétique.
Ces différences lexicales confirment la pertinence du clustering : chaque groupe d’hymnes possède un style et des valeurs dominantes propres.

### k=12


```python
for c in sorted(df["hier_cluster_k12"].unique()):
    plot_wordcloud(df, "hier_cluster_k12", c)
```

### Interpretation
Les wordclouds montrent que les 12 clusters identifiés par K‑means correspondent à des styles d’hymnes très distincts : religieux, guerriers, poétiques, historiques, africains, européens, ou centrés sur la liberté et l’unité.
Chaque cluster possède un vocabulaire dominant qui reflète des valeurs, des contextes historiques et des identités culturelles différentes.
Cette analyse confirme la pertinence du clustering et met en évidence la diversité thématique des hymnes nationaux.

# 3-) Partitionnement hiérarchique


```python
from sklearn.cluster import AgglomerativeClustering

# modèle hiérarchique
hierarchical_model = AgglomerativeClustering(
    n_clusters=4,   # tu peux aussi tester 12
    linkage="ward"
)

df["cluster_hier_4"] = hierarchical_model.fit_predict(X.toarray())
```


```python
hierarchical_model_12 = AgglomerativeClustering(
    n_clusters=12,
    linkage="ward"
)

df["cluster_hier_12"] = hierarchical_model_12.fit_predict(X.toarray())
```


```python
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
```


```python
linked = linkage(X.toarray(), method="ward")
```


```python
plt.figure(figsize=(12, 6))

dendrogram(
    linked,
    truncate_mode="level",  # simplifie l'affichage
    p=5                     # profondeur affichée
)

plt.title("Dendrogramme - Clustering hiérarchique des hymnes")
plt.xlabel("Hymnes")
plt.ylabel("Distance")
plt.show()
```

### Interpretation 
 Le dendrogramme obtenu avec la méthode de Ward montre une structure hiérarchique claire dans les hymnes nationaux. Les fusions basses indiquent des hymnes très similaires, tandis que les fusions hautes révèlent des groupes textuellement distincts. Une coupure horizontale au niveau intermédiaire suggère 4 clusters principaux, tandis qu’une coupure plus fine permet d’obtenir 12 clusters plus détaillés. Cette représentation confirme la cohérence des regroupements observés avec K‑means et met en évidence la diversité thématique des hymnes (patriotisme, liberté, religion, histoire, unité, etc.).

## Comparaison K-Means / Hiérarchique

| Critère             | K-Means            | Hiérarchique        |
| ------------------- | ------------------ | ------------------- |
| Principe            | Centres de gravité | Fusions successives |
| Paramètre principal | k                  | k ou dendrogramme   |
| Rapidité            | Élevée             | Plus lente          |
| Interprétation      | Moyenne            | Excellente          |
| Visualisation       | PCA                | PCA + dendrogramme  |


### Conclusion comparative
K-Means produit rapidement une partition des hymnes nationaux tandis que le clustering hiérarchique fournit une représentation plus riche de la structure des données grâce au dendrogramme. Les deux approches conduisent à des regroupements cohérents mais mettent en évidence des aspects différents des similarités entre hymnes.

### Conclusion finale

Dans ce projet, nous avons appliqué plusieurs techniques de partitionnement sur les hymnes nationaux. Après un prétraitement comprenant nettoyage, lemmatisation, tokenisation et vectorisation TF-IDF, nous avons utilisé l'algorithme K-Means et le clustering hiérarchique.

La méthode du coude a suggéré un nombre optimal de 4 clusters tandis que le score silhouette a indiqué 12 clusters. Les résultats ont été étudiés à travers des projections PCA, des cartes géographiques et des nuages de mots.

Les regroupements obtenus montrent que les hymnes nationaux partagent des thèmes récurrents tels que la nation, la liberté, l'unité, la patrie et le peuple. Enfin, le clustering hiérarchique a permis de visualiser les relations entre les hymnes grâce au dendrogramme et d'obtenir une vision complémentaire de la structure des données.
