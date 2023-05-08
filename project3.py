import argparse
import os
import re

import contractions
import numpy as np
import pandas as pd
import spacy
import nltk

from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords
from PyPDF2 import PdfReader
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')


def preprocess_text(input_text):
    input_text = contractions.fix(input_text)
    input_text = re.sub(r'[^a-zA-Z0-9\s]', '', input_text)
    tkns = word_tokenize(input_text)
    stp_wrds = set(stopwords.words('english'))
    tkns = [token for token in tkns if token.lower() not in stp_wrds]
    lmtzr = WordNetLemmatizer()
    tkns = [lmtzr.lemmatize(token) for token in tkns]
    prcsd_text = ' '.join(tkns)

    return prcsd_text

nlp = spacy.load('en_core_web_sm')

max_fld_cities = 15
fld_cities_count = 0
parser = argparse.ArgumentParser()
parser.add_argument('--document', help='Path to the PDF document for clustering')
args = parser.parse_args()

if not args.document:
    raise ValueError('Please provide the path to the PDF document for clustering')

city_n = os.path.basename(args.document).split(".")[0]
cities_n = {city_n}
city_dict = {}
try:
    with open(args.document, 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
        preprcsd_data = ''
        for pg_n in range(num_pages):
            pg_o = pdf_reader.pages[pg_n]
            text = pg_o.extract_text()
            preprcsd_data += text
        prcsd_text = preprocess_text(preprcsd_data)
        city_dict[city_n] = prcsd_text
        print(prcsd_text)

except:
    raise ValueError(f'Failed to process document {args.document}')

vectorizer = TfidfVectorizer()
text_data = vectorizer.fit_transform(list(city_dict.values())).toarray()

for data in text_data:
    k_values = [9, 18, 36]
    k_scores = np.zeros((len(k_values), 3))
    h_scores = np.zeros((len(k_values), 3))
    d_scores = np.zeros((len(k_values), 3))

    for x, y in enumerate(k_values):
        kmeans_i = KMeans(n_clusters=y, random_state=0)
        data = data.reshape(-1, 1)
        kmeans_i.fit(data)
        kmeans_labels = kmeans_i.labels_
        print(y)
        kmeans_i = KMeans(n_clusters=y)
        kmeans_labels = kmeans_i.fit_predict(data)
        k_scores[x, 0] = silhouette_score(data, kmeans_labels)
        k_scores[x, 1] = calinski_harabasz_score(data, kmeans_labels)
        k_scores[x, 2] = davies_bouldin_score(data, kmeans_labels)

        hierarchical = AgglomerativeClustering(n_clusters=y)
        hierarchical_labels = hierarchical.fit_predict(data)
        h_scores[x, 0] = silhouette_score(data, hierarchical_labels)
        h_scores[x, 1] = calinski_harabasz_score(data, hierarchical_labels)
        h_scores[x, 2] = davies_bouldin_score(data, hierarchical_labels)

        dbscan = DBSCAN(eps=0.5, min_samples=y)
        dbscan_labels = dbscan.fit_predict(data)
        if len(np.unique(dbscan_labels)) > 1:
            d_scores[x, 0] = silhouette_score(data, dbscan_labels)
            d_scores[x, 1] = calinski_harabasz_score(data, dbscan_labels)
            d_scores[x, 2] = davies_bouldin_score(data, dbscan_labels)


    kmeans_optimal_k = k_values[np.argmax(k_scores[:, 0])]
    hierarchical_optimal_k = k_values[np.argmax(h_scores[:, 0])]


    print("K-means Scores:")
    print(k_scores)
    print("Optimal k for K-means:", kmeans_optimal_k)
    print()
    print("Hierarchical Scores:")
    print(h_scores)
    print("Optimal k for Hierarchical:", hierarchical_optimal_k)
    print()
    print("DBSCAN Scores:")
    print(d_scores)