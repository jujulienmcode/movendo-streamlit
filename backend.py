from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import os
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import numpy as np
import pickle

load_dotenv()  # Charge les variables du fichier .env

app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv("SECRET_KEY")
app.config['DEBUG'] = os.getenv("DEBUG", "False") == "True"
file_path = os.path.join(os.path.dirname(__file__), 'data', 'Structures_accompagnement_Movendo_v1.xlsm')

model_path = 'word2vec_model.pkl'
@app.route('/')
def serve_home():
    return send_from_directory('static', 'Frontend.html')

def preprocess(text):
    if isinstance(text, str):
        return simple_preprocess(text, deacc=True)
    return []

def load_data():
    sheets = pd.ExcelFile(file_path, engine='openpyxl').sheet_names
    frames = []
    for sheet in sheets:
        df = pd.read_excel(file_path, sheet_name=sheet, engine='openpyxl')
        df['Catégorie'] = sheet
        frames.append(df)
    return pd.concat(frames, ignore_index=True), sheets

# Load or train model
def train_model(texts):
    return Word2Vec(sentences=texts, vector_size=100, window=5, min_count=1, workers=4)

def document_vector(model, doc):
    words = preprocess(doc)
    vecs = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vecs, axis=0) if vecs else None

# Charger données
data, categories = load_data()

# Initialiser BM25 après avoir chargé data
tokenized_corpus = [simple_preprocess(' '.join(row.dropna().astype(str))) for _, row in data.iterrows()]
bm25 = BM25Okapi(tokenized_corpus)

# Prétraitement pour entraînement
texts = data.apply(lambda row: preprocess(' '.join(row.dropna().astype(str))), axis=1).tolist()

# Charger ou entraîner le modèle
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    model = train_model(texts)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

def train_model(texts):
    return Word2Vec(sentences=texts, vector_size=100, window=5, min_count=1, workers=4)

def document_vector(model, doc):
    words = preprocess(doc)
    vecs = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vecs, axis=0) if vecs else None

@app.route('/categories', methods=['GET'])
def get_categories():
    return jsonify(categories)

@app.route('/category/<cat>', methods=['GET'])
def get_category(cat):
    fiches = data[data['Catégorie'] == cat].dropna(axis=1, how='all')
    # Remplacer tous les NaN par None pour un JSON valide
    fiches = fiches.where(pd.notnull(fiches), None)
    return jsonify(fiches.to_dict(orient='records'))

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    query_vec = document_vector(model, query)

    if query_vec is None:
        return jsonify({"error": "Aucun vecteur trouvé pour la requête."}), 400

    similarities = []
    for idx, row in data.iterrows():
        row_text = ' '.join(row.dropna().astype(str))
        text_vec = document_vector(model, row_text)
        if text_vec is not None:
            sim = np.dot(query_vec, text_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(text_vec))
            similarities.append((idx, float(sim)))
        else:
            similarities.append((idx, 0.0))

    # Trier et prendre les 10 meilleurs résultats
    top_results = sorted(similarities, key=lambda x: x[1], reverse=True)[:10]
    results = []
    for idx, sim in top_results:
        if sim > 0.2:  # seuil raisonnable
            row = data.iloc[idx].dropna().to_dict()
            row['similarity'] = round(sim, 3)
            results.append(row)

    return jsonify(results)

# 2. Route de recherche BM25
@app.route('/search_bm25', methods=['GET'])
def search_bm25():
    query = request.args.get('query', '')
    tokenized_query = simple_preprocess(query)

    scores = bm25.get_scores(tokenized_query)
    top_n = np.argsort(scores)[::-1][:10]

    results = []
    for idx in top_n:
        score = scores[idx]
        row = data.iloc[idx]
        if score > 0 and pd.notna(row.get("Dispositif")) and (pd.notna(row.get("Contact")) or pd.notna(row.get("Adresse"))):
            row_dict = row.dropna().to_dict()
            row_dict['score'] = round(float(score), 2)
            results.append(row_dict)

    return jsonify(results)

# 3. Route de recherche combinée BM25 + Word2Vec
@app.route('/search_combined', methods=['GET'])
def search_combined():
    query = request.args.get('query', '')
    tokenized_query = simple_preprocess(query)

    # BM25 scores
    bm25_scores = bm25.get_scores(tokenized_query)

    # Word2Vec scores
    query_vec = document_vector(model, query)
    w2v_scores = []
    for idx, row in data.iterrows():
        row_text = ' '.join(row.dropna().astype(str))
        text_vec = document_vector(model, row_text)
        if text_vec is not None and query_vec is not None:
            sim = np.dot(query_vec, text_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(text_vec))
            w2v_scores.append(sim)
        else:
            w2v_scores.append(0.0)

    # Normalisation des scores
    bm25_scores_norm = (bm25_scores - np.min(bm25_scores)) / (np.ptp(bm25_scores) + 1e-8)
    w2v_scores_norm = (np.array(w2v_scores) - np.min(w2v_scores)) / (np.ptp(w2v_scores) + 1e-8)

    # Score combiné (moyenne pondérée, ajustez les poids si besoin)
    combined_scores = 0.8 * bm25_scores_norm + 0.2 * w2v_scores_norm

    # Top 10 résultats
    top_n = np.argsort(combined_scores)[::-1][:10]
    results = []
    for idx in top_n:
        score = combined_scores[idx]
        row = data.iloc[idx]
        row_dict = row.dropna().to_dict()
        row_dict['similarity'] = round(float(score), 3)
        results.append(row_dict)

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'])
