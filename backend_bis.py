import streamlit as st
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from rank_bm25 import BM25Okapi
import pickle, os

FILE_PATH = "data/Structures_accompagnement_Movendo_v1.xlsm"
MODEL_PATH = "word2vec_model.pkl"

# --- Chargement optimisé ---
@st.cache_resource
def load_data_and_models():
    sheets = pd.ExcelFile(FILE_PATH, engine='openpyxl').sheet_names
    frames = []
    for sheet in sheets:
        df = pd.read_excel(FILE_PATH, sheet_name=sheet, engine="openpyxl")
        df["Catégorie"] = sheet
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype("string")
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    # Tokenisation limitée
    tokenized_corpus = [
        simple_preprocess(' '.join(row.dropna().astype(str)))
        for _, row in data.iterrows()
    ]
    bm25 = BM25Okapi(tokenized_corpus)

    # Texts pour Word2Vec
    texts = tokenized_corpus

    # Charger ou entraîner Word2Vec
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    else:
        model = Word2Vec(sentences=texts, vector_size=50, window=4, min_count=2, workers=2)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)

    return data, sheets, bm25, model

# --- Similarité W2V ---
def document_vector(model, doc):
    words = simple_preprocess(doc)
    vecs = [model.wv[w] for w in words if w in model.wv]
    return np.mean(vecs, axis=0) if vecs else None

# --- Recherche combinée ---
def search_combined(query, data, bm25, model, top_k=10):
    tokenized_query = simple_preprocess(query)
    bm25_scores = bm25.get_scores(tokenized_query)

    query_vec = document_vector(model, query)
    w2v_scores = []
    for _, row in data.iterrows():
        row_text = " ".join(row.dropna().astype(str))
        text_vec = document_vector(model, row_text)
        if text_vec is not None and query_vec is not None:
            sim = np.dot(query_vec, text_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(text_vec))
            w2v_scores.append(sim)
        else:
            w2v_scores.append(0.0)

    bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.ptp(bm25_scores) + 1e-8)
    w2v_norm = (np.array(w2v_scores) - np.min(w2v_scores)) / (np.ptp(w2v_scores) + 1e-8)

    combined_scores = 0.8 * bm25_norm + 0.2 * w2v_norm

    top_idx = np.argsort(combined_scores)[::-1][:top_k]
    results = []
    for idx in top_idx:
        score = combined_scores[idx]
        row = data.iloc[idx].dropna().to_dict()
        row["similarity"] = round(float(score), 3)
        results.append(row)
    return results

# ==========================
# --- INTERFACE STREAMLIT ---
# ==========================

st.set_page_config(page_title="Recherche Movendo", layout="wide")

st.title("🔍 Recherche dans Movendo")
st.write("Tapez un mot-clé puis appuyez sur **Entrée** pour lancer la recherche")

# Chargement lazy avec spinner
with st.spinner("Chargement des données..."):
    data, sheets, bm25, model = load_data_and_models()

# --- Choix de catégorie ---
st.sidebar.header("🔖 Filtres")
categorie_choisie = st.sidebar.selectbox(
    "Filtrer par catégorie :", 
    ["Toutes"] + sheets
)

# Champ de recherche : déclenche auto sur Entrée
query = st.text_input("Rechercher :", placeholder="Ex: accompagnement emploi")

# --- Filtrer les données selon la catégorie ---
if categorie_choisie != "Toutes":
    data_filtrée = data[data["Catégorie"] == categorie_choisie]
else:
    data_filtrée = data

# --- Lancer la recherche ---
if query:  
    with st.spinner("🔎 Recherche en cours..."):
        # BM25 doit être recalculé si on filtre par catégorie
        tokenized_corpus = [
            simple_preprocess(' '.join(row.dropna().astype(str)))
            for _, row in data_filtrée.iterrows()
        ]
        bm25_filtered = BM25Okapi(tokenized_corpus)
        
        results = search_combined(query, data_filtrée, bm25_filtered, model)

    st.subheader(f"✅ {len(results)} résultats trouvés "
                 f"{'dans '+categorie_choisie if categorie_choisie!='Toutes' else ''}")

    # Affichage sous forme de "cartes"
    for r in results:
        with st.container(border=True):
            titre = (
    r.get('Dispositif') 
    or r.get('Sites') 
    or r.get('Lieu') 
    or "Sans titre"
)
            st.markdown(f"**🏷 {titre}**")
            if "Description" in r:
                st.write(r["Description"])
            col1, col2 = st.columns(2)
            with col1:
                if "Contact" in r: st.write("📞", r["Contact"])
            with col2:
                if "Adresse" in r: st.write("📍", r["Adresse"])
            st.caption(f"Catégorie : {r.get('Catégorie', '')} | Score : {r['similarity']}")

else:
    st.info("💡 Commencez par saisir une recherche ci-dessus ou choisissez une catégorie dans la barre latérale.")

