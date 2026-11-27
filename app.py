import os

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "models/CACIP_V3.keras"
LOGO_PATH = "/content/drive/MyDrive/CACIP/assets/logo_cacip.png"

st.set_page_config(
    page_title="CACIP",
    page_icon="🔬",
    layout="centered"
)

@st.cache_resource
def charger_modele():
    return tf.keras.models.load_model(MODEL_PATH)

st.title("🔬 CACIP")
st.subheader("Cancer Artificial Intelligence Project")
st.caption("Analyse expérimentale d'images histologiques mammaires")

if os.path.exists(LOGO_PATH):
    st.image(LOGO_PATH, width=220)

st.divider()
st.header("Analyse d'une image")

fichier = st.file_uploader(
    "Importer une image histologique",
    type=["jpg", "jpeg", "png"]
)

if fichier is not None:
    image = Image.open(fichier).convert("RGB")
    st.image(image, caption="Image importée", use_container_width=True)

    if st.button("Analyser l'image", type="primary"):
        if not os.path.exists(MODEL_PATH):
            st.error("Modèle V3 introuvable. Vérifie son emplacement dans Google Drive.")
        else:
            with st.spinner("Analyse en cours..."):
                modele = charger_modele()
                image_preparee = image.resize((224, 224))
                tableau = np.array(image_preparee, dtype=np.float32) / 255.0
                tableau = np.expand_dims(tableau, axis=0)

                prediction = modele.predict(tableau, verbose=0)

                if prediction.shape[-1] == 2:
                    score_malin = float(prediction[0][1])
                else:
                    score_malin = float(np.ravel(prediction)[0])

                if score_malin >= 0.5:
                    resultat = "Maligne"
                    confiance = score_malin
                else:
                    resultat = "Bénigne"
                    confiance = 1 - score_malin

                st.session_state.setdefault("historique", [])
                st.session_state.historique.append({
                    "résultat": resultat,
                    "score": round(confiance * 100, 2)
                })

            st.subheader("Résultat expérimental")
            st.write("Classification proposée :", resultat)
            st.metric("Score associé à la classe proposée", f"{confiance * 100:.2f} %")
            st.progress(float(np.clip(confiance, 0, 1)))

            st.warning(
                "CACIP est un prototype de recherche. "
                "Ce résultat ne constitue pas un diagnostic médical."
            )

st.divider()
st.header("Historique des analyses")

for numero, analyse in enumerate(
    st.session_state.get("historique", []), start=1
):
    st.write(
        f"{numero}. {analyse['résultat']} — "
        f"{analyse['score']} %"
    )

st.caption("CACIP V1 — Modèle expérimental V3")
