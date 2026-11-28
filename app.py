import os
from pathlib import Path

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =====================================
# CONFIGURATION CACIP
# =====================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "CACIP_V3.keras (4)"
LOGO_PATH = BASE_DIR / "assets" / "logo_cacip.png"

st.set_page_config(
    page_title="CACIP",
    page_icon="🔬",
    layout="centered"
)

# =====================================
# CHARGEMENT DU MODELE
# =====================================

@st.cache_resource
def charger_modele():
    return tf.keras.models.load_model(str(MODEL_PATH))


# =====================================
# IDENTITE VISUELLE
# =====================================

st.title("🔬 CACIP")
st.subheader("Cancer Artificial Intelligence Project")

st.caption(
    "Analyse expérimentale d'images histologiques mammaires"
)

if LOGO_PATH.is_file():
    st.image(str(LOGO_PATH), width=220)

st.divider()

# =====================================
# ANALYSE D'IMAGE
# =====================================

st.header("Analyse d'une image")

fichier = st.file_uploader(
    "Importer une image histologique",
    type=["jpg", "jpeg", "png"]
)

if fichier is not None:

    try:
        image = Image.open(fichier).convert("RGB")

        st.image(
            image,
            caption="Image importée",
            use_container_width=True
        )

        if st.button("Analyser l'image", type="primary"):

            if not MODEL_PATH.is_file():

                st.error(
                    f"Modèle V3 introuvable. "
                    f"Chemin recherché : {MODEL_PATH}"
                )

                dossier = MODEL_PATH.parent

                if dossier.exists():
                    st.write(
                        "Fichiers présents dans models :",
                        [f.name for f in dossier.iterdir()]
                    )
                else:
                    st.error("Le dossier models est absent.")

            else:

                try:
                    with st.spinner("Chargement du modèle CACIP V3..."):
                        modele = charger_modele()

                    with st.spinner("Analyse expérimentale en cours..."):

                        image_preparee = image.resize((224, 224))

                        tableau = np.array(
                            image_preparee,
                            dtype=np.float32
                        ) / 255.0

                        tableau = np.expand_dims(tableau, axis=0)

                        prediction = modele.predict(
                            tableau,
                            verbose=0
                        )

                        prediction = np.asarray(prediction)

                        if prediction.shape[-1] == 2:
                            score_malin = float(prediction[0][1])

                        elif prediction.size == 1:
                            score_malin = float(prediction.ravel()[0])

                        else:
                            st.error(
                                "Format de sortie du modèle non reconnu."
                            )
                            st.stop()

                        if not 0 <= score_malin <= 1:
                            st.error(
                                "La sortie du modèle n'est pas "
                                "une probabilité comprise entre 0 et 1."
                            )
                            st.stop()

                        if score_malin >= 0.5:
                            resultat = "Maligne"
                            confiance = score_malin
                        else:
                            resultat = "Bénigne"
                            confiance = 1 - score_malin

                    st.subheader("Résultat expérimental")

                    st.write(
                        "Classification proposée :",
                        resultat
                    )

                    st.metric(
                        "Score associé à la classe proposée",
                        f"{confiance * 100:.2f} %"
                    )

                    st.progress(
                        float(np.clip(confiance, 0, 1))
                    )

                    st.session_state.setdefault(
                        "historique", []
                    )

                    st.session_state.historique.append({
                        "résultat": resultat,
                        "score": round(confiance * 100, 2)
                    })

                    st.warning(
                        "CACIP est un prototype de recherche. "
                        "Ce résultat ne constitue pas un diagnostic "
                        "médical et ne doit pas guider une décision "
                        "de traitement."
                    )

                except Exception as erreur:
                    st.error(
                        "Erreur pendant le chargement ou l'analyse "
                        "du modèle CACIP V3."
                    )
                    st.exception(erreur)

    except Exception as erreur:
        st.error("Impossible de lire cette image.")
        st.exception(erreur)


# =====================================
# HISTORIQUE
# =====================================

st.divider()
st.header("Historique des analyses")

for numero, analyse in enumerate(
    st.session_state.get("historique", []),
    start=1
):
    st.write(
        f"{numero}. {analyse['résultat']} — "
        f"{analyse['score']} %"
    )

st.caption(
    "CACIP — Prototype de recherche utilisant le modèle V3"
)
