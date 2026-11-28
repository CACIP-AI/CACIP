import os
import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
from PIL import Image

st.set_page_config(
    page_title="CACIP",
    page_icon="🔬",
    layout="centered"
)

# CHEMINS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "logo_cacip.png")
MODEL_PATH = "/tmp/CACIP_V3.keras"
DRIVE_FILE_ID = "1j16Edxolos8a9nZmnoagiby34VRnUR5j"

# CHARGEMENT DU MODELE
@st.cache_resource
def charger_modele():
    if not os.path.exists(MODEL_PATH):
        resultat = gdown.download(
            id=DRIVE_FILE_ID,
            output=MODEL_PATH,
            quiet=False
        )
        if resultat is None:
            raise RuntimeError("Téléchargement du modèle impossible.")

    return tf.keras.models.load_model(MODEL_PATH)


# INTERFACE CACIP
st.title("🔬 CACIP")
st.subheader("Cancer Artificial Intelligence Project")
st.caption("Analyse expérimentale d'images histologiques mammaires")

# LOGO
if os.path.isfile(LOGO_PATH):
    try:
        logo = Image.open(LOGO_PATH).convert("RGB")
        st.image("https://raw.githubusercontent.com/CACIP-AI/CACIP/main/logo_cacip.png", width=280)
    except Exception as erreur:
        st.error("Impossible de lire le logo.")
        st.exception(erreur)
else:
    st.warning("Logo CACIP introuvable.")

st.divider()

# IMPORTATION IMAGE
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
                        st.error("Format de sortie du modèle non reconnu.")
                        st.stop()

                    if not 0 <= score_malin <= 1:
                        st.error("Sortie du modèle invalide.")
                        st.stop()

                    if score_malin >= 0.5:
                        resultat = "Maligne"
                        confiance = score_malin
                    else:
                        resultat = "Bénigne"
                        confiance = 1 - score_malin

                st.subheader("Résultat expérimental")
                st.write("Classification proposée :", resultat)

                st.metric(
                    "Score associé à la classe proposée",
                    f"{confiance * 100:.2f} %"
                )

                st.progress(
                    float(np.clip(confiance, 0, 1))
                )

                st.warning(
                    "CACIP est un prototype de recherche. "
                    "Ce résultat ne constitue pas un diagnostic médical "
                    "et ne doit pas guider une décision de traitement."
                )

            except Exception as erreur:
                st.error("Erreur pendant le chargement ou l'analyse du modèle.")
                st.exception(erreur)

    except Exception as erreur:
        st.error("Impossible de lire cette image.")
        st.exception(erreur)

st.divider()

st.caption(
    "CACIP — Prototype de recherche utilisant le modèle V3"
)
