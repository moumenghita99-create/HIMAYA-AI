
import streamlit as st
import joblib

st.set_page_config(
    page_title="HIMAYA AI",
    page_icon="🛡️",
    layout="centered"
)

@st.cache_resource
def charger_modeles():
    return {
        "Anglais": (
            joblib.load("himaya_modeles/modele_anglais.pkl"),
            joblib.load("himaya_modeles/tfidf_anglais.pkl")
        ),
        "Français": (
            joblib.load("himaya_modeles/modele_francais.pkl"),
            joblib.load("himaya_modeles/tfidf_francais.pkl")
        ),
        "Arabe": (
            joblib.load("himaya_modeles/modele_arabe.pkl"),
            joblib.load("himaya_modeles/tfidf_arabe.pkl")
        )
    }

modeles = charger_modeles()

st.title("🛡️ HIMAYA AI")
st.subheader("Détection intelligente des messages suspects")
st.write(
    "Analysez un message en anglais, en français ou en arabe "
    "pour détecter les signes potentiels de spam."
)

langue = st.selectbox(
    "Choisissez la langue du message :",
    ["Anglais", "Français", "Arabe"]
)

message = st.text_area(
    "Entrez votre message :",
    placeholder="Écrivez ou collez votre message ici..."
)

if st.button("Analyser le message", type="primary"):
    if not message.strip():
        st.warning("Veuillez saisir un message.")
    else:
        modele, vectoriseur = modeles[langue]
        message_tfidf = vectoriseur.transform([message])
        prediction = modele.predict(message_tfidf)[0]

        if langue in ["Anglais", "Français"]:
            est_spam = prediction == "spam"
        else:
            est_spam = prediction == 1

        if est_spam:
            st.error("⚠️ Message suspect (Spam)")
        else:
            st.success("✅ Message normal (Ham)")

        st.caption(
            "Résultat automatique : il peut être incorrect. "
            "Vérifiez toujours les messages suspects."
        )
