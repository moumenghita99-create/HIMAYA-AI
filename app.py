
import streamlit as st
import joblib

def expliquer_message(message, langue):
    texte = message.lower()

    regles = {
        "Anglais": [
            (["win", "winner", "prize", "free gift"],
             "Présence d'une promesse de gain ou de cadeau."),
            (["click here", "click now", "claim now"],
             "Incitation à cliquer sur un lien."),
            (["urgent", "immediately", "act now"],
             "Utilisation d'un langage urgent.")
        ],

        "Français": [
            (["gagné", "gagner", "cadeau", "prix gratuit"],
             "Présence d'une promesse de gain ou de cadeau."),
            (["cliquez ici", "clique maintenant"],
             "Incitation à cliquer sur un lien."),
            (["urgent", "immédiatement", "sans attendre"],
             "Utilisation d'un langage urgent.")
        ],

        "Arabe": [
            (["مبروك", "ربحت", "جائزة", "هدية مجانية"],
             "وجود وعد بالفوز أو الحصول على جائزة."),
            (["اضغط هنا", "انقر هنا"],
             "تشجيع المستخدم على الضغط على رابط."),
            (["عاجل", "فورا", "بسرعة"],
             "استعمال عبارات تدل على الاستعجال.")
        ]
    }

    indices = []

    for mots, explication in regles[langue]:
        if any(mot in texte for mot in mots):
            indices.append(explication)

    return indices

st.set_page_config(
    page_title="HIMAYA AI",
    page_icon="🛡️",
    layout="centered"
)

@st.cache_resource
def charger_modeles():
    return {
        "Anglais": (
            joblib.load("modele_anglais.pkl"),
            joblib.load("tfidf_anglais.pkl")
        ),
        "Français": (
            joblib.load("modele_francais.pkl"),
            joblib.load("tfidf_francais.pkl")
        ),
        "Arabe": (
            joblib.load("modele_arabe.pkl"),
            joblib.load("tfidf_arabe.pkl")
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

        indices = expliquer_message(message, langue)

        if est_spam:
            st.markdown("### Pourquoi ce message est-il suspect ?")

            if indices:
                for indice in indices:
                    st.write("🔎", indice)
            else:
                st.write(
                    "Le modèle a classé ce message comme Spam, "
                    "mais aucun indice précis n'a été détecté par nos règles."
                )

        else:
            st.markdown("### Pourquoi ce message est-il classé comme Ham ?")

            if indices:
                st.write(
                    "Le modèle a classé ce message comme Ham, "
                    "mais certains mots méritent une vérification :"
                )
                for indice in indices:
                    st.write("🔎", indice)
            else:
                st.write(
                    "Aucun des indices suspects recherchés "
                    "n'a été détecté dans ce message."
                )

        st.caption(
            "Résultat automatique : il peut être incorrect. "
            "Vérifiez toujours les messages suspects."
        )
