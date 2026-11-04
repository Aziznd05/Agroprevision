import streamlit as st
import requests
import pandas as pd
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="AgroPrévision", layout="wide", page_icon="🌱")

# Variables d'environnement pour Docker
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")

# ==========================================
# 2. BARRE LATÉRALE
# ==========================================
with st.sidebar:
    st.title("AgroPrévision")
    st.caption("Intelligence agricole")
    st.divider()
    page = st.radio("Navigation", ["Vue d'ensemble", "Tableau de bord & Prévisions", "Assistant Agronomique"])
    st.divider()
    st.caption("Campagne 2025")
    st.progress(0.74)

# ==========================================
# PAGE 1 : VUE D'ENSEMBLE
# ==========================================
if page == "Vue d'ensemble":
    st.title("Vue d'ensemble")
    st.markdown("Suivez les prévisions de rendement et l'état des données de la campagne agricole.")
    st.write("")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Parcelles suivies", "500", "4 régions")
    col2.metric("Rendement moyen prévu", "1,18 t/ha", "+8,4 % vs 2024")
    col3.metric("Prévisions disponibles", "1 284", "Campagne courante")
    col4.metric("Parcelles à surveiller", "36", "7,2 % du total", delta_color="inverse")
    
    st.write("")
    st.write("")
    
    # --- LA CARTE NATIVE (ULTRA STABLE) ---
    st.subheader("Carte des parcelles")
    st.markdown("Répartition géographique des parcelles dans le bassin arachidier.")
    
    data_carte = pd.DataFrame({
        'lat': [14.10, 14.26, 14.17, 14.79],
        'lon': [-15.54, -16.28, -16.07, -15.91]
    })
    
    st.map(data_carte, zoom=7)

# ==========================================
# PAGE 2 : PREVISIONS
# ==========================================
elif page == "Tableau de bord & Prévisions":
    st.title("Prévisions de rendement")
    st.markdown("Estimez les rendements à date partielle, avec intervalle d'incertitude et facteurs explicatifs.")
    
    col_input, col_resultat = st.columns([1, 2])
    
    with col_input:
        st.markdown("### Paramètres")
        parcelle_id = st.text_input("Identifiant de parcelle", "PAR0002")
        annee = st.number_input("Campagne", min_value=2022, max_value=2025, value=2025)
        st.write("")
        bouton_predire = st.button("Calculer la prévision", use_container_width=True)

    with col_resultat:
        if bouton_predire:
            with st.spinner("Interrogation de l'API..."):
                try:
                    url = f"{API_URL}/predict"
                    reponse = requests.post(url, json={"parcelle_id": parcelle_id, "annee": annee}, timeout=10)
                    
                    if reponse.status_code == 200:
                        res = reponse.json()
                        st.success(f"Prévision pour {res['parcelle_id']} ({res['annee']})")
                        
                        c1, c2 = st.columns(2)
                        c1.metric("Rendement médian", f"{res['rendement_median_t_ha']} t/ha")
                        c2.metric("Intervalle de confiance (80%)", f"{res['intervalle_80'][0]} à {res['intervalle_80'][1]} t/ha")
                        
                        st.write("")
                        st.subheader("Facteurs d'influence (SHAP)")
                        df_shap = pd.DataFrame(res['facteurs_shap'])
                        st.bar_chart(df_shap.set_index("facteur")["impact_t_ha"])
                        
                        for f in res['facteurs_shap']:
                            signe = "+" if f['impact_t_ha'] > 0 else ""
                            emoji = "🔺" if f['impact_t_ha'] > 0 else "🔻"
                            st.write(f"{emoji} **{f['facteur']}** : impact de {signe}{f['impact_t_ha']} t/ha")
                    else:
                        st.error("Parcelle introuvable.")
                except Exception:
                    st.error("Erreur de connexion. Assurez-vous que l'API est lancée.")
        else:
            st.info("Renseignez un identifiant et cliquez sur Calculer pour voir le résultat.")

# ==========================================
# PAGE 3 : ASSISTANT
# ==========================================
elif page == "Assistant Agronomique":
    st.title("Assistant Agronomique IA")
    st.markdown("Posez des questions sur vos parcelles ou sur les bonnes pratiques agricoles.")
    
    @st.cache_resource(show_spinner=False)
    def charger_agent():
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
        from langchain_core.tools import tool
        
        @tool
        def prevoir_rendement(parcelle_id: str, annee: str) -> str:
            """Outil pour obtenir la prévision de rendement."""
            try:
                rep = requests.post(f"{API_URL}/predict", json={"parcelle_id": parcelle_id, "annee": int(annee)}, timeout=10)
                if rep.status_code == 200:
                    d = rep.json()
                    return f"Médiane: {d['rendement_median_t_ha']} t/ha. Facteurs: {d['facteurs_shap']}"
                return "Erreur: Parcelle introuvable."
            except Exception:
                return "Erreur: Impossible de contacter l'API."

        db_path = os.path.abspath("data/chroma_db")
        embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        @tool
        def rechercher_documents(question: str) -> str:
            """Outil pour chercher des conseils agronomiques."""
            docs = retriever.invoke(question)
            return "\n\n".join([f"Source [{doc.metadata.get('source', 'Inconnu')}] : {doc.page_content}" for doc in docs])

        llm = ChatOllama(base_url=OLLAMA_URL, model="llama3.1:8b", temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Tu es un assistant agronomique. Utilise tes outils et réponds en français."),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        agent = create_tool_calling_agent(llm, [prevoir_rendement, rechercher_documents], prompt)
        return AgentExecutor(agent=agent, tools=[prevoir_rendement, rechercher_documents], verbose=False)

    with st.spinner("Chargement de l'IA..."):
        agent_executor = charger_agent()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ex: Quel est le rendement de PAR0002 ?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("L'assistant réfléchit..."):
                try:
                    reponse = agent_executor.invoke({"input": prompt})
                    st.markdown(reponse["output"])
                    st.session_state.messages.append({"role": "assistant", "content": reponse["output"]})
                except Exception as e:
                    st.error(f"Erreur : {e}")