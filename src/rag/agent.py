import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.messages import HumanMessage

# 1. Définir l'outil pour appeler notre API de prévision
@tool
def prevoir_rendement(parcelle_id: str, annee: str) -> str:
    """Utilise cet outil pour obtenir la prévision de rendement, l'intervalle de confiance et les facteurs SHAP d'une parcelle spécifique."""
    url = "http://127.0.0.1:8000/predict"
    try:
        # On force annee en nombre (int) et on ajoute un timeout de 10 secondes
        reponse = requests.post(url, json={"parcelle_id": parcelle_id, "annee": int(annee)}, timeout=10)
        if reponse.status_code == 200:
            data = reponse.json()
            return f"Rendement médian: {data['rendement_median_t_ha']} t/ha. Intervalle: {data['intervalle_80']}. Facteurs: {data['facteurs_shap']}"
        return "Erreur: Parcelle introuvable."
    except requests.exceptions.ConnectionError:
        return "Erreur: Impossible de contacter l'API. Avez-vous lancé Uvicorn ?"
    except Exception as e:
        return f"Erreur: {e}"
    
# 2. Définir l'outil pour chercher dans les documents (RAG)
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
vectorstore = Chroma(persist_directory="data/chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

@tool
def rechercher_documents(question: str) -> str:
    """Utilise cet outil pour chercher des conseils agronomiques dans les documents (ISRA, FAO, etc.)."""
    docs = retriever.invoke(question)
    # On renvoie le texte ET la source (le nom du fichier)
    contexte = ""
    for doc in docs:
        source = doc.metadata.get("source", "Inconnu")
        contexte += f"Source [{source}] : {doc.page_content}\n\n"
    return contexte

# 3. Créer le LLM et l'Agent
llm = ChatOllama(model="llama3.1:8b", temperature=0)

tools = [prevoir_rendement, rechercher_documents]

# Le prompt système (les règles de l'agent)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Tu es un assistant agronomique intelligent. Tu as accès à deux outils : "
               "1. `prevoir_rendement` pour les questions sur une parcelle spécifique (ex: PAR0002). "
               "2. `rechercher_documents` pour les questions techniques ou agronomiques générales. "
               "IMPORTANT : Ne réponds JAMAIS avec du texte JSON ou du code. "
               "Utilise tes outils en arrière-plan et rédige ensuite une réponse naturelle en français. "
               "Si tu utilises l'outil de recherche, n'oublie pas de citer la source du document. "
               "Si tu ne trouves pas la réponse, dis-le."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# --- C'EST ICI QUE CA SE JOUAIT ---
# Il faut créer l'agent, puis l'exécuteur
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 4. Tester l'agent
print("\n--- TEST DE L'AGENT ---")

# Question 1 : Il doit utiliser l'API
print("\nQuestion 1 : Quel est le rendement de la parcelle PAR0002 en 2025 ?")
res1 = agent_executor.invoke({"input": "Quel est le rendement de la parcelle PAR0002 en 2025 ?"})
print(f"Réponse de l'IA : {res1['output']}")

# Question 2 : Il doit utiliser le RAG
print("\nQuestion 2 : Que faire si la pluie s'arrête pendant 15 jours ?")
res2 = agent_executor.invoke({"input": "Que faire si la pluie s'arrête pendant 15 jours ?"})
print(f"Réponse de l'IA : {res2['output']}")
