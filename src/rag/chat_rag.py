from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Charger la base de connaissances
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
vectorstore = Chroma(persist_directory="data/chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 2. Charger le LLM local (Llama 3 via Ollama)
print("Connexion au LLM local...")
llm = OllamaLLM(model="llama3")

# 3. Créer le prompt
prompt_template = """Tu es un assistant agronomique. Utilise uniquement le contexte suivant pour répondre à la question. 
Si tu ne connais pas la réponse, dis simplement "Je ne trouve pas l'information dans mes documents". 
Ne rajoute pas d'informations qui ne sont pas dans le texte.

Contexte :
{context}

Question : {question}

Réponse en français :
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# 4. Construire la chaîne RAG
rag_chain = (
    {"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)), 
     "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Poser une question de test
question = "Que faire si la pluie s'arrête pendant 15 jours ?"
print(f"\nQuestion : {question}")
print("Recherche dans les documents et génération de la réponse...\n")

reponse = rag_chain.invoke(question)

print("--- REPONSE DE L'ASSISTANT ---")
print(reponse)