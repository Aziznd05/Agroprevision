from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

print("Chargement de la base vectorielle...")
# On recharge la base que l'on vient de créer
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
vectorstore = Chroma(persist_directory="data/chroma_db", embedding_function=embeddings)

# La question de test (on n'utilise pas les mots exacts du document)
question = "Que faire si la pluie s'arrête pendant 15 jours ?"
print(f"\nQuestion posée : {question}\n")

# On cherche les 2 passages les plus pertinents dans la base
resultats = vectorstore.similarity_search(question, k=2)

print("--- RESULTATS DE LA RECHERCHE (Top 2) ---")
for i, doc in enumerate(resultats):
    print(f"\nPassage {i+1} :")
    print(doc.page_content)