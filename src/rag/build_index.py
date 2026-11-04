import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Charger le document
print("Chargement du document...")
loader = TextLoader("data/rag_docs/fiche_arachide.txt", encoding="utf-8")
documents = loader.load()

# 2. Découper le texte en petits morceaux (chunks)
# On garde 100 caractères de chevauchement pour ne pas couper une phrase en deux
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"Le document a été découpé en {len(chunks)} morceaux.")

# 3. Créer le modèle d'embedding (multilingue, recommandé par le cahier des charges)
print("Chargement du modèle d'embedding (cela peut prendre un peu de temps la 1ère fois)...")
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

# 4. Stocker dans la base vectorielle Chroma
print("Création de la base vectorielle...")
# On sauvegarde la base dans un dossier local "chroma_db"
vectorstore = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings,
    persist_directory="data/chroma_db"
)

print("Indexation terminée. La base est prête.")