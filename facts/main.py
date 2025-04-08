from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()


text_splitter = CharacterTextSplitter(
    separator = "\n", # Separator between chunks
    chunk_size = 150, # Maximum size of each chunk in characters. Primero va tomar pedazos de 150 caracteres y despues dentro de eso va buscar el separator para cortar
    chunk_overlap = 0 # Overlap between chunks in characters. Si es 0 no va a haber overlap. con un overlap de x toma los ultimos x caracteres del chunk anterior y los pone al principio del siguiente
)

loader = TextLoader("facts.txt")
docs = loader.load_and_split(
    text_splitter = text_splitter
)

db = Chroma.from_documents( # Calcula los embeddings de los documentos utilizando OpenAI y los guarda en una base de datos.
    docs,
    embedding = embeddings,
    persist_directory = "emb"
)

# results = db.similarity_search_with_score( # Busca documentos similares a una consulta y devuelve una dupla que contiene el score de similitud y el texto del documento.
#     "What is an interesting fact about the English language?", # Query text to compare
#     k = 2 # Number of results to return
# )

# for result in results:
#     print("\n")
#     print(result[1]) # Imprime el score de similitud
#     print(result[0].page_content) # Imprime el contenido de la página


results = db.similarity_search( # Busca documentos similares a una consulta y devuelve solo el texto del documento.
    "What is an interesting fact about the English language?", # Query text to compare
    # k = 1 # Number of results to return
)

for result in results:
    print("\n")
    print(result.page_content) # Imprime el contenido de la página
