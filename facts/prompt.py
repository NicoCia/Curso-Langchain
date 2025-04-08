from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from redundant_filter_retriever import RedundantFilterRetriever
from dotenv import load_dotenv
import langchain

langchain.debug = True

load_dotenv()

chat = ChatOpenAI()

embeddings = OpenAIEmbeddings()

db = Chroma( # Crea el vector store a partir de los embeddings calculados previamente
    persist_directory = "emb",
    embedding_function = embeddings
)
# retriever = db.as_retriever()
retriever = RedundantFilterRetriever(
    embeddings = embeddings,
    db = db
)

chain = RetrievalQA.from_chain_type(
    llm = chat,
    retriever = retriever, # Aca le pasamos el vector store que creamos antes pero en una instancia de Retriver que es lo que espera el modelo. Cuenta con el metodo get_relevant_documents que es el que se encarga de buscar los documentos mas relevantes para una consulta.
    chain_type = "stuff"
)

result = chain.run("What is an interesting fact about the English language?")

print(result)
