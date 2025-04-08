from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryMemory# ConversationBufferMemory, FileChatMessageHistory
from dotenv import load_dotenv
from langchain_core.prompts import MessagesPlaceholder

load_dotenv()

chat =  ChatOpenAI(verbose = True)# Para que muestre el detalle de la respuesta generada)

# memory = ConversationBufferMemory(
#     chat_memory = FileChatMessageHistory("messages.json"),
#     memory_key = "messages",
#     return_messages  = True
# )

memory = ConversationSummaryMemory(
    memory_key = "messages",
    # return_messages  = True,
    llm = chat, #Hay que pasarle que LLM queremos que use para generar el resumen
)

prompt = ChatPromptTemplate(
    input_variables = ["content", "messages"],
    messages = [
        MessagesPlaceholder(variable_name = "messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm = chat,
    prompt = prompt,
    memory = memory,
    # verbose = True #Para que muestre el resumen generado
)

while True:
    content = input(">> ")

    # print(f"You entered: {content}")
    result = chain({"content": content})

    print(result["text"])