from langchain_community.document_loaders import TextLoader
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from operator import itemgetter
from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

docs = [
    TextLoader("advanced_search.txt",encoding='utf-8').load(),
    TextLoader("buying_as_customer.txt",encoding='utf-8').load(),
]
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100,
    separators=["\n\n", "\n", ".", ";", ",", " "],
)
docs_chunks = [text_splitter.split_documents(doc) for doc in docs]
embedder = NVIDIAEmbeddings(
  model="nvidia/nv-embedqa-mistral-7b-v2",
  api_key="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",
  truncate="NONE",
  )


vecstores=[]
vecstores += [FAISS.from_documents(doc_chunks, embedder) for doc_chunks in docs_chunks]
embed_dims = len(embedder.embed_query("test"))

agg_vstore = FAISS(
        embedding_function=embedder,
        index=IndexFlatL2(embed_dims),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        normalize_L2=False
    )

for vstore in vecstores:
        agg_vstore.merge_from(vstore)
    
print(type(agg_vstore))
agg_vstore.save_local("docstore")

#!tar xzvf docstore_index.tgz
#new_db = FAISS.load_local("docstore", embedder, allow_dangerous_deserialization=True)