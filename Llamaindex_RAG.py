import asyncio
import nest_asyncio
nest_asyncio.apply()
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import os
from dotenv import load_dotenv
load_dotenv()

#############################

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

embed_model = OllamaEmbedding(model_name="nomic-embed-text",additional_kwargs={"num_gpu": 0,"num_ctx": 8192})
llm = Ollama(model='mistral-nemo',
             #temperature=0,
              additional_kwargs={"num_gpu": -1,"num_ctx": 8192},
              request_timeout=120.0,
              context_window=8192,)


from llama_index.core import Settings
# Set globally
Settings.embed_model = embed_model
Settings.llm = llm

#############################

from llama_index.core import SimpleDirectoryReader
from llama_index.core import PropertyGraphIndex
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.node_parser import SentenceSplitter

reader = SimpleDirectoryReader(
    input_files=["./ww2.txt"],
)
documents = reader.load_data()
print(f"Loaded {len(documents)} docs")

splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=20,
)
# splitter = SemanticSplitterNodeParser(embed_model=embed_model)
nodes = splitter.get_nodes_from_documents(documents)
print(f"Loaded {len(nodes)} nodes")

#############################

import GraphRAG 

kg_extractor = GraphRAG.GraphRAGExtractor(
    llm=llm,
    extract_prompt=GraphRAG.KG_TRIPLET_EXTRACT_TMPL,
    max_paths_per_chunk=20,
    parse_fn=GraphRAG.parse_fn,
)

graph_store = GraphRAG.GraphRAGStore(
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"), 
    url="bolt://localhost:7687"
)

#############################

# from llama_index.core import VectorStoreIndex

# index = VectorStoreIndex(nodes=nodes)
# query_engine = index.as_query_engine()

#############################

# Load existing knowledge graph
# index = PropertyGraphIndex.from_existing(
#     property_graph_store=graph_store,
#     llm=llm,
#     embed_model=embed_model,)

# Build a new knowledge graph
index = PropertyGraphIndex(
    nodes=nodes,
    kg_extractors=[kg_extractor],
    property_graph_store=graph_store,
    show_progress=True,
)

index.property_graph_store.build_communities()

query_engine = GraphRAG.GraphRAGQueryEngine(
    graph_store=index.property_graph_store,
    llm=llm,
    index=index,
    similarity_top_k=10,
)

# retriever = index.as_retriever(include_text=True)

#############################

while True:
    query = input("Enter query (Type q to exit) ")
    if query.lower() == "q":
        break
    else:
        print(query_engine.query(query)) 
        #print(retriever.retrieve(query))


