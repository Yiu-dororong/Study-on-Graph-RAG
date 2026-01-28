# Study of Graph RAG

This repository mainly serves as a study record on Graph RAG and demonstration of my understanding and work.

## Introduction

RAG (Retrieval-Augmented Generation) has been a useful measure to extend LLM's capability to external knowledge base, such as specific or private domain and current information beyond model's knowledge cutoff. In 2024, Microsoft developped the idea of [Graph RAG](https://arxiv.org/abs/2404.16130) which is a new idea putting RAG into a graph structure connected by relationship of different entities, then grouped into communites and generated community summaries.

*Chart from Microsoft's paper*

<img width="712" height="427" alt="image" src="https://github.com/user-attachments/assets/d34062cd-b0d0-4b98-8d62-4f87378ed6f6" />

## Why Graph RAG?

The paper mentioned two main usage - enable global sensemaking and multi-hop, which traditional RAG struggles, as traditional RAG only retrieves seperate chunks of text without making any connections among them. Use cases includes academic research and discovering complex relationships.

## How it works?

Based on [Microsoft's documentation](https://microsoft.github.io/graphrag/query/overview/), there are 3 main types of search, local, global and DRIFT, which are similar to depth, breath and hybird. Local search finds the relevant entities and their neighbours, global search refines from communities summaries. DRIFT search is essentially questioning by ToT (tree of thoughts) along the knowledge graph. It first generates HyDE (Hypothetical Document Embeddings) as a synthetic result to find the most relevant community, and then continuously conducts local search to refine the answer step by step.

Alex Lucek provided a great explaination on it. Here is his [video](https://youtu.be/6vG_amAshTk?si=s_8V6Wp5TU9WSEHv) and [notebook](https://github.com/ALucek/GraphRAG-Breakdown/blob/main/graph_examples.ipynb). He also demostrated the difference on output given by Graph RAG over traditional RAG.

## Motivation

When the word "knowledge graph" comes to me, my first intuition is that it can visualise or at least figure out the underlying relationship inside the knowledge. I believe this would be extremely useful for scholars, for example, compliing all materials of a course into a graph knowledge database so as to support graph RAG, help organise the relations among items(entities).

[This paper](https://arxiv.org/html/2509.16780v1) has conducted experiments on this idea, and also provided some examples of applications like RetLLM-E and CourseAssist. Therefore, I chose to implement GraphRAG on modelling complex relationships and to observe its performances.

## Setup

* Platform: LlamaIndex

* Models: Ollama (mistral-nemo for LLM, nomic-embed-text for embedding)

* Database: Neo4j (It provide an [introduction](https://neo4j.com/blog/genai/what-is-graphrag) to graph RAG too.)

* GPU: RTX 4070 (12GB VRAM)

* Data: Sythetic data from Gemini3, around 2400 words passage about whole course of the World War II

The idea is to create a graph database by user itself, so I would use a small-scale perspective to evaluate the process.

## Process

* Graph RAG

LlamaIndex has provided a its complete own end-to-end setup on [their website](https://developers.llamaindex.ai/python/examples/cookbooks/graphrag_v2/). Please note that you need docker to set up local Neo4j. You may also read the [notebook written by tuhinsharma121](https://github.com/tuhinsharma121/ai-playground/blob/main/rag/graphrag/llamaindex-graphrag/graphrag-llamaindex-relationship-summary.ipynb), which is basically the same thing.

*Tips:* I found out that DoclingReader and DoclingNodeParser from LlamaIndex can easily exceed the context window of embedding model(usually 8192) even with the aid of HybirdChunker.

Also, models definitely matters. If you have access to heavyweight LLM via API, you are advised to use it because building graph is computationally intensive and this is likely to happen with lightweight models.
<img width="1835" height="257" alt="image" src="https://github.com/user-attachments/assets/fcd8956e-7607-4c79-a4fa-55573fd8e569" />

First, we load the models from Ollama, set ```additional_kwargs={"num_gpu": -1}``` to fully use GPU to run the process. ```temperature = 0``` can also be set to stabilize output.

Second, we read the file(s) and then split into chunks. ```SentenceSplitter``` would do the job, but you can choose others like ```SemanticSplitterNodeParser```. However, I do not observe any huge differences.

Next, it would be the main process. To keep the code clean, I copy all the source code from LlamaIndex into ```GraphRAG.py```. What we do here is to use its extractor to build triplets (entity-relationship-entity) to contruct the knowledge graph, and then store it in the graph store along with indexes. We will also build communites that group similar nodes together using ```build_communities()``` from ```GraphRAGStore```.

If you would want to visualize the knowledge graph, you can use the built-in method ```_create_nx_graph()``` as follow:

```
# Optional: plot the knowledge graph
import matplotlib.pyplot as plt
import networkx as nx

nx_graph = graph_store._create_nx_graph()

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(nx_graph)
nx.draw(nx_graph, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10)
plt.title("GraphRAG Knowledge Graph")
plt.show()
```

However, I believe you can simply view it in Neo4j. The result should be like this:

<img width="3294" height="1926" alt="bloom-visualisation" src="https://github.com/user-attachments/assets/517c0b76-3033-4776-96b8-d4664492847d" />

If you take a closer look, it indeed captured quite well on different relationships. (Each green node represent a work chunk.)

<img width="1076" height="781" alt="image" src="https://github.com/user-attachments/assets/906acdbd-1b4c-4ac5-87c8-8417d7b4ffd9" />

Finally, the ```GraphRAGQueryEngine``` will complete the job, the method is similar to Microsoft's global search. Unfortunately, this is only way to search.

Since we save the whole graph into graph store, the graph can be reused if needed.

```
index = PropertyGraphIndex.from_existing(
property_graph_store=graph_store,
llm=llm,
embed_model=embed_model,
)
# Rebuild communites is needed
index.property_graph_store.build_communities()  
```

It took me around 10 minutes to complete a short passage (6 pages of text). For a 20-page academic paper, it may take half an hour, and it may takes hours to complete 200-pages textbook. It is definitely resource intensive. For a larger scale, this [YouTube video](https://youtu.be/vX3A96_F3FU?si=Er50SdrkzVlxCRCc&t=885) has revealed that turning [A Christmas Carol](https://www.gutenberg.org/cache/epub/24022/pg24022.txt) (32k words, 105 pages) into a knowledge graph based on Microsoft's framework can spent up to 1 million tokens (GPT-4o model) which costs USD 7.12.

Reusing graph by loading from Neo4j in LlamaIndex tends to perform weaker than the first run, but you may use Cypher to explore. ```mistral-nemo``` model tends to be very "helpful" to adding its own knowledge to the query, so please always specify that use the retrieval information only. Inference time is acceptable, consider giving fine-grained answers.

I did not calculate the recall for it, but if you want the source text, you can use:

```
retriever = index.as_retriever(include_text=True)
nodes = retriever.retrieve(query)
```

Overall, more than half of the query can answer correctly with accurate details, but it sometimes would tell me that the summaries did not mention when the keywords are explictly in the original text. I would not this is a huge improvement correspond to traditional RAG.

* Simple RAG

Now, I will implement a simple RAG using LlamaIndex again, using the same settings (models, splitter and so on). Only few lines of code are needed.

```
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex(nodes=nodes)
query_engine = index.as_query_engine()
```

This is just for demostration purpose, you can use other stores like ChromaDB, Milvus or Neo4j.

Here I tested some queries to how well they performs. 

### Query 1 (Local): Without using external knowledge, what were the names of the atomic bombs dropped in 1945, and which specific aircraft delivered the first one?

<details>
<summary> Simple RAG </summary>
The atomic bombs dropped on Hiroshima and Nagasaki in 1945 were named "Little Boy" and "Fat Man," respectively. The B-29 bomber Enola Gay delivered the first bomb, Little Boy, over Hiroshima on August 6, 1945.
</details>

<details>
<summary> Graph RAG </summary>
The atomic bombs dropped in 1945 were named "Little Boy" and "Fat Man", delivered by a B-29 bomber named "Enola Gay".
</details>

Both of them gave the correct answer.

### Query 2 (Global summary): Without using external knowledge, how did Germany change the tide from winning to losing?  

<details>
<summary> Simple RAG </summary>
Germany's fortunes changed dramatically when they failed to achieve a decisive victory in Russia. The Battle of Stalingrad was a turning point; despite initial success, Hitler's refusal to allow retreat led to the destruction of the entire Sixth Army. This loss opened up an opportunity for Allied advancements in both the Eastern and Western Fronts, ultimately leading to Germany's defeat.
</details>

<details>
<summary> Graph RAG </summary>
Based solely on the provided summaries, here's how Germany shifted from initial successes to ultimate losses in World War II:

1. **Overreach and Strategic Missteps:**
   - Hitler initiated several offensive campaigns as desperate attempts to reverse his fortunes:
     - **Battle of the Bulge (1944):** A significant strategic misstep that diverted resources from other fronts.
     - **Operation Barbarossa (1941):** Invasion of the Soviet Union opened a vast eastern front, stretching German resources thin.

2. **Allied Resistance and Counteroffensives:**
   - The Allies successfully repulsed Hitler's offensive campaigns:
     - Battle of Britain: British Royal Air Force defended their skies, marking Germany's first major defeat.
     - Operation Dynamo (Dunkirk): Evacuation of nearly 340,000 troops from France boosted Allied morale and demonstrated that Germany could be stopped.

3. **Loss of Key Territories:**
   - By mid-1944, Allies invaded France and were pushing into Germany from the west, while Soviet forces advanced rapidly from the east.
   - This loss of territories made it increasingly difficult for Germany to maintain its war effort.

4. **War on Multiple Fronts:**
   - Germany was engaged in a war on multiple fronts: Western Europe, Eastern Europe, North Africa, and Italy.
   - This spread of resources and manpower made it challenging for Germany to maintain a decisive advantage anywhere.

5. **Economic Strain:**
   - The prolonged war effort put significant strain on Germany's economy:
     - Shortages of materials, fuel, and labor became increasingly severe, hampering Germany's ability to sustain its military efforts.
     - Economic instability and national humiliation after World War I led to the rise of Adolf Hitler and the Nazi Party, ultimately causing another world war.

6. **Battle of Stalingrad (1942-1943):** Despite initial successes, this battle became a stalemate that lasted over six months, resulting in heavy casualties for Germany.
7. **Warsaw Uprising (1944) and other partisan activities:** These resistance efforts tie down German forces, diverting resources away from other fronts.

These factors, combined with the relentless pressure from Allied forces, contributed to Germany's ultimate defeat in World War II.
</details>

Graph RAG provide much more detailed answer compared to simple RAG. This is due to the fact that Graph RAG usually receive more context. However, not all information is from the original text like point 7, Warsaw uprising is never mentioned, lowering its faithfulness.

### Query 3 (Global linkage): Without using external knowledge, how did the failure of the League of Nations in Manchuria relate to Hitler’s later actions in Europe?

<details>
<summary> Simple RAG </summary>
The failure of the League of Nations to effectively respond to Japan's invasion of Manchuria in 1931 demonstrated the limitations of international collective security. This incident highlighted that appeasement policies and lack of decisive action by global powers could embolden aggressive nations like Germany, led by Hitler later on in Europe.
</details>

<details>
<summary> Graph RAG </summary>
Based on the information provided across multiple summaries, here's a concise response:

1. **Direct Relation (without external knowledge)**: The failure of the League of Nations to effectively address Japan's invasion of Manchuria might have encouraged Adolf Hitler's aggressive actions in Europe by signaling limited consequences for aggressive expansion and emboldening an expansionist mindset among leaders.

2. **Indirect Relation**: Some summaries suggest that the League's powerlessness to stop Germany's aggression in Manchuria could have encouraged Hitler to continue his aggressive policies in Europe, with Neville Chamberlain's appeasement policy also reflecting the League's ineffectiveness.

3. **No Direct Relation**: Most summaries do not directly relate the failure of the League of Nations in Manchuria to Hitler’s later actions in Europe due to lack of explicit mention or implication.
</details>

Graph RAG indeed gave the same answer as simple RAG, but with more noises from the huge amount of communities summaries.

Please note that Graph RAG sometimes fails to answer like this:

```
Based solely on the provided community summaries, there is no mention of a "Scorched Earth" policy in relation to any operations launched on June 22, 1941 by Germany, Japan, or other powers involved in World War II.
```

```
Based solely on the provided community summary, there is no explicit mention of a direct connection or relationship between the failure of the League of Nations in Manchuria and Hitler's later actions in Europe.
```

## Conclusion

It seems that Graph RAG barely won in my implementation, but make no mistake, this is only true for a short passage like mine that simple RAG can capture the global context easily. For a clear comparison, please check [Microsoft's post](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/).

However, considering the huge cost difference compared to build the vector/graph and querying, opting for Graph RAG is still not optimal. Indeed, [LangChain](https://reference.langchain.com/v0.3/python/neo4j/graphs/langchain_neo4j.graphs.graph_document.GraphDocument.html#langchain_neo4j.graphs.graph_document.GraphDocument) has a graph RAG plugin with Neo4J as well, but it is outdated already and I believe it gives similar result like LlamaIndex.

Simple RAG has made serveral improvements on the quality of search in recent years, such as MMR, BM25 and re-ranker, it can already give high-quality outputs with careful design of search method. Meanwhile, Graph RAG is making progress too, [LightRAG](https://github.com/HKUDS/LightRAG) adopted the idea of Graph RAG to capture major connections between entities from local and global levels. This greatly reduced the computations needed while still able to give satisfactory results. However, the set-up cost is still much higher than simple RAG. It recommends that the LLM model needs to have at least 32 billion parameters. Hence, Graph RAG remains a choice to improve quality with high marginal cost. Although contructing a knowledge graph may be impressive and useful, the costs of set-up, maintainence and query have to be acknowledged.





