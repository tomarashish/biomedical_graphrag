import neo4j
from neo4j_graphrag.llm import AzureOpenAILLM
from neo4j_graphrag.embeddings.openai import AzureOpenAIEmbeddings 
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.retrievers import VectorRetriever, VectorCypherRetriever
from neo4j_graphrag.generation.graphrag import GraphRAG
from neo4j_graphrag.generation import RagTemplate
from neo4j_graphrag.indexes import create_vector_index 
import httpx
import os
import asyncio 
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

httpx_client = httpx.Client(verify='./cacert.pem')

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE')

# Read the API key from a file

neo4j_driver = neo4j.GraphDatabase.driver(NEO4J_URI,
                                          auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

try:
   neo4j_driver.verify_connectivity()
   print("Connection successful!")
except Exception as e:
   print(f"Failed to connect to Neo4j: {e}")

openai_llm = AzureOpenAILLM(
    model_name="gpt-4o",
    azure_endpoint="https://api.nlp.dev.uptimize.merckgroup.com",  # update with your endpoint
    api_version="2024-10-21",  # update appropriate version
    api_key= os.getenv('API_KEY') # api_key is optional and can also be set with OPENAI_API_KEY env va
)

embedder = AzureOpenAIEmbeddings(model="text-embedding-ada-002-v2",
                                 azure_endpoint="https://api.nlp.dev.uptimize.merckgroup.com",
                                 api_key=os.getenv('API_KEY'), 
                                 api_version="2024-10-21")

basic_node_labels = ["Object", "Entity", "Group", "Person", "Organization", "Place"]

academic_node_labels = ["ArticleOrPaper", "PublicationOrJournal"]

medical_node_labels = ["Anatomy", "BiologicalProcess", "Cell", "CellularComponent",
                      "CellType", "Condition", "Disease", "Drug",
                      "EffectOrPhenotype", "Exposure", "GeneOrProtein", "Molecule",
                      "MolecularFunction", "Pathway"]

node_labels = basic_node_labels + academic_node_labels + medical_node_labels

# define relationship types
rel_types = ["ACTIVATES", "AFFECTS", "ASSESSES", "ASSOCIATED_WITH", "AUTHORED", "BIOMARKER_FOR"]

kg_builder_pdf = SimpleKGPipeline(
   llm=openai_llm,
   driver=neo4j_driver,
   text_splitter=FixedSizeSplitter(chunk_size=50, chunk_overlap=10),
   embedder=embedder,
   entities=node_labels,
   relations=rel_types,
   from_pdf=True,
   neo4j_database=NEO4J_DATABASE
)

pdf_file_paths = ["./nature_articles/npre.2010.5006.1.pdf", "./nature_articles/s41385-020-00365-4.pdf",
                    "nature_articles/s41385-022-00539-2.pdf"]

# 1. Build KG and Store in Neo4j Database
for path in pdf_file_paths:
        print(f"Processing : {path}")
        asyncio.run(
            kg_builder_pdf.run_async(file_path=path)
         )

# The Vector Retriever uses Approximate Nearest Neighbor (ANN) vector search to retrieve data from your knowledge graph.
#We can create a vector index in Neo4j to allow this retriever to pull back information from Chunk nodes.
create_vector_index(neo4j_driver, name="text_embeddings", label="Chunk",
                   embedding_property="embedding", dimensions=1536, similarity_fn="cosine")

# 2. KG Retriever
vector_retriever = VectorRetriever(
   neo4j_driver,
   index_name="text_embeddings",
   embedder=embedder,
   neo4j_database=NEO4J_DATABASE,
)

vc_retriever = VectorCypherRetriever(
   neo4j_driver,
   index_name="text_embeddings",
   embedder=embedder,
   neo4j_database=NEO4J_DATABASE,
   retrieval_query="""
//1) Go out 2-3 hops in the entity graph and get relationships
WITH node AS chunk
MATCH (chunk)<-[:FROM_CHUNK]-()-[relList:!FROM_CHUNK]-{1,2}()
UNWIND relList AS rel

//2) collect relationships and text chunks
WITH collect(DISTINCT chunk) AS chunks,
 collect(DISTINCT rel) AS rels

//3) format and return context
RETURN '=== text ===\n' + apoc.text.join([c in chunks | c.text], '\n---\n') + '\n\n=== kg_rels ===\n' +
 apoc.text.join([r in rels | startNode(r).name + ' - ' + type(r) + '(' + coalesce(r.details, '') + ')' +  ' -> ' + endNode(r).name ], '\n---\n') AS info
"""
)

prompt_template = '''
You are a medical researcher tasks with extracting information from papers 
and structuring it in a property graph to inform further medical and research Q&A.

Extract the entities (nodes) and specify their type from the following Input text.
Also extract the relationships between these nodes. the relationship direction goes from the start node to the end node. 


Return result as JSON using the following format:
{{"nodes": [ {{"id": "0", "label": "the type of entity", "properties": {{"name": "name of entity" }} }}],
  "relationships": [{{"type": "TYPE_OF_RELATIONSHIP", "start_node_id": "0", "end_node_id": "1", "properties": {{"details": "Description of the relationship"}} }}] }}

- Use only the information from the Input text. Do not add any additional information.  
- If the input text is empty, return empty Json. 
- Make sure to create as many nodes and relationships as needed to offer rich medical context for further research.
- An AI knowledge assistant must be able to read this graph and immediately understand the context to inform detailed research questions. 
- Multiple documents will be ingested from different sources and we are using this property graph to connect information, so make sure entity types are fairly general. 

Use only fhe following nodes and relationships (if provided):
{schema}

Assign a unique ID (string) to each node, and reuse it to define relationships.
Do respect the source and target node types for relationship and
the relationship direction.

Do not return any additional information other than the JSON in it.

Examples:
{examples}

Input text:

{text}
'''

## The GraphRAG Python package makes instantiating and running GraphRAG pipelines easy. 
# We can use a dedicated GraphRAG class. 
# At a minimum, you need to pass the constructor an LLM and a retriever. 
# You can optionally pass a custom prompt template. 
# We will do so here, just to provide a bit more guidance for the LLM to stick to information from our data source.

rag_template = RagTemplate(template='''Answer the Question using the following Context. Only respond with information mentioned in the Context. Do not inject any speculative information not mentioned.
# Question:
{query_text}
# Context:
{context}
# Answer:
''', expected_inputs=['query_text', 'context'])

v_rag  = GraphRAG(llm=openai_llm, retriever=vector_retriever, prompt_template=rag_template)
vc_rag = GraphRAG(llm=openai_llm, retriever=vc_retriever, prompt_template=rag_template)

# 4. Run
q = "what are methods used in the document? list "
q = "what are experiments used in the document? list "
q = "what are library mentioned in the experiments? list all the names"
q = "what are extraction|Extraction used in the experiments? list all the names"
#print(f"Vector Response: \n{v_rag.search(q, retriever_config={'top_k':5}).answer}")
#print(f"Vector + Cypher Response: \n{vc_rag.search(q, retriever_config={'top_k':5}).answer}")
#print("\n===========================\n")

v_rag_result = v_rag.search(q, retriever_config={'top_k': 5}, return_context=True)
vc_rag_result = vc_rag.search(q, retriever_config={'top_k': 5}, return_context=True)

print(f"Vector Response: \n{v_rag_result.answer}")
print("\n===========================\n")
print(f"Vector + Cypher Response: \n{vc_rag_result.answer}")

neo4j_driver.close()
