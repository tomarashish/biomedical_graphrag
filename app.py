import streamlit as st
import os, time
import neo4j
from neo4j_graphrag.llm import AzureOpenAILLM
from neo4j_graphrag.embeddings.openai import AzureOpenAIEmbeddings 
from neo4j_graphrag.retrievers import VectorRetriever, VectorCypherRetriever
from neo4j_graphrag.generation.graphrag import GraphRAG
from neo4j_graphrag.generation import RagTemplate
import openai
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

st.set_page_config(
    layout="wide",
    page_title="UPTIMIZE App Service Streamlit Playground",
    page_icon=":robot_face:",
)

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE')
if not NEO4J_URI or not NEO4J_USERNAME or not NEO4J_PASSWORD or not NEO4J_DATABASE:
    st.error("Please set the environment variables: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, and NEO4J_DATABASE.")
    st.stop()
# Connect to Neo4j
neo4j_driver = neo4j.GraphDatabase.driver(NEO4J_URI,
                                          auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

openai.api_base = "https://api.nlp.dev.uptimize.merckgroup.com"
openai.api_type = "azure"
openai.api_version = "2024-10-21"
#httpx_client = httpx.Client(http2=True, verify=False)

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

vc_rag = GraphRAG(llm=openai_llm, retriever=vc_retriever, prompt_template=rag_template)

def query_neo4j(query, top_k=5):

    # Creates embedding vector from user query
    query_results = ""
    vc_rag_result = vc_rag.search(query, retriever_config={'top_k': 5}, return_context=True)

    return vc_rag_result.answer



#st.image('assets/Lab.jfif',width = 250)

st.title('PubMed GPT')


st.subheader('Welcome To The PubMed Graph Querying Engine')
with st.sidebar:
    st.title("PubMed GPT")
    st.subheader("The GraphRAG Prompt")
    st.markdown(
        """Pubmed Chat is a chatbot built with [Streamlit](https://streamlit.io/) and [Neo4j](https://neo4j.com/) to query pubmed articles using GraphRAG. 
        It offers multiple search options such as Semantic, Hybrid, and Generative Search. """
    )
    st.header("Settings")
    try:
       neo4j_driver.verify_connectivity()
       st.success("Connected to Neo4j client", icon="💚")
    except Exception as e:
        st.error("Connection to Neo4j client Failed", icon="❤️")
    
questions = [
    "How is precision medicine applied to Lupus? provide in list format.",
    "Summarize systemic lupus erythematosus (SLE)? including common effects, biomarkers, and treatments. Provide in detailed list format.",
    "Summarize relationship between IBS and microbiome as list",
    "What is role of microbiome on human health? provide in list format",
    "List reasons for dysbiosis",
    "how gut microbiota contributes to host defense against infection? provide in list format."
]

st.markdown("##### Demo questions for querying AvelumabKG 👇")
st.markdown("Select a question from the list below to quickly query the AvelumabKG:") 
selection = st.pills("", questions, selection_mode="single")
st.sidebar.markdown("---")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Let's start querying pubmed ! 👇"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Seearch PubMed Artciles...") or selection:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    st.chat_message("user").write(prompt)

    prompt = prompt.replace('"', "").replace("'", "")

    if prompt != "":
        query = prompt.strip().lower()
        
        # Display assistant response in chat message container
         
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            assistant_response  = query_neo4j(query)
        
            # Simulate stream of response with milliseconds delay
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
        
                 # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
    
            message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.write(full_response)







