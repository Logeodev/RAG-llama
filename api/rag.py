import ollama, chromadb, os
from smolagents import ToolCallingAgent, Tool, LiteLLMModel
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

## Based on smolagents RAG example: https://huggingface.co/docs/smolagents/examples/rag

# ChromaDB setup
CHROMA_CLIENT = chromadb.Client()
COLLECTION_NAME = "rag_docs"
COLLECTION = CHROMA_CLIENT.get_or_create_collection(COLLECTION_NAME)

# LLM and embedding model setup (Ollama)
LLM_MODEL = "granite3-dense:8b" #"qwen3:8b"
OLLAMA_URI = os.getenv("LLM_URL", "http://localhost:11434")
OLLAMA_CLIENT = ollama.Client(host=OLLAMA_URI)

# Text splitter for processing documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)

def embed_file_to_chroma(filename: str, content: str):
    """Embed a file's content into ChromaDB with chunking"""
    # Create document and split into chunks
    doc = Document(page_content=content, metadata={"source": filename})
    chunks = text_splitter.split_documents([doc])
    
    # Embed each chunk
    for i, chunk in enumerate(chunks):
        doc_id = f"{filename}_chunk_{i}"
        
        COLLECTION.add(
            ids=[doc_id],
            documents=[chunk.page_content],
            metadatas=[{"source": filename, "chunk_index": i}]
        )
    
    return {"id": filename, "chunks": len(chunks)}

class ChromaRetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve the most relevant document chunks from the knowledge base to answer your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, query: str) -> str:
        """Execute the retrieval based on the provided query."""
        assert isinstance(query, str), "Your search query must be a string"
        
        # Retrieve relevant documents from ChromaDB
        results = COLLECTION.query(
            query_texts=query,
            n_results=10,
            include=["documents", "distances", "metadatas"],
        )
        
        if not results['documents'] or not results['documents'][0]:
            return "No relevant documents found in the knowledge base."
        
        docs = results['documents'][0]
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        
        # Format the retrieved documents for readability
        formatted_docs = []
        for i, (doc, metadata) in enumerate(zip(docs, metadatas)):
            source = metadata.get('source', 'Unknown') if metadata else 'Unknown'
            chunk_idx = metadata.get('chunk_index', '') if metadata else ''
            formatted_docs.append(
                f"\n\n===== Document {i+1} (Source: {source}, Chunk: {chunk_idx}) =====\n{doc}"
            )
        
        return "\nRetrieved documents:\n" + "".join(formatted_docs)

class RAG_Agent:
    def __init__(self):
        self.tools = [ChromaRetrieverTool()]
        self.model = LiteLLMModel(model_id=f"ollama_chat/{LLM_MODEL}", api_base=OLLAMA_URI)
        self.agent = ToolCallingAgent(
            tools=self.tools,
            model=self.model,
            max_steps=2,
            verbosity_level=2
        )
    
    def run(self, prompt: str) -> str:
        """Run the agent with the given prompt"""
        return self.agent.run(prompt)
    
    def embed_document(self, filename: str, content: str):
        """Convenience method to embed documents"""
        return embed_file_to_chroma(filename, content)
    
    def get_collection_info(self):
        """Get information about the current collection"""
        count = COLLECTION.count()
        return {"collection_name": COLLECTION_NAME, "document_count": count}
