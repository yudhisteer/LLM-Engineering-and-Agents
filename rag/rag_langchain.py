import os
import glob
from dotenv import load_dotenv
import gradio as gr
import logging
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings()
MODEL = "gpt-4o-mini"
DB_NAME = "vector_db"


def create_vector_store(
        documents: list[Document],
        embeddings: OpenAIEmbeddings,
        db_name: str = "vector_db"
    ) -> Chroma:

    """
    Create a vector store from a list of documents.
    """
    
    # Check if a Chroma Datastore already exists - if so, delete the collection to start from scratch
    if os.path.exists(db_name):
        Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=db_name)
    logger.info(f"Vectorstore created with {vectorstore._collection.count()} documents")
    return vectorstore


def load_documents(parent_folder: str) -> list[Document]:
    """
    Load documents from a parent folder.
    """
    # set the text loader kwargs
    text_loader_kwargs = {'encoding': 'utf-8'}
    documents = []
    for folder in parent_folder:
        # get the folder name
        doc_type = os.path.basename(folder)
        # load the documents
        loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
        folder_docs = loader.load()
        for doc in folder_docs:
            # add the doc type to the metadata
            doc.metadata["doc_type"] = doc_type
            # add the document to the list
            documents.append(doc)

    logger.info(f"Loaded {len(documents)} documents from {parent_folder}")
    return documents



def visualize_vector_space(vectorstore: Chroma, type = "2D") -> None:

    """
    Visualize the vector space of the vector store.
    """
    collection = vectorstore._collection
    result = collection.get(include=['embeddings', 'documents', 'metadatas'])
    vectors = np.array(result['embeddings'])
    documents = result['documents']
    doc_types = [metadata['doc_type'] for metadata in result['metadatas']]
    colors = [['blue', 'green', 'red', 'orange'][['products', 'employees', 'contracts', 'company'].index(t)] for t in doc_types]

    

    if type == "2D":
        logger.info("Visualizing vector space in 2D...")
        tsne = TSNE(n_components=2, random_state=42)
        reduced_vectors = tsne.fit_transform(vectors)

        # Create the 2D scatter plot
        fig = go.Figure(data=[go.Scatter(
            x=reduced_vectors[:, 0],
            y=reduced_vectors[:, 1],
            mode='markers',
            marker=dict(size=5, color=colors, opacity=0.8),
            text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
            hoverinfo='text'
        )])

        fig.update_layout(
            title='2D Chroma Vector Store Visualization',
            scene=dict(xaxis_title='x',yaxis_title='y'),
            width=800,
            height=600,
            margin=dict(r=20, b=10, l=10, t=40)
        )

    elif type == "3D":
        logger.info("Visualizing vector space in 3D...")
        tsne = TSNE(n_components=3, random_state=42)
        reduced_vectors = tsne.fit_transform(vectors)

        # Create the 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=reduced_vectors[:, 0],
            y=reduced_vectors[:, 1],
            z=reduced_vectors[:, 2],
            mode='markers',
            marker=dict(size=5, color=colors, opacity=0.8),
            text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
            hoverinfo='text'
        )])

        fig.update_layout(
            title='3D Chroma Vector Store Visualization',
            scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
            width=900,
            height=700,
            margin=dict(r=20, b=10, l=10, t=40)
        )
        
    else:
        raise ValueError(f"Invalid visualization type: {type}. Choose between '2D' or '3D'.")

    fig.show()

if __name__ == "__main__":

    # --------------------------------------------------------------
    # Step 1: Load documents
    # --------------------------------------------------------------

    parent_folder = glob.glob("rag/data/*")
    documents = load_documents(parent_folder)
    # print("Documents: ", documents)

    # print the metadata
    # print(documents[0].metadata)
    # print the page content
    # print(documents[0].page_content)

    # --------------------------------------------------------------
    # Step 2: Split documents
    # --------------------------------------------------------------

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print("Number of chunks: ", len(chunks)) #123
    # print the first chunk
    # print("First chunk: ", chunks[0].page_content)

    # print the document types
    doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)
    print(f"Document types found: {', '.join(doc_types)}")

    # print the chunks that contain "CEO"
    # for chunk in chunks:
    #     if 'CEO' in chunk.page_content:
    #         print(chunk)
    #         print("_________")


    # --------------------------------------------------------------
    # Step 3: Create a vector store
    # --------------------------------------------------------------

    vectorstore = create_vector_store(chunks, embeddings, DB_NAME)

    # Get one vector and find how many dimensions it has
    collection = vectorstore._collection
    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"The vectors have {dimensions:,} dimensions") #1,536 dimensions
    print("Sample embedding: ", sample_embedding)


    # --------------------------------------------------------------
    # Step 4: Visualize the vector space
    # --------------------------------------------------------------

    visualize_vector_space(vectorstore, type="2D")
