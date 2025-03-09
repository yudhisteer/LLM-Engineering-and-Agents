import os
import glob
from dotenv import load_dotenv
import gradio as gr
import logging
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from openai import OpenAI


# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"






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
    print("Number of chunks: ", len(chunks))
    # print the first chunk
    print("First chunk: ", chunks[0].page_content)

    # print the document types
    doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)
    print(f"Document types found: {', '.join(doc_types)}")

    # print the chunks that contain "CEO"
    for chunk in chunks:
        if 'CEO' in chunk.page_content:
            print(chunk)
            print("_________")

