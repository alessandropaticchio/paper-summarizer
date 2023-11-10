import os

import pymilvus
from exceptions import ServiceNotAvailable
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Milvus


class RetrievedAbstract:
    def __init__(self, file_name: str, abstract: str, score: dict):
        self.file_name = file_name
        self.abstract = abstract
        self.score = score

    def to_display_dict(self) -> dict:
        return {
            "file_name": self.file_name,
            "abstract": self.abstract,
            "abstract_similarity_score": self.score,
        }


class Retriever:
    def __init__(self):
        try:
            vector_db = Retriever._get_vector_db()
        except pymilvus.exceptions.MilvusException as ex:
            raise ServiceNotAvailable(
                service_name="milvus", status_code=ex.code, error_string=str(ex)
            )
        self.vector_db = vector_db

    @staticmethod
    def extract_abstracts() -> dict:
        abstracts = {}

        file_list = os.listdir("data")
        txt_files = [file for file in file_list if file.endswith(".txt")]

        for file_name in txt_files:
            file_path = os.path.join("data", file_name)

            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            # Find the index of the first "TITLE PARAGRAPH"
            title_paragraph_index = content.find("TITLE PARAGRAPH")

            # Extract the text before the first "TITLE PARAGRAPH"
            if title_paragraph_index != -1:
                abstract = content[:title_paragraph_index].strip()
                abstracts[file_name] = abstract

        return abstracts

    @staticmethod
    def print_abstract_stats(abstracts: dict):
        for file_name, abstract in abstracts.items():
            print(f"Length of {file_name}: {len(abstract)}")

    @staticmethod
    def save_abstracts(abstracts):
        output_folder = os.getenv("ABSTRACTS_OUTPUT_FOLDER", "data/abstracts")

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        for file_name, abstract in abstracts.items():
            file_path = os.path.join(output_folder, file_name)
            with open(file_path, "w", encoding="utf-8") as text_file:
                text_file.write(abstract)

    @staticmethod
    def generate_abstracts() -> dict:
        abstracts = Retriever.extract_abstracts()
        Retriever.print_abstract_stats(abstracts)
        Retriever.save_abstracts(abstracts)
        return abstracts

    @staticmethod
    def get_embeddings() -> OpenAIEmbeddings:
        embeddings_model_name = os.getenv("OPENAI_EMBEDDINGS", "text-embedding-ada-002")
        return OpenAIEmbeddings(model_name=embeddings_model_name)

    @staticmethod
    def generate_docs(abstracts: dict) -> list:
        documents = []
        output_folder = os.getenv("ABSTRACTS_OUTPUT_FOLDER", "data/abstracts")

        for file_name, abstract in abstracts.items():
            loader = TextLoader(f"{output_folder}/{file_name}")
            doc = loader.load()
            documents += doc

        return documents

    @staticmethod
    def _get_vector_db() -> Milvus:
        embeddings = Retriever.get_embeddings()
        return Milvus(
            embeddings,
            connection_args={"host": "standalone", "port": "19530"},
            collection_name="papers",
        )

    @staticmethod
    def populate_db() -> Milvus:
        print(">>> Extracting abstracts from papers...")
        abstracts = Retriever.generate_abstracts()
        print(">>> Converting abstracts into LangChain docs...")
        docs = Retriever.generate_docs(abstracts)

        embeddings = Retriever.get_embeddings()

        return Milvus.from_documents(
            docs,
            embeddings,
            collection_name="papers",
            connection_args={"host": "standalone", "port": "19530"},
        )

    def _retrieve(self, query: str, top_k: int = 3) -> list:
        return self.vector_db.similarity_search_with_score(query, k=top_k)

    def get_similar_abstracts(self, query: str, top_k: int = 3) -> list:
        items = self._retrieve(query, top_k)

        retrieved_abstracts = []

        for doc, score in items:
            file_name = doc.metadata["source"]
            retrieved_abstract = RetrievedAbstract(
                file_name=file_name, abstract=doc.page_content, score=score
            )
            retrieved_abstracts.append(retrieved_abstract)

        return retrieved_abstracts
