import os

import requests
from exceptions import ServiceNotAvailable
from langchain.prompts import PromptTemplate
from utils import remove_description_table


class Summary:
    def __init__(self, file_name: str, paper: str, summary: str):
        self.file_name = file_name
        self.paper = paper
        self.summary = summary

    def to_display_dict(self) -> dict:
        return {"file_name": self.file_name, "summary": self.summary}


class Summarizer:
    prompt_template = PromptTemplate.from_template(
        """\
        You are an helpful assistant who summarizes paragraphs of research papers in two sentences.
        
        Paragraph: {paragraph}
        
        Summary: 
        """
    )

    @staticmethod
    def llm(user_prompt: str) -> str:
        data = {"prompt": user_prompt}
        llm_api_url = os.getenv(
            "LLM_API_URL", "http://host.docker.internal:8000/v1/completions"
        )

        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }

        response = requests.post(llm_api_url, headers=headers, json=data)

        response.raise_for_status()

        content = response.json()["choices"][0]["text"]
        return content

    @staticmethod
    def _remove_description_tables(paragraphs: list) -> list:
        paragraphs = [remove_description_table(p) for p in paragraphs]
        return paragraphs

    @staticmethod
    def _split_paragraphs(paper: str) -> list:
        paragraphs = paper.split("----")
        return paragraphs

    @staticmethod
    def _get_paragraphs(paper: str) -> list:
        paragraphs = Summarizer._split_paragraphs(paper)
        paragraphs = Summarizer._remove_description_tables(paragraphs)
        return paragraphs

    @staticmethod
    def _clean_paragraph(paragraph: str) -> str:
        paragraph = paragraph.replace("TITLE PARAGRAPH:", "")
        return paragraph.strip()

    @staticmethod
    def summarize_paragraph(paragraph: str) -> str:
        clean_paragraph = Summarizer._clean_paragraph(paragraph)
        if clean_paragraph:
            user_prompt = Summarizer.prompt_template.format(paragraph=clean_paragraph)
            try:
                answer = Summarizer.llm(user_prompt)
            except requests.exceptions.ConnectionError as ex:
                raise ServiceNotAvailable(
                    service_name="llama", status_code=0, error_string=str(ex)
                )
            return answer
        else:
            return ""

    @staticmethod
    def _postprocess_answer(answer: str) -> str:
        answer = answer.strip()
        answer = answer + "\n"
        return answer

    @staticmethod
    def summarize_paper(paper: str) -> str:
        paragraphs = Summarizer._get_paragraphs(paper)
        answers = []

        for i, paragraph in enumerate(paragraphs):
            print(f">>> Summarizing {i + 1}/{len(paragraphs)} paragraphs...")
            answer = Summarizer.summarize_paragraph(paragraph)
            answer = Summarizer._postprocess_answer(answer)
            answers.append(answer)
        return "".join(answers).strip()

    def run(self, papers: dict) -> list:
        summaries = []
        for file_name, paper in papers.items():
            print(f">>> Summarizing paper {file_name}...")
            summary = Summarizer.summarize_paper(paper)
            summaries.append(Summary(file_name=file_name, paper=paper, summary=summary))
            print(f">>> Finished summarizing paper {file_name}!")
        print(">>> Finished summarizing!")
        return summaries
