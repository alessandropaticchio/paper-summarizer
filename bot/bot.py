from evaluator import EvaluatedSummary, Evaluator
from retriever import RetrievedAbstract, Retriever
from summarizer import Summarizer


class Response:
    def __init__(
        self, retrieved_abstract: RetrievedAbstract, evaluated_summary: EvaluatedSummary
    ):
        self.retrieved_abstract = retrieved_abstract
        self.evaluated_summary = evaluated_summary

    def to_display_dict(self) -> dict:
        return {
            "retrieved_abstract": self.retrieved_abstract.to_display_dict(),
            "summary": self.evaluated_summary.to_display_dict(),
        }


class Bot:
    def __init__(
        self,
        retriever: Retriever,
        summarizer: Summarizer,
        evaluator: Evaluator,
        top_k=3,
    ):
        self.retriever = retriever
        self.summarizer = summarizer
        self.evaluator = evaluator
        self.top_k = top_k

    @staticmethod
    def _read_papers_files(abstracts: list) -> dict:
        file_names = [a.file_name for a in abstracts]
        papers = {}

        for file_name in file_names:
            file_name = file_name.replace("abstracts/", "")
            with open(file_name, "r", encoding="utf-8") as file:
                paper = file.read()
                papers[file_name] = paper

        return papers

    @staticmethod
    def _build_responses(scored_abstracts: list, evaluated_summaries: list) -> list:
        responses = []
        for scored_abstract, evaluated_summary in zip(
            scored_abstracts, evaluated_summaries
        ):
            response = Response(scored_abstract, evaluated_summary)
            responses.append(response)

        return responses

    def run(self, query: str) -> list:
        print(">>> Extracting similar papers...")
        scored_abstracts = self.retriever.get_similar_abstracts(query, self.top_k)

        print(f">>> Extracted papers: {[a.file_name for a in scored_abstracts]}")

        print(">>> Summarizing similar papers...")
        papers = self._read_papers_files(scored_abstracts)
        summaries = self.summarizer.run(papers)

        print(">>> Evaluating summarized papers...")
        evaluated_summaries = self.evaluator.run(summaries)

        responses = self._build_responses(scored_abstracts, evaluated_summaries)

        responses = [r.to_display_dict() for r in responses]

        print(">>> Retrieval successfully completed!")

        return responses
