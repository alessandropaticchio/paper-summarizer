import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from rouge_score import rouge_scorer
from summarizer import Summary
from utils import remove_description_table

nltk.download("stopwords")
nltk.download("punkt")


class EvaluatedSummary:
    def __init__(self, summary: Summary, metrics: dict):
        self.summary = summary
        self.metrics = metrics

    def to_display_dict(self) -> dict:
        return {"summary": self.summary.to_display_dict(), "metrics": self.metrics}


class Evaluator:
    @staticmethod
    def _tokenize(text: str) -> list:
        text = text.lower()
        return nltk.word_tokenize(text)

    @staticmethod
    def _remove_punctuation(tokens: list) -> list:
        tokens = [token for token in tokens if token.isalpha()]
        return tokens

    @staticmethod
    def _remove_stop_words(tokens: list) -> list:
        stop_words = set(
            stopwords.words("english") + ["TITLE", "PARAGRAPH", "ABSTRACT"]
        )
        tokens = [word for word in tokens if word not in stop_words]
        return tokens

    @staticmethod
    def _stem(tokens: list) -> list:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
        return tokens

    @staticmethod
    def preprocess_text(text: str) -> str:
        tokens = Evaluator._tokenize(text)

        tokens = Evaluator._remove_punctuation(tokens)

        tokens = Evaluator._remove_stop_words(tokens)

        tokens = Evaluator._stem(tokens)

        preprocessed_text = " ".join(tokens)

        return preprocessed_text

    @staticmethod
    def _reformat_rouge_score(rouge_score: dict) -> dict:
        rouge_dict = {}

        for k in rouge_score.keys():
            rouge_dict[k] = {}
            rouge_dict[k]["precision"] = rouge_score[k].precision
            rouge_dict[k]["recall"] = rouge_score[k].recall
            rouge_dict[k]["fmeasure"] = rouge_score[k].fmeasure

        return rouge_dict

    @staticmethod
    def compute_rouge(actual: str, predicted: str) -> dict:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
        rouge_score = scorer.score(actual, predicted)
        return Evaluator._reformat_rouge_score(rouge_score)

    @staticmethod
    def get_metrics(actual: str, predicted: str):
        rouge = Evaluator.compute_rouge(actual, predicted)

        return {**rouge}

    def run(self, summaries: list) -> list:
        evaluated_summaries = []
        for summary in summaries:
            actual = summary.paper
            predicted = summary.summary
            actual = remove_description_table(actual)
            actual = self.preprocess_text(actual)
            predicted = self.preprocess_text(predicted)
            metrics = Evaluator.get_metrics(actual, predicted)
            evaluated_summaries.append(EvaluatedSummary(summary, metrics))
        return evaluated_summaries
