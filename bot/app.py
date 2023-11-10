import streamlit as st
from evaluator import Evaluator
from retriever import Retriever
from summarizer import Summarizer

from bot import Bot


def instructions():
    st.title("Paper Summarizer")
    st.write("👋 Hello, this is the **Paper Summarizer**. Here's how to use it:")
    st.write("1. 📝 Enter your query in the 'User Input' text box below.")
    st.write("2. 🚀 Once you've entered your query, click the 'Submit' button.")
    st.write(
        "3. 📚 If your query is valid and you've submitted it, the bot will retrieve the top three most relevant "
        "papers from the **Milvus** database by comparing the embeddings of your query with the papers abstracts embeddings."
    )
    st.write("4. 📋 The papers are then passed to **Llama-2** to be summarized.")
    st.write(
        "5. 📊 The summaries are finally evaluated by computing **rouge-1**, **rouge-2** and **rouge-L** scores with respect"
        " to the original papers."
    )

    st.write("The output will be a list of three responses, built as follows:")

    st.code(
        """
    {
  "retrieved_abstract": {
    "file_name": "", # file name of the paper abstract retrieved
    "abstract": "", # abstract of the paper
    "abstract_similarity_score": 0.0 # cosine similarity between query embedding and abstract embedding
  },
  "summary": {
    "summary": {
      "file_name": "", # file name of the paper retrieved 
      "summary": "" # summary of the paper
    },
    "metrics": {
      "rouge1": { # rouge-1 scores between summary and original paper
        "precision": 0.0,
        "recall": 0.0,
        "fmeasure": 0.0
      },
      "rouge2": { # rouge-2 scores between summary and original paper
        "precision": 0.0,
        "recall": 0.0,
        "fmeasure": 0.0
      },
      "rougeL": { # rouge-L scores between summary and original paper
        "precision": 0.0,
        "recall": 0.0,
        "fmeasure": 0.0
      }
    }
  }
}

        """
    )


def main():
    retriever = Retriever()
    summarizer = Summarizer()
    evaluator = Evaluator()
    bot = Bot(retriever, summarizer, evaluator)

    instructions()

    user_input = st.text_input("User Input")

    if st.button("Submit"):
        if user_input:
            bot_response = bot.run(user_input)

            st.write("Summaries:")
            st.write(bot_response)
        else:
            st.warning("Please enter a valid input.")


if __name__ == "__main__":
    main()
