FROM python:3.8

WORKDIR /app

COPY requirements.txt requirements.txt
COPY bot/ bot/
COPY data/ data/

RUN pip install -r requirements.txt

CMD ["tail -f /dev/null"]
