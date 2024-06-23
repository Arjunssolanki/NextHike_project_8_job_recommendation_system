FROM python:3.10
COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir --verbose -r requirements.txt
RUN python -m nltk.downloader punkt stopwords
CMD python app.py