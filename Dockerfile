FROM ubuntu:latest
LABEL authors="yanjunwang"
FROM python:3.11-slim

WORKDIR /app

# 1. Install your project dependencies as usual
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install langchain-google-vertexai

# 2. PATCH THE RAGAS BUG
RUN python -c "import os, ragas; print(os.path.join(os.path.dirname(ragas.__file__), 'llms/base.py'))" > /tmp/ragas_path.txt \
    && sed -i 's/from langchain_community.chat_models.vertexai import ChatVertexAI/from langchain_google_vertexai import ChatVertexAI/g' $(cat /tmp/ragas_path.txt) \
    && rm /tmp/ragas_path.txt

# Copy your application source code
COPY . .

CMD ["python", "main.py"]

#ENTRYPOINT ["top", "-b"]