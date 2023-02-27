FROM python:3.9.16-slim-bullseye

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=on \
    PIP_DISABLE_PIP_VERSION_CHECK=on

WORKDIR /usr/src

RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install fontconfig -y
RUN apt-get install fonts-nanum*
RUN fc-cache -fv
RUN pip install sqlalchemy pandas plotly streamlit stqdm transformers psycopg2-binary
RUN pip install transformers[sentencepiece]
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

COPY . /usr/src/airtc
WORKDIR /usr/src/airtc/

EXPOSE 8501

CMD ["streamlit", "run", "run.py"]