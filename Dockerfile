FROM python:3.10 AS build

ARG POETRY_VERSION=2.1.1

RUN set -ex \
    && curl -sSL https://install.python-poetry.org -o /get-poetry.py \
    && POETRY_HOME=/opt/poetry python /get-poetry.py

ENV PATH=/opt/poetry/bin:$PATH

COPY pyproject.toml README.md /workspace/
COPY tone /workspace/tone

WORKDIR /workspace

RUN set -ex \
    && python -m venv /venv --without-pip \
    && . /venv/bin/activate \
    && poetry install --only main -E demo \
    # Reinstall main package in non-editable mode
    && poetry build -f wheel \
    && pip --python /venv/bin/python install dist/*.whl --no-deps --force-reinstall --no-compile \
    && rm -rf ~/.cache

ENV PATH=/venv/bin:$PATH

FROM python:3.10-slim

COPY --from=build /venv /venv
COPY simple_api.py /workspace/

ENV PATH=/venv/bin:$PATH
# Set env variable LOAD_FROM_FOLDER to load model from a local folder instead of downloading from HuggingFace
ENV LOAD_FROM_FOLDER=/models

WORKDIR /workspace

RUN useradd -s /bin/bash python

USER python

STOPSIGNAL SIGINT

CMD ["python", "simple_api.py"]