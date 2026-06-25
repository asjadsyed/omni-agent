FROM python:3.12

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    jq \
    less \
    bsdmainutils \
    xxd \
    netcat-openbsd \
    net-tools \
    iproute2 \
    lsof \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN mkdir -p /artifacts/

COPY . /app

WORKDIR /workspace
CMD [ "python3", "/app/agent.py" ]
