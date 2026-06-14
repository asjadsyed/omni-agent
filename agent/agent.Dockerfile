FROM python:3.12

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

WORKDIR /workspace
CMD [ "python3", "/app/agent.py" ]
