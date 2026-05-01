# Stage 1: Build React frontend
FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend

COPY frontend/package*.json ./

RUN npm install

COPY frontend/ ./

RUN npm run build

# Stage 2: Final runtime image
FROM python:3.10-slim

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ENV CONTAINER_HOME=/var/www

WORKDIR $CONTAINER_HOME

COPY src/ $CONTAINER_HOME/src/
COPY data/ $CONTAINER_HOME/data/
COPY --from=frontend-build /app/frontend/dist $CONTAINER_HOME/frontend/dist

CMD ["python", "-m", "gunicorn", "--chdir", "src", "app:app", "--bind", "0.0.0.0:5000", "--log-level", "debug"]
