# app/Dockerfile

FROM python:3.9-alpine

WORKDIR /app

# Copy all files to the docker app
COPY . .

RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit","run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
