# app/Dockerfile

FROM python:3.10

WORKDIR /app

# Copy all files to the docker app
COPY . .

RUN pip3 install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit","run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
