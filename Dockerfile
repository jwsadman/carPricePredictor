# base image
FROM python:3.11-slim

# workdir
WORKDIR /app

#copy
COPY . /app

#run
RUN pip install -r requirements.txt

# Run FastAPI with Uvicorn on port 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]