# base image
FROM python:3.11-slim

# workdir
WORKDIR /app

#copy
COPY . /app

#run
RUN pip install -r requirements.txt

# commands
CMD ["python3", "app.py"]