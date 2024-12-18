# Use an official Python image as the base
FROM python:3.11-slim

RUN apt-get update && apt-get upgrade -y

#RUN apt-get -y install opencv-python

EXPOSE 8000

COPY requirements.txt requirements.txt

# Install necessary Python libraries
RUN pip install --no-cache-dir -r requirements.txt 

COPY . .

# Set the command to run when the container starts
CMD ["uvicorn", "deployment:app", "--host", "0.0.0.0", "--port", "8000"]
#CMD "ls"