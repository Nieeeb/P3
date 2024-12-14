# Use an official Python image as the base
FROM python:3.11-slim

EXPOSE 8000

COPY requirements.txt /opt/app/requirements.txt
COPY ../../Modules /opt/app/Modules
COPY ../../Models /opt/app/Models

WORKDIR /opt/app

# Install necessary Python libraries
RUN pip install --no-cache-dir -r requirements.txt 

COPY . /opt/app

# Set the command to run when the container starts
CMD ["uvicorn Modules.Deployment.deployment:app --reload"]
