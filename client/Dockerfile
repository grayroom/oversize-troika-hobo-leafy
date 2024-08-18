# Streamlit client Dockerfile

# Use the official lightweight Python image
FROM  python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY ../requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code to the working directory
COPY . .

# Make port 8501 available to the world outside this container
EXPOSE 8501


