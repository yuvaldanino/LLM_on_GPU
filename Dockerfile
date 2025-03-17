# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all files from your project into the container
COPY requirements.txt /app/
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app


# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask API on port 5000
EXPOSE 5000

# Run Flask API
CMD ["python", "server.py"]
