# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for layer caching)
COPY app/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY app/ .

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]

COPY pipeline.pkl /app/

RUN sed -i 's/\r$//' /app/app.py