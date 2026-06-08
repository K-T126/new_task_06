# Use official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app/

# Expose port (Cloud Run defaults to 8080)
EXPOSE 8080

# Command to run the application
# We use gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
