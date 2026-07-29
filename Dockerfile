FROM python:3.10-slim

WORKDIR /app

# Install git and system packages
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Set environment port
ENV PORT=7860

# Run Flask server
CMD ["python", "app.py"]
