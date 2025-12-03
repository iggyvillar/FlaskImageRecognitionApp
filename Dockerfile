FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for TensorFlow/Pillow/OpenCV)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Create directories for uploads if needed
RUN mkdir -p uploads static/uploads

# Expose Flask port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Run the application using Flask's production server
CMD ["flask", "run", "--host", "0.0.0.0", "--port", "5000"]
