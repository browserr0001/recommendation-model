FROM python:3.10-slim

# Set working directory to the project root (not inside /app yet)
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole app directory (so Python sees `app` as a package)
COPY app ./app

# Copy the router script into the image root
COPY router.py ./router.py

# Expose backend port
EXPOSE 5001

# Run the app as a module (so "app.main" is importable)
CMD ["python", "-m", "app.main"]