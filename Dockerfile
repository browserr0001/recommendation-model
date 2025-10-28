FROM python:3.10-slim

# Set working directory to the project root (not inside /app yet)
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole app directory (so Python sees `app` as a package)
COPY app ./app

# Expose Flask port
EXPOSE 8082

# Run the app as a module (so "app.main" is importable)
CMD ["python", "-m", "app.main"]