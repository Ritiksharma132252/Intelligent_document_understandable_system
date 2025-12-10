# Use official Python slim image as a small base
FROM python:3.10-slim

# ---- Metadata / labels (optional) ----
LABEL maintainer="you@example.com"
LABEL description="IDU - Intelligent Document Understanding (Flask + OCR + FAISS + RAG)"

# ---- set environment variables ----
# Prevent Python from writing .pyc files and enable stdout/stderr logging immediately
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# ---- Install system-level dependencies ----
# - tesseract-ocr and poppler-utils are required for OCR (pytesseract & pdf2image)
# - build-essential, gcc, libopenblas-dev help pip install binary wheels (faiss, numpy)
# - libmagic for python-magic (file type detection)
# - git is handy if you fetch packages from VCS
RUN apt-get update && apt-get install -y --no-install-recommends \
  
# ---- Copy dependency files first (leverages Docker cache) ----
# If you use requirements.txt (recommended), copy it and install dependencies
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install Python packages
# Note: faiss-cpu wheels are available on manylinux; if pip fails, consider building from source or using a different base image
RUN pip install --upgrade pip setuptools wheel \
  && pip install --no-cache-dir -r /app/requirements.txt

# ---- Copy application code ----
# Copy the whole project (except files in .dockerignore)
COPY . /app

# ---- Create a non-root user for running the app ----
# Improves container security
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser
ENV HOME=/home/appuser

# ---- Expose port (Flask default) ----
EXPOSE 5000

# ---- Environment variables for Flask ----
ENV FLASK_APP=run.py
ENV FLASK_ENV=production

# ---- Optional: create path for FAISS persistence or uploads ----
RUN mkdir -p /app/data/uploads && mkdir -p /app/data/faiss_index
VOLUME ["/app/data"]

# ---- Entrypoint / Command ----
# Use gunicorn for production or simple python for dev. Example uses gunicorn.
# If you prefer the Flask dev server, replace the CMD below with: ["python","run.py"]
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "run:app", "--workers", "2", "--threads", "4", "--timeout", "120"]
