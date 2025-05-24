#!/bin/bash

# Importante: dar permisos de superusuario (solo necesario en App Service)
export DEBIAN_FRONTEND=noninteractive

# Instalar librerías del sistema necesarias para OpenCV, torch, matplotlib, etc.
apt update && apt install -y \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgtk2.0-dev \
    libgtk-3-dev \
    ffmpeg \
    libatlas-base-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libdc1394-22 \
    libdc1394-22-dev

# Ir al directorio donde está el manage.py y el wsgi.py
cd backend

# Ejecutar gunicorn apuntando a backend.wsgi
gunicorn --workers=2 backend.wsgi:application --bind=0.0.0.0:8000 --timeout 600
