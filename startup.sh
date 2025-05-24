#!/bin/bash

# Importante: dar permisos de superusuario (solo necesario en App Service)
export DEBIAN_FRONTEND=noninteractive

# Instalar librerías del sistema necesarias para OpenCV, torch, matplotlib, etc.
apt update && apt install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 libgtk2.0-dev libgtk-3-dev ffmpeg libatlas-base-dev libjpeg-dev zlib1g-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev libtbb2 libtbb-dev libdc1394-25 libdc1394-dev
# Ir al directorio donde está el manage.py y el wsgi.py
cd backend

# Ejecutar gunicorn apuntando a backend.wsgi

