#!/bin/bash

# Instalar libGL para que OpenCV no falle
apt update && apt install -y libgl1

# Ir al directorio donde está el manage.py y el wsgi.py
cd backend

# Ejecutar gunicorn apuntando a backend.wsgi
gunicorn --workers=2 backend.wsgi:application --bind=0.0.0.0:8000 --timeout 600