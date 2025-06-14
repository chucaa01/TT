
import os
import cv2
import base64
import unittest
import numpy as np
from django.test import TestCase
from rest_framework.test import APIClient
from unittest.mock import patch
from api.models import Billete
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RUTA_MODELO = os.path.join(BASE_DIR, "runs", "detect", "train11", "weights", "best.pt")
modelo_yolo = YOLO(RUTA_MODELO)

class BilleteTestCase(TestCase):
    def test_creacion_billete(self):
        billete = Billete.objects.create(imagen="test.jpg")
        self.assertEqual(str(billete), f"Billete {billete.id}")

class SubirImagenTestCase(TestCase):
    def setUp(self):
        self.client = APIClient()

    def test_subir_imagen_valida(self):
        ruta_imagen = r"C:\Users\Jesus\Documents\GitHub\TT\pruebas\prueba_billete.jpg"
        with open(ruta_imagen, "rb") as imagen:
            response = self.client.post('/api/subir-imagen/', {'imagen': imagen}, format='multipart')
        self.assertEqual(response.status_code, 201)
        self.assertIn("detecciones", response.data)

    def test_subir_imagen_vacia(self):
        response = self.client.post('/api/subir-imagen/', {}, format='multipart')
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.data)

    def test_inferencia_modelo_yolo(self):
        ruta_imagen = r"C:\Users\Jesus\Documents\GitHub\TT\pruebas\prueba_billete.jpg"
        img = cv2.imread(ruta_imagen)
        resultado = modelo_yolo.predict(img, conf=0.4)[0]
        self.assertTrue(len(resultado.boxes) > 0)

    @patch("api.views.guardar_imagen_en_azure")
    def test_subida_azure_mock(self, mock_subida):
        mock_subida.return_value = "https://fakeurl.com/imagen.png"
        ruta_imagen = r"C:\Users\Jesus\Documents\GitHub\TT\pruebas\prueba_billete.jpg"
        with open(ruta_imagen, "rb") as imagen:
            response = self.client.post('/api/subir-imagen/', {'imagen': imagen}, format='multipart')
        self.assertEqual(response.status_code, 201)
        self.assertIn("url_azure", response.data)
