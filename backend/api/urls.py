from django.urls import path
from .views import subir_imagen

urlpatterns = [
    path('subir-imagen/', subir_imagen, name='subir_imagen'),
]
