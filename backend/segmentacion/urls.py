from django.urls import path
from .views import segmentar_imagen

urlpatterns = [
    path('segmentar/', segmentar_imagen, name='segmentar_imagen'),
]
