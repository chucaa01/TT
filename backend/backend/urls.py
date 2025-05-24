from django.urls import path, include
from django.http import JsonResponse

urlpatterns = [
    path('', lambda request: JsonResponse({"message": "API de backend"}), name='api_root'),
    path('api/', include('api.urls')),
    path('segmentacion/', include('segmentacion.urls')),  # Nueva API
    path('*', lambda request: JsonResponse({"error": "Not found"}), name='not_found'),  # Manejo de rutas no encontradas
]