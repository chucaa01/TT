from django.urls import path, include

urlpatterns = [
    path('api/', include('api.urls')),
    path('segmentacion/', include('segmentacion.urls')),  # Nueva API

]
