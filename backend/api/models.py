from django.db import models

class Billete(models.Model):
    imagen = models.ImageField(upload_to='billetes/')
    fecha_subida = models.DateTimeField(auto_now_add=True)

class CandadoBillete(models.Model):
    billete = models.ForeignKey(Billete, on_delete=models.CASCADE, related_name="candados")
    nombre = models.CharField(max_length=50)
    imagen = models.ImageField(upload_to='candados/')
