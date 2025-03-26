from rest_framework import serializers
from .models import Billete

class BilleteSerializer(serializers.ModelSerializer):
    class Meta:
        model = Billete
        fields = '__all__'
