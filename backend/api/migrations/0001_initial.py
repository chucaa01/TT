# Generated by Django 5.1.7 on 2025-03-18 17:48

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Billete',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('imagen', models.ImageField(upload_to='billetes/')),
                ('fecha_subida', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
