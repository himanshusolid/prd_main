from django.urls import path
from . import views
from django.http import JsonResponse
from django.shortcuts import redirect, get_object_or_404
from django.contrib import messages
from .models import Post  # Adjust if your model is named differently

urlpatterns = [
    path('', views.upload_csv, name='upload_csv'),
]

