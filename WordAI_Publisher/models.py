from django.db import models
from django.utils import timezone
from ckeditor.fields import RichTextField
from django.contrib import admin
from django.urls import path
from django.urls import reverse
from django.shortcuts import get_object_or_404, redirect
from django.http import JsonResponse
from django.utils.html import format_html
from django.contrib import messages
import re
from django.conf import settings
import requests
from django.core.files.base import ContentFile
import openai
import random
from django.db.models import JSONField
import json
import os
import threading

class ModelInfo(models.Model):
    model_id = models.CharField(max_length=255, unique=True)
    ethnicity = models.CharField(max_length=255, blank=True, null=True)
    skin_tone = models.CharField(max_length=255, blank=True, null=True)
    hair_texture = models.CharField(max_length=255, blank=True, null=True)
    face_shape = models.CharField(max_length=255, blank=True, null=True)
    tshirt = models.CharField(max_length=255, blank=True, null=True)
    eye_color = models.CharField(max_length=255, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        managed = False
        db_table = 'wordai_publisher_modelinfo'

class Prompt(models.Model):
    prompt_id = models.CharField(max_length=255, unique=True)
    master_prompt = models.TextField(blank=True, null=True)  # New field
    title_prompt = models.TextField(blank=True, null=True)
    intro_prompt = models.TextField(blank=True, null=True)
    style_prompt = models.TextField(blank=True, null=True)
    conclusion_prompt = models.TextField(blank=True, null=True)
    meta_data_prompt = models.TextField(blank=True, null=True)
    image_prompt = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.prompt_id}"

    class Meta:
        managed = False
        db_table = 'wordai_publisher_prompt'

class Keyword(models.Model):
    keyword = models.CharField(max_length=255, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.keyword

    class Meta:
        managed = False
        db_table = 'wordai_publisher_keyword'

class Post(models.Model):
    keyword = models.ForeignKey(Keyword, on_delete=models.CASCADE, related_name='posts')
    prompt = models.ForeignKey(Prompt, on_delete=models.SET_NULL, null=True, blank=True)
    model_info = models.ForeignKey(ModelInfo, on_delete=models.SET_NULL, null=True, blank=True)
    version = models.PositiveIntegerField(default=1)
    generated_title = models.TextField(blank=True, null=True)
    generated_intro = models.TextField(blank=True, null=True)
    generated_style_section = models.TextField(blank=True, null=True)
    generated_conclusion = models.TextField(blank=True, null=True)
    meta_title = models.TextField(blank=True, null=True)
    meta_description = models.TextField(blank=True, null=True)
    featured_prompt_text = models.TextField(null=True, blank=True)
    style_prompts = models.JSONField(null=True, blank=True)  # style name → prompt
    style_image_descriptions = models.JSONField(null=True, blank=True)  # New field

    STATUS_CHOICES = [
        ('draft', 'Draft'),
        ('pushed', 'Pushed'),
        # Add more statuses if needed
    ]
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='draft'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    featured_image = models.ImageField(upload_to='featured_images/', blank=True, null=True)
    style_images = JSONField(blank=True, null=True, default=dict)  # To store {style_name: image_url}
    content_generated = models.CharField(max_length=32, default='completed')
    FEATURED_IMAGE_STATUS_CHOICES = [
        ('not_generated', 'Not Generated'),
        ('in_process', 'In Process'),
        ('completed', 'Completed'),
    ]
    featured_image_status = models.CharField(max_length=16, choices=FEATURED_IMAGE_STATUS_CHOICES, default='not_generated')
    STYLE_IMAGES_STATUS_CHOICES = [
        ('not_generated', 'Not Generated'),
        ('in_process', 'In Process'),
        ('completed', 'Completed'),
    ]
    style_images_status = models.CharField(max_length=16, choices=STYLE_IMAGES_STATUS_CHOICES, default='not_generated')

    def __str__(self):
        return f"{self.keyword.keyword} - v{self.version} ({self.status})"

    def model_info_display(self):
        if self.model_info:
            return self.model_info.model_id
        return "-"
    model_info_display.short_description = 'Model ID'

    class Meta:
        managed = False
        db_table = 'wordai_publisher_post'

class KeywordAdmin(admin.ModelAdmin):
    def changelist_view(self, request, extra_context=None):
        if extra_context is None:
            extra_context = {}
        extra_context['all_prompts'] = Prompt.objects.all()
        return super().changelist_view(request, extra_context=extra_context)

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('ajax-generate-versions/', self.admin_site.admin_view(self.ajax_generate_versions), name='ajax_generate_versions'),
        ]
        return custom_urls + urls

def extract_styles_by_h2(style_section_html):
    pattern = r'<h2>(.*?)</h2>'
    return re.findall(pattern, style_section_html)
