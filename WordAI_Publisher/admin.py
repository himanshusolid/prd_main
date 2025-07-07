from django.contrib import admin
from django.urls import path
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
import csv
from django.utils.html import format_html
from django.urls import reverse
from django.http import HttpResponse, JsonResponse
from django.template.response import TemplateResponse
import openai
from django.conf import settings
import requests
import markdown # Import the markdown library
from django import forms # Re-add forms import
from ckeditor_uploader.widgets import CKEditorUploadingWidget # Re-add CKEditor import
import base64
import json
import time
import os # Import os for path operations
from django.core.files.base import ContentFile # Import ContentFile
from django.core.files.storage import default_storage # Import default_storage
import re # Import re for regular expressions
import random
from django.db import models
from django.db.models import JSONField  # ✅ Works with MySQL (Django 3.1+)
from WordAI_Publisher.tasks import generate_post_images_task
import threading
from .models import Keyword, Prompt, ModelInfo, Post
from .admin_forms import CSVUploadForm, PostAdminForm
from django.utils.safestring import mark_safe
from django.views.decorators.csrf import csrf_exempt

# JWT token for WordPress authentication
# WORDPRESS_JWT_TOKEN should be set in your settings.py file.

@admin.register(ModelInfo)
class ModelInfoAdmin(admin.ModelAdmin):
    list_display = ('model_id', 'ethnicity', 'skin_tone', 'hair_texture', 'face_shape', 'created_at')
    change_list_template = "admin/WordAI_Publisher/modelinfo_changelist.html"

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('upload-csv/', self.admin_site.admin_view(self.upload_csv))
        ]
        return custom_urls + urls

    def upload_csv(self, request):
        context = self.admin_site.each_context(request)
        if request.method == "POST":
            form = CSVUploadForm(request.POST, request.FILES)
            if form.is_valid():
                csv_file = form.cleaned_data['csv_file']
                try:
                    decoded_file = csv_file.read().decode('utf-8').splitlines()
                except UnicodeDecodeError:
                    csv_file.seek(0)
                    decoded_file = csv_file.read().decode('latin-1').splitlines()
                reader = csv.DictReader(decoded_file)
                for row in reader:
                    ModelInfo.objects.create(
                        model_id=row.get('model_id', ''),
                        ethnicity=row.get('ethnicity', ''),
                        skin_tone=row.get('skin_tone', ''),
                        hair_texture=row.get('hair_texture', ''),
                        face_shape=row.get('face_shape', '')
                    )
                self.message_user(request, "CSV uploaded successfully!", level=messages.SUCCESS)
                return redirect("..")
        else:
            form = CSVUploadForm()
        context.update({
            'form': form,
            'opts': self.model._meta,
            'app_label': self.model._meta.app_label,
            'title': 'Upload CSV',
        })
        return render(request, "admin/WordAI_Publisher/upload_csv.html", context)

@admin.register(Prompt)
class PromptAdmin(admin.ModelAdmin):
    list_display = ('prompt_id', 'title_prompt', 'intro_prompt', 'created_at')
    search_fields = ('prompt_id', 'title_prompt', 'image_prompt')
    list_filter = ('created_at',)
    fieldsets = (
        ('Basic Information', {
            'fields': ('prompt_id',)
        }),
        ('Prompts', {
            'fields': ('title_prompt', 'intro_prompt', 'style_prompt', 'conclusion_prompt', 'meta_data_prompt', 'image_prompt'),
            'classes': ('wide',)
        }),
        ('Timestamps', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )
    readonly_fields = ('created_at',)

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('upload-csv/', self.admin_site.admin_view(self.upload_csv))
        ]
        return custom_urls + urls

    def upload_csv(self, request):
        context = self.admin_site.each_context(request)
        if request.method == "POST":
            form = CSVUploadForm(request.POST, request.FILES)
            if form.is_valid():
                csv_file = form.cleaned_data['csv_file']
                try:
                    decoded_file = csv_file.read().decode('utf-8').splitlines()
                except UnicodeDecodeError:
                    csv_file.seek(0)
                    decoded_file = csv_file.read().decode('latin-1').splitlines()
                reader = csv.DictReader(decoded_file)
                for row in reader:
                    Prompt.objects.create(
                        prompt_id=row.get('prompt_id', ''),
                        title_prompt=row.get('title_prompt', ''),
                        intro_prompt=row.get('intro_prompt', ''),
                        style_prompt=row.get('style_prompt', ''),
                        conclusion_prompt=row.get('conclusion_prompt', ''),
                        meta_data_prompt=row.get('meta_data_prompt', ''),
                        image_prompt=row.get('image_prompt', '')
                    )
                self.message_user(request, "CSV uploaded successfully!", level=messages.SUCCESS)
                return redirect("..")
        else:
            form = CSVUploadForm()
        context.update({
            'form': form,
            'opts': self.model._meta,
            'app_label': self.model._meta.app_label,
            'title': 'Upload CSV',
        })
        return render(request, "admin/WordAI_Publisher/upload_csv.html", context)

class KeywordAdminForm(forms.ModelForm):
    class Meta:
        model = Keyword
        fields = '__all__'

@admin.register(Keyword)
class KeywordAdmin(admin.ModelAdmin):
    form = KeywordAdminForm # Re-add the form
    list_display = ('keyword', 'created_at', 'generate_content_button')
    change_list_template = "admin/WordAI_Publisher/keyword_changelist.html"
    actions = ['generate_content_for_selected']

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('upload-csv/', self.admin_site.admin_view(self.upload_csv)),
            path('generate-content/<int:pk>/', self.admin_site.admin_view(self.generate_content_view), name='generate_content'),
            path('ajax-regenerate-versions/', self.admin_site.admin_view(self.ajax_regenerate_versions), name='ajax_regenerate_versions'),
            path('ajax-generate-versions/', self.admin_site.admin_view(self.ajax_generate_versions), name='ajax_generate_versions'),
            path('ajax-regenerate-single-style/', self.admin_site.admin_view(self.ajax_regenerate_single_style), name='ajax_regenerate_single_style'),  # ✅ ADD THIS
        ]
        return custom_urls + urls
    
    def ajax_regenerate_single_style(self, request):
        if request.method == 'POST':
            data = json.loads(request.body.decode())
            post_id = data.get('post_id')
            style_name = data.get('style_name')
            
            post = get_object_or_404(Post, pk=post_id)
            post.style_images_status = 'in_process'
            post.save()

            # start generation just for this style name
            threading.Thread(
                target=generate_post_images_task,
                args=(post.id,),
                kwargs={'only_style': True, 'specific_style': style_name}
            ).start()

            # threading.Thread(
            #     target=generate_post_images_task,
            #     args=(post.id,),
            #     kwargs={'only_style': True}
            # ).start()


            return JsonResponse({'success': True, 'message': f'Style "{style_name}" regeneration started.'})
        return JsonResponse({'success': False, 'message': 'Invalid request.'})
        
    def generate_content_view(self, request, pk):
        keyword = get_object_or_404(Keyword, pk=pk)
        prompt_obj = Prompt.objects.filter(prompt_id=keyword.prompt_id).first()
        model_info = ModelInfo.objects.filter(model_id=keyword.model_id).first()

        # Define prompt templates for each section (these will now be fallback if prompt_obj fields are empty)
        title_prompt_template = prompt_obj.title_prompt if prompt_obj and prompt_obj.title_prompt else ""
        intro_prompt_template = prompt_obj.intro_prompt if prompt_obj and prompt_obj.intro_prompt else ""
        style_section_prompt_template = prompt_obj.style_prompt if prompt_obj and prompt_obj.style_prompt else ""
        conclusion_prompt_template = prompt_obj.conclusion_prompt if prompt_obj and prompt_obj.conclusion_prompt else ""
        meta_data_prompt_template = prompt_obj.meta_data_prompt if prompt_obj and prompt_obj.meta_data_prompt else ""

        # Replace keyword placeholder in prompts (using the new prompt variables)
        title_prompt = title_prompt_template
        if title_prompt and keyword.keyword:
            title_prompt = title_prompt.replace('{{keyword}}', keyword.keyword)
            title_prompt = title_prompt.replace('{{hairstyle_name}}', keyword.keyword)
        
        intro_prompt = intro_prompt_template
        if intro_prompt and keyword.keyword:
            intro_prompt = intro_prompt.replace('{{keyword}}', keyword.keyword)
            intro_prompt = intro_prompt.replace('{{hairstyle_name}}', keyword.keyword)

        style_section_prompt = style_section_prompt_template
        if style_section_prompt and keyword.keyword:
            style_section_prompt = style_section_prompt.replace('{{keyword}}', keyword.keyword)
            style_section_prompt = style_section_prompt.replace('{{hairstyle_name}}', keyword.keyword)

        conclusion_prompt = conclusion_prompt_template
        if conclusion_prompt and keyword.keyword:
            conclusion_prompt = conclusion_prompt.replace('{{keyword}}', keyword.keyword)
            conclusion_prompt = conclusion_prompt.replace('{{hairstyle_name}}', keyword.keyword)

        meta_data_prompt = meta_data_prompt_template
        if meta_data_prompt and keyword.keyword:
            meta_data_prompt = meta_data_prompt.replace('{{keyword}}', keyword.keyword)
            meta_data_prompt = meta_data_prompt.replace('{{hairstyle_name}}', keyword.keyword)

        image_prompt = prompt_obj.image_prompt if prompt_obj else 'N/A'
        if image_prompt and keyword.keyword:
            image_prompt = image_prompt.replace('{{hairstyle_name}}', keyword.keyword)
        
        # Replace placeholders in image_prompt with model_info attributes
        if image_prompt and model_info:
            image_prompt = image_prompt.replace('{{ethnicity}}', model_info.ethnicity if model_info.ethnicity else '')
            image_prompt = image_prompt.replace('{{skin_tone}}', model_info.skin_tone if model_info.skin_tone else '')
            image_prompt = image_prompt.replace('{{hair_texture}}', model_info.hair_texture if model_info.hair_texture else '')
            image_prompt = image_prompt.replace('{{face_shape}}', model_info.face_shape if model_info.face_shape else '')

        wordpress_post_status = None

        if request.method == "POST":
            action = request.POST.get("action")
            content_type = request.POST.get("content_type") # Get the content type from the request

            if action == "generate_text":
                client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
                
                # Determine which prompt to use based on content_type
                if content_type == 'title':
                    full_prompt = title_prompt
                    field_to_update = 'generated_title'
                    system_prompt = "You are a helpful assistant that generates SEO-optimized article titles. Format the title as a single H1 heading using markdown (e.g., '# Title Here')."
                elif content_type == 'intro':
                    full_prompt = intro_prompt
                    field_to_update = 'generated_intro'
                    system_prompt = "You are a helpful assistant that generates engaging article introductions. Start with an H2 heading using markdown (e.g., '## Introduction')."
                elif content_type == 'style_section':
                    full_prompt = style_section_prompt
                    field_to_update = 'generated_style_section'
                    system_prompt = "You are a helpful assistant that generates detailed style descriptions. Use H2 headings for each style section using markdown (e.g., '## Style Name')."
                elif content_type == 'conclusion':
                    full_prompt = conclusion_prompt
                    field_to_update = 'generated_conclusion'
                    system_prompt = "You are a helpful assistant that generates article conclusions. Start with an H2 heading using markdown (e.g., '## Conclusion')."
                elif content_type == 'meta_seo':
                    full_prompt = meta_data_prompt
                    system_prompt = "You are an SEO assistant that generates concise meta titles and descriptions in JSON format."
                else:
                    return JsonResponse({
                        'success': False,
                        'message': 'Invalid content type provided.'
                    })

                try:
                    if content_type == 'meta_seo':
                        # Special handling for meta_seo as it returns JSON
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": full_prompt}
                            ]
                        )
                        raw_meta_output = response.choices[0].message.content
                        print(f"Raw Meta Data API Response: {raw_meta_output}")
                        try:
                            meta_data = json.loads(raw_meta_output)
                            meta_title = meta_data.get('meta_title', '').replace('\\"', '"').strip('"')
                            print(f"DEBUG (after get and strip): meta_title = {meta_title}")
                            meta_description = meta_data.get('meta_description', '').replace('\\"', '"').strip('"')
                            print(f"DEBUG (after get and strip): meta_description = {meta_description}")

                            if keyword.keyword:
                                meta_title = meta_title.replace('{{hairstyle_name}}', keyword.keyword)
                                meta_description = meta_description.replace('{{hairstyle_name}}', keyword.keyword)
                                print(f"DEBUG (after replacement): meta_title = {meta_title}")
                                print(f"DEBUG (after replacement): meta_description = {meta_description}")

                        except json.JSONDecodeError:
                            # Fallback if AI doesn't return perfect JSON
                            meta_title = "" # Could parse from raw_meta_output if a pattern is consistent
                            meta_description = raw_meta_output # Store raw output if parsing fails
                            print("JSON Decode Error for Meta Data. Storing raw output.")

                        keyword.meta_title = meta_title
                        keyword.meta_description = meta_description
                        keyword.save()

                        print(f"Sending to frontend - Meta Title: {meta_title}, Meta Description: {meta_description}")
                        return JsonResponse({
                            'success': True,
                            'message': 'Meta Title and Description generated successfully!',
                            'meta_title': meta_title,
                            'meta_description': meta_description,
                        })
                    else:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": full_prompt}
                            ]
                        )
                        raw_generated_content = response.choices[0].message.content
                        print(f"Raw generated content from API for {content_type}: {raw_generated_content}")
                        # Strip quotes from generated content
                        generated_content = raw_generated_content.replace('\\"', '"').strip('"')
                        print(f"Processed generated content for {content_type} after strip: {generated_content}")
                        # Convert markdown to HTML before saving and sending to frontend
                        html_content = markdown.markdown(generated_content)
                        setattr(keyword, field_to_update, html_content) # Update the specific field
                        keyword.save()
                        return JsonResponse({
                            "success": True,
                            "message": f"{content_type.replace('_', ' ').title()} generated successfully!",
                            "content": html_content # Send back HTML content
                        })
                except openai.OpenAIError as e:
                    return JsonResponse({
                        'success': False,
                        'message': f'ChatGPT API Error: {e}'
                    })
                except requests.exceptions.RequestException as e:
                    return JsonResponse({
                        'success': False,
                        'message': f'Network/WordPress API Error: {e}'
                    })

            elif action == "generate_image":
                client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
                try:
                    response = client.images.generate(
                        model="dall-e-2",
                        prompt=image_prompt,
                        size="256x256", # Changed to smaller size as per user request
                        n=1,
                    )
                    image_url = response.data[0].url
                    keyword.generated_image_url = image_url
                    keyword.save()
                    return JsonResponse({
                        'success': True,
                        'message': 'Image generated successfully!',
                        'image_url': image_url
                    })
                except openai.OpenAIError as e:
                    return JsonResponse({
                        'success': False,
                        'message': f'DALL-E API Error: {e}'
                    })

            elif action == "push_to_wordpress":
                # Get content from all individual CKEditor instances
                title_content = request.POST.get('title_content', '')
                intro_content = request.POST.get('intro_content', '')
                style_section_content = request.POST.get('style_section_content', '')
                conclusion_content = request.POST.get('conclusion_content', '')

                # Strip HTML tags from the title content
                title_content_plain = re.sub(r'<[^>]*>', '', title_content).strip()

                print(f"DEBUG: title_content: {title_content}")
                print(f"DEBUG: intro_content: {intro_content}")
                print(f"DEBUG: style_section_content: {style_section_content}")
                print(f"DEBUG: conclusion_content: {conclusion_content}")

                # Combine all content into a single post content, excluding meta for now
                combined_content = f"{intro_content}\n\n{style_section_content}\n\n{conclusion_content}"

                # WordPress API setup
                wordpress_api_url = settings.WORDPRESS_API_URL
                
                # JWT token for WordPress authentication
                # Ensure WORDPRESS_JWT_TOKEN is set in your settings.py
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {settings.WORDPRESS_JWT_TOKEN}"
                }
                
                # Data for the WordPress post
                post_data = {
                    "title": title_content_plain, # Use the plain text title here
                    "content": combined_content, # All generated content in the main editor
                    "status": "publish",
                }
                
                print(f"DEBUG: post_data dictionary: {post_data}")
                print(f"DEBUG: JSON payload sent: {json.dumps(post_data)}")
                print("Attempting to post to WordPress with data:", post_data)
                print("Headers:", headers)

                try:
                    # Extract base URL from WORDPRESS_API_URL for user check
                    base_wordpress_url = settings.WORDPRESS_API_URL.split('/wp-json/')[0]
                    user_check_url = f"{base_wordpress_url}/wp-json/wp/v2/users/me"
                    user_response = requests.get(user_check_url, headers=headers)
                    if user_response.status_code == 200:
                        print("WordPress user verified successfully:", user_response.json().get('name'))
                    else:
                        print(f"WordPress user verification failed. Status: {user_response.status_code}, Response: {user_response.text}")
                        return JsonResponse({
                            'success': False,
                            'message': f'WordPress user verification failed. Status: {user_response.status_code}, Error: {user_response.json().get("message", user_response.text)}'
                        })

                    response = requests.post(wordpress_api_url, headers=headers, json=post_data)
                    print("WordPress API Response Status:", response.status_code)
                    print("WordPress API Response Body:", response.text)

                    if response.status_code == 201: # 201 Created for successful post
                        return JsonResponse({
                            'success': True,
                            'message': 'Content pushed to WordPress as draft successfully!'
                        })
                    else:
                        error_message = response.json().get('message', response.text) if response.json() else response.text
                        return JsonResponse({
                            'success': False,
                            'message': f'Failed to push content to WordPress. Status: {response.status_code}, Error: {error_message}'
                        })
                except requests.exceptions.RequestException as e:
                    return JsonResponse({
                        'success': False,
                        'message': f'Network/WordPress API Error: {e}'
                    })

            elif action == "generate_meta_seo": # This action is now deprecated, handled by generate_text
                # The logic for meta_seo generation is now integrated into "generate_text" action.
                return JsonResponse({
                    'success': False,
                    'message': "This action is deprecated. Use generate_text with content_type='meta_seo' instead."
                })

            elif action == "generate_all_content":
                client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
                # Generate all content sections using their respective prompts
                all_generated_data = {}
                content_types = ['title', 'intro', 'style_section', 'conclusion', 'meta_seo']
                prompts_map = {
                    'title': title_prompt,
                    'intro': intro_prompt,
                    'style_section': style_section_prompt,
                    'conclusion': conclusion_prompt,
                    'meta_seo': meta_data_prompt,
                }
                system_prompts_map = {
                    'title': "You are a helpful assistant that generates SEO-optimized article titles. Format the title as a single H1 heading using markdown (e.g., '# Title Here').",
                    'intro': "You are a helpful assistant that generates engaging article introductions. Start with an H2 heading using markdown (e.g., '## Introduction').",
                    'style_section': "You are a helpful assistant that generates detailed style descriptions. Use H2 headings for each style section using markdown (e.g., '## Style Name').",
                    'conclusion': "You are a helpful assistant that generates article conclusions. Start with an H2 heading using markdown (e.g., '## Conclusion').",
                    'meta_seo': "You are an SEO assistant that generates concise meta titles and descriptions in JSON format.",
                }

                for c_type in content_types:
                    current_prompt = prompts_map[c_type]
                    current_system_prompt = system_prompts_map[c_type]
                    try:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": current_system_prompt},
                                {"role": "user", "content": current_prompt}
                            ]
                        )
                        generated_content = response.choices[0].message.content
                        
                        if c_type == 'meta_seo':
                            try:
                                meta_data = json.loads(generated_content)
                                meta_title = meta_data.get('meta_title', '').replace('\\"', '"').strip('"')
                                meta_description = meta_data.get('meta_description', '').replace('\\"', '"').strip('"')
                                keyword.meta_title = meta_title
                                keyword.meta_description = meta_description
                                all_generated_data['meta_title'] = meta_title
                                all_generated_data['meta_description'] = meta_description
                            except json.JSONDecodeError:
                                print(f"JSON Decode Error for Meta Data in generate_all_content. Raw output: {generated_content}")
                                # Store raw output if parsing fails
                                all_generated_data['meta_title'] = "Error generating meta title"
                                all_generated_data['meta_description'] = generated_content
                        else:
                            # Strip quotes from generated content
                            generated_content = generated_content.replace('\\"', '"').strip('"')
                            html_content = markdown.markdown(generated_content)
                            setattr(keyword, f'generated_{c_type}', html_content)
                            all_generated_data[c_type] = html_content

                    except openai.OpenAIError as e:
                        print(f"Error generating {c_type} content: {e}")
                        all_generated_data[c_type] = f"Error: {e}"

                keyword.save()
                return JsonResponse({
                    "success": True,
                    "message": "All content sections generated successfully!",
                    "meta_title": all_generated_data.get('meta_title', ''),
                    "meta_description": all_generated_data.get('meta_description', ''),
                    "title": all_generated_data.get('title', ''),
                    "intro": all_generated_data.get('intro', ''),
                    "style_section": all_generated_data.get('style_section', ''),
                    "conclusion": all_generated_data.get('conclusion', ''),
                })

        print(f"Meta Data Prompt being passed to template: {meta_data_prompt}")
        form = KeywordAdminForm(instance=keyword)
        
        context = self.admin_site.each_context(request)
        context.update({
            'opts': self.model._meta,
            'app_label': self.model._meta.app_label,
            'original': keyword, # Pass the keyword object as 'original'
            'title': f'Generate Content for: {keyword.keyword}',
            'image_prompt': prompt_obj.image_prompt if prompt_obj else 'N/A',
            'generated_image_url': keyword.generated_image_url,
            'wordpress_post_status': wordpress_post_status,
            'meta_title': keyword.meta_title,
            'meta_description': keyword.meta_description,
            # Pass individual content fields to the template
            'generated_title': keyword.generated_title,
            'generated_intro': keyword.generated_intro,
            'generated_style_section': keyword.generated_style_section,
            'generated_conclusion': keyword.generated_conclusion,
            # Pass individual prompt templates to the template
            'title_prompt': title_prompt,
            'intro_prompt': intro_prompt,
            'style_section_prompt': style_section_prompt,
            'conclusion_prompt': conclusion_prompt,
            'form': form, # Pass the form instance
        })
        return TemplateResponse(request, "admin/WordAI_Publisher/generated_content.html", context)
    
    def upload_csv(self, request):
        context = self.admin_site.each_context(request)
        if request.method == "POST":
            form = CSVUploadForm(request.POST, request.FILES)
            if form.is_valid():
                csv_file = form.cleaned_data['csv_file']
                try:
                    decoded_file = csv_file.read().decode('utf-8').splitlines()
                except UnicodeDecodeError:
                    csv_file.seek(0)
                    decoded_file = csv_file.read().decode('latin-1').splitlines()
                reader = csv.DictReader(decoded_file)
                for row in reader:
                    Keyword.objects.create(
                        keyword=row.get('keyword', '')
                    )
                self.message_user(request, "CSV uploaded successfully!", level=messages.SUCCESS)
                return redirect("..")
        else:
            form = CSVUploadForm()
        context.update({
            'form': form,
            'opts': self.model._meta,
            'app_label': self.model._meta.app_label,
            'title': 'Upload CSV',
        })
        return render(request, "admin/WordAI_Publisher/upload_csv.html", context)

        keyword = get_object_or_404(Keyword, pk=pk)
        prompt_obj = Prompt.objects.filter(prompt_id=keyword.prompt_id).first()
        model_info = ModelInfo.objects.filter(model_id=keyword.model_id).first()

        # Define prompt templates for each section (these will now be fallback if prompt_obj fields are empty)
        title_prompt_template = prompt_obj.title_prompt if prompt_obj and prompt_obj.title_prompt else ""
        intro_prompt_template = prompt_obj.intro_prompt if prompt_obj and prompt_obj.intro_prompt else ""
        style_section_prompt_template = prompt_obj.style_prompt if prompt_obj and prompt_obj.style_prompt else ""
        conclusion_prompt_template = prompt_obj.conclusion_prompt if prompt_obj and prompt_obj.conclusion_prompt else ""
        meta_data_prompt_template = prompt_obj.meta_data_prompt if prompt_obj and prompt_obj.meta_data_prompt else ""

        # Replace keyword placeholder in prompts (using the new prompt variables)
        title_prompt = title_prompt_template
        if title_prompt and keyword.keyword:
            title_prompt = title_prompt.replace('{{keyword}}', keyword.keyword)
            title_prompt = title_prompt.replace('{{hairstyle_name}}', keyword.keyword)
        
        intro_prompt = intro_prompt_template
        if intro_prompt and keyword.keyword:
            intro_prompt = intro_prompt.replace('{{keyword}}', keyword.keyword)
            intro_prompt = intro_prompt.replace('{{hairstyle_name}}', keyword.keyword)

        style_section_prompt = style_section_prompt_template
        if style_section_prompt and keyword.keyword:
            style_section_prompt = style_section_prompt.replace('{{keyword}}', keyword.keyword)
            style_section_prompt = style_section_prompt.replace('{{hairstyle_name}}', keyword.keyword)

        conclusion_prompt = conclusion_prompt_template
        if conclusion_prompt and keyword.keyword:
            conclusion_prompt = conclusion_prompt.replace('{{keyword}}', keyword.keyword)
            conclusion_prompt = conclusion_prompt.replace('{{hairstyle_name}}', keyword.keyword)

        meta_data_prompt = meta_data_prompt_template
        if meta_data_prompt and keyword.keyword:
            meta_data_prompt = meta_data_prompt.replace('{{keyword}}', keyword.keyword)
            meta_data_prompt = meta_data_prompt.replace('{{hairstyle_name}}', keyword.keyword)

        image_prompt = prompt_obj.image_prompt if prompt_obj else 'N/A'
        if image_prompt and keyword.keyword:
            image_prompt = image_prompt.replace('{{hairstyle_name}}', keyword.keyword)
        
        # Replace placeholders in image_prompt with model_info attributes
        if image_prompt and model_info:
            image_prompt = image_prompt.replace('{{ethnicity}}', model_info.ethnicity if model_info.ethnicity else '')
            image_prompt = image_prompt.replace('{{skin_tone}}', model_info.skin_tone if model_info.skin_tone else '')
            image_prompt = image_prompt.replace('{{hair_texture}}', model_info.hair_texture if model_info.hair_texture else '')
            image_prompt = image_prompt.replace('{{face_shape}}', model_info.face_shape if model_info.face_shape else '')

        wordpress_post_status = None

        if request.method == "POST":
            action = request.POST.get("action")
            post_id = request.POST.get("post_id")

            if action == "push_post_to_wordpress" and post_id:
                from .models import Post
                post = get_object_or_404(Post, pk=post_id)
                # Strip HTML tags from the title content
                title_content_plain = re.sub(r'<[^>]*>', '', post.generated_title or '').strip()
                combined_content = f"{post.generated_intro or ''}\n\n{post.generated_style_section or ''}\n\n{post.generated_conclusion or ''}"
                wordpress_api_url = settings.WORDPRESS_API_URL
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {settings.WORDPRESS_JWT_TOKEN}"
                }
                post_data = {
                    "title": title_content_plain,
                    "content": combined_content,
                    "status": "publish",
                }
                try:
                    response = requests.post(wordpress_api_url, headers=headers, json=post_data)
                    if response.status_code == 201:
                        post.status = 'pushed'
                        post.save()
                        return JsonResponse({
                            'success': True,
                            'message': 'Version pushed to WordPress successfully!'
                        })
                    else:
                        error_message = response.json().get('message', response.text) if response.json() else response.text
                        return JsonResponse({
                            'success': False,
                            'message': f'Failed to push version to WordPress. Status: {response.status_code}, Error: {error_message}'
                        })
                except requests.exceptions.RequestException as e:
                    return JsonResponse({
                        'success': False,
                        'message': f'Network/WordPress API Error: {e}'
                    })

            elif action == "generate_text":
                client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
                
                # Determine which prompt to use based on content_type
                if content_type == 'title':
                    full_prompt = title_prompt
                    field_to_update = 'generated_title'
                    system_prompt = "You are a helpful assistant that generates SEO-optimized article titles. Write the title as plain text, no markdown, no HTML tags."
                elif content_type == 'intro':
                    full_prompt = intro_prompt
                    field_to_update = 'generated_intro'
                    system_prompt = "You are a helpful assistant that generates engaging article introductions. Start with an <h2>Introduction</h2> followed by a well-spaced paragraph. Do not use markdown."
                elif content_type == 'style_section':
                    full_prompt = style_section_prompt
                    field_to_update = 'generated_style_section'
                    system_prompt = "You are a helpful assistant that generates detailed style descriptions. Use <h2> HTML headings for each style section, followed by paragraphs. Do not use markdown."
                elif content_type == 'conclusion':
                    full_prompt = conclusion_prompt
                    field_to_update = 'generated_conclusion'
                    system_prompt = "You are a helpful assistant that generates article conclusions. Start with an <h2>Conclusion</h2> followed by a well-spaced paragraph. Do not use markdown."
                elif content_type == 'meta_seo':
                    full_prompt = meta_data_prompt
                    system_prompt = "You are an SEO assistant that generates concise meta titles and descriptions in JSON format."
                else:
                    return JsonResponse({
                        'success': False,
                        'message': 'Invalid content type provided.'
                    })

                try:
                    if content_type == 'meta_seo':
                        # Special handling for meta_seo as it returns JSON
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": full_prompt}
                            ]
                        )
                        raw_meta_output = response.choices[0].message.content
                        print(f"Raw Meta Data API Response: {raw_meta_output}")
                        try:
                            meta_data = json.loads(raw_meta_output)
                            meta_title = meta_data.get('meta_title', '').replace('\\"', '"').strip('"')
                            print(f"DEBUG (after get and strip): meta_title = {meta_title}")
                            meta_description = meta_data.get('meta_description', '').replace('\\"', '"').strip('"')
                            print(f"DEBUG (after get and strip): meta_description = {meta_description}")

                            if keyword.keyword:
                                meta_title = meta_title.replace('{{hairstyle_name}}', keyword.keyword)
                                meta_description = meta_description.replace('{{hairstyle_name}}', keyword.keyword)
                                print(f"DEBUG (after replacement): meta_title = {meta_title}")
                                print(f"DEBUG (after replacement): meta_description = {meta_description}")

                        except json.JSONDecodeError:
                            # Fallback if AI doesn't return perfect JSON
                            meta_title = "" # Could parse from raw_meta_output if a pattern is consistent
                            meta_description = raw_meta_output # Store raw output if parsing fails
                            print("JSON Decode Error for Meta Data. Storing raw output.")

                        keyword.meta_title = meta_title
                        keyword.meta_description = meta_description
                        keyword.save()

                        print(f"Sending to frontend - Meta Title: {meta_title}, Meta Description: {meta_description}")
                        return JsonResponse({
                            'success': True,
                            'message': 'Meta Title and Description generated successfully!',
                            'meta_title': meta_title,
                            'meta_description': meta_description,
                        })
                    else:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": full_prompt}
                            ]
                        )
                        raw_generated_content = response.choices[0].message.content
                        print(f"Raw generated content from API for {content_type}: {raw_generated_content}")
                        # Strip quotes from generated content
                        generated_content = raw_generated_content.replace('\\"', '"').strip('"')
                        print(f"Processed generated content for {content_type} after strip: {generated_content}")
                        # Convert markdown to HTML before saving and sending to frontend
                        html_content = markdown.markdown(generated_content)
                        setattr(keyword, field_to_update, html_content) # Update the specific field
                        keyword.save()
                        return JsonResponse({
                            "success": True,
                            "message": f"{content_type.replace('_', ' ').title()} generated successfully!",
                            "content": html_content # Send back HTML content
                        })
                except openai.OpenAIError as e:
                    return JsonResponse({
                        'success': False,
                        'message': f'ChatGPT API Error: {e}'
                    })
                except requests.exceptions.RequestException as e:
                    return JsonResponse({
                        'success': False,
                        'message': f'Network/WordPress API Error: {e}'
                    })

            elif action == "generate_image":
                client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
                try:
                    image_response = client.images.generate(
                        model="dall-e-3",
                        prompt=image_prompt,
                        size="1024x1024",
                        n=1,
                    )
                    image_url = image_response.data[0].url
                    keyword.generated_image_url = image_url
                    keyword.save()
                    return JsonResponse({
                        'success': True,
                        'message': 'Image generated successfully!',
                        'image_url': image_url
                    })
                except openai.OpenAIError as e:
                    return JsonResponse({
                        'success': False,
                        'message': f'DALL-E API Error: {e}'
                    })

            elif action == "generate_meta_seo": # This action is now deprecated, handled by generate_text
                # The logic for meta_seo generation is now integrated into "generate_text" action.
                return JsonResponse({
                    'success': False,
                    'message': "This action is deprecated. Use generate_text with content_type='meta_seo' instead."
                })

            elif action == "generate_all_content":
                client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
                # Generate all content sections using their respective prompts
                all_generated_data = {}
                content_types = ['title', 'intro', 'style_section', 'conclusion', 'meta_seo']
                prompts_map = {
                    'title': title_prompt,
                    'intro': intro_prompt,
                    'style_section': style_section_prompt,
                    'conclusion': conclusion_prompt,
                    'meta_seo': meta_data_prompt,
                }
                system_prompts_map = {
                    'title': "You are a helpful assistant that generates SEO-optimized article titles. Write the title as plain text, no markdown, no HTML tags.",
                    'intro': "You are a helpful assistant that generates engaging article introductions. Start with an <h2>Introduction</h2> followed by a well-spaced paragraph. Do not use markdown.",
                    'style_section': "You are a helpful assistant that generates detailed style descriptions. Use <h2> HTML headings for each style section, followed by paragraphs. Do not use markdown.",
                    'conclusion': "You are a helpful assistant that generates article conclusions. Start with an <h2>Conclusion</h2> followed by a well-spaced paragraph. Do not use markdown.",
                    'meta_seo': "You are an SEO assistant that generates concise meta titles and descriptions in JSON format.",
                }

                for c_type in content_types:
                    current_prompt = prompts_map[c_type]
                    current_system_prompt = system_prompts_map[c_type]
                    try:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": current_system_prompt},
                                {"role": "user", "content": current_prompt}
                            ]
                        )
                        generated_content = response.choices[0].message.content
                        
                        if c_type == 'meta_seo':
                            try:
                                meta_data = json.loads(generated_content)
                                meta_title = meta_data.get('meta_title', '').replace('\\"', '"').strip('"')
                                meta_description = meta_data.get('meta_description', '').replace('\\"', '"').strip('"')
                                keyword.meta_title = meta_title
                                keyword.meta_description = meta_description
                                all_generated_data['meta_title'] = meta_title
                                all_generated_data['meta_description'] = meta_description
                            except json.JSONDecodeError:
                                print(f"JSON Decode Error for Meta Data in generate_all_content. Raw output: {generated_content}")
                                # Store raw output if parsing fails
                                all_generated_data['meta_title'] = "Error generating meta title"
                                all_generated_data['meta_description'] = generated_content
                        else:
                            # Strip quotes from generated content
                            generated_content = generated_content.replace('\\"', '"').strip('"')
                            html_content = markdown.markdown(generated_content)
                            setattr(keyword, f'generated_{c_type}', html_content)
                            all_generated_data[c_type] = html_content

                    except openai.OpenAIError as e:
                        print(f"Error generating {c_type} content: {e}")
                        all_generated_data[c_type] = f"Error: {e}"

                keyword.save()
                return JsonResponse({
                    "success": True,
                    "message": "All content sections generated successfully!",
                    "meta_title": all_generated_data.get('meta_title', ''),
                    "meta_description": all_generated_data.get('meta_description', ''),
                    "title": all_generated_data.get('title', ''),
                    "intro": all_generated_data.get('intro', ''),
                    "style_section": all_generated_data.get('style_section', ''),
                    "conclusion": all_generated_data.get('conclusion', ''),
                })

        print(f"Meta Data Prompt being passed to template: {meta_data_prompt}")
        form = KeywordAdminForm(instance=keyword)
        
        context = self.admin_site.each_context(request)
        context.update({
            'opts': self.model._meta,
            'app_label': self.model._meta.app_label,
            'original': keyword, # Pass the keyword object as 'original'
            'title': f'Generate Content for: {keyword.keyword}',
            'image_prompt': prompt_obj.image_prompt if prompt_obj else 'N/A',
            'generated_image_url': keyword.generated_image_url,
            'wordpress_post_status': wordpress_post_status,
            'meta_title': keyword.meta_title,
            'meta_description': keyword.meta_description,
            # Pass individual content fields to the template
            'generated_title': keyword.generated_title,
            'generated_intro': keyword.generated_intro,
            'generated_style_section': keyword.generated_style_section,
            'generated_conclusion': keyword.generated_conclusion,
            # Pass individual prompt templates to the template
            'title_prompt': title_prompt,
            'intro_prompt': intro_prompt,
            'style_section_prompt': style_section_prompt,
            'conclusion_prompt': conclusion_prompt,
            'form': form, # Pass the form instance
            # Fetch all Post versions for this keyword
            'post_versions': Post.objects.filter(keyword=keyword).order_by('version'),
        })
        return TemplateResponse(request, "admin/WordAI_Publisher/generated_content.html", context)

    def generate_content_button(self, obj):
        return format_html(
            '<a class="button" href="{}">Generate Content</a>&nbsp;',
            reverse('admin:generate_content', args=[obj.pk])
        )
    generate_content_button.short_description = "Actions"
    generate_content_button.allow_tags = True
    
    def ajax_generate_versions(self, request):
        if request.method == 'POST':
            keyword_id = request.POST.get('keyword_id')
            prompt_id = request.POST.get('prompt_id')
            version_count = int(request.POST.get('version_count', 1))
            keyword = get_object_or_404(Keyword, pk=keyword_id)
            prompt = get_object_or_404(Prompt, prompt_id=prompt_id)
            models = list(ModelInfo.objects.all())
            if not models:
                return JsonResponse({'success': False, 'message': 'No models available.'})
            client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
            model_info = random.choice(models)

            # Prepare prompts (replace placeholders)
            def fill_prompt(tmpl):
                if tmpl:
                    return tmpl.replace('{{keyword}}', keyword.keyword).replace('{{hairstyle_name}}', keyword.keyword)
                return ''
            title_prompt = fill_prompt(prompt.title_prompt)
            intro_prompt = fill_prompt(prompt.intro_prompt)
            style_prompt = fill_prompt(prompt.style_prompt)
            conclusion_prompt = fill_prompt(prompt.conclusion_prompt)
            meta_data_prompt = fill_prompt(prompt.meta_data_prompt)

            # Generate content for each section (synchronously)
            def gpt_content(system_prompt, user_prompt):
                if not user_prompt:
                    return ''
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return response.choices[0].message.content

            generated_title = gpt_content("", title_prompt)
            if generated_title:
                generated_title = generated_title.strip().strip('"').strip("'")
            generated_intro = gpt_content("You are a helpful assistant that generates engaging article introductions. Do not use markdown.", intro_prompt)
            generated_style_section = gpt_content(
                f"You are a helpful assistant that generates detailed style descriptions. Use <h2> HTML headings for each style section, followed by paragraphs. Do not use markdown. Generate exactly {version_count} unique hairstyles.",
                style_prompt
            )
            generated_conclusion = gpt_content("You are a helpful assistant that generates article conclusions. Do not use markdown.", conclusion_prompt)
            meta_title = ''
            meta_description = ''
            if meta_data_prompt:
                meta_json = gpt_content("You are an SEO assistant that generates concise meta titles and descriptions in JSON format.", meta_data_prompt)
                try:
                    meta_data = json.loads(meta_json)
                    meta_title = meta_data.get('meta_title', '')
                    meta_description = meta_data.get('meta_description', '')
                except Exception:
                    meta_title = ''
                    meta_description = meta_json

            # Save as Post (without images for now)
            post = Post.objects.create(
                keyword=keyword,
                prompt=prompt,
                model_info=model_info,
                version=1,
                generated_title=generated_title,
                generated_intro=generated_intro,
                generated_style_section=generated_style_section,
                generated_conclusion=generated_conclusion,
                meta_title=meta_title,
                meta_description=meta_description,
                status='draft',
                content_generated='completed',
                featured_image_status='in_process',
                style_images_status='in_process',
            )
            post.save()

            # Start image generation in a background thread (no Celery)
            threading.Thread(target=generate_post_images_task, args=(post.id,)).start()

            return JsonResponse({
                'success': True,
                'message': 'Content generated. Images are being generated in the background.',
                'post_id': post.id,
            })
        return JsonResponse({'success': False, 'message': 'Invalid request.'})
    
    def ajax_regenerate_versions(self, request):
        if request.method == 'POST':
            data = json.loads(request.body.decode())
            post_id = data.get('post_id')
            content_type = data.get('content_type')

            post = get_object_or_404(Post, pk=post_id)
            keyword = post.keyword
            prompt = post.prompt
            model_info = post.model_info
            version_count = post.version
            client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

            def fill_prompt(tmpl):
                if tmpl:
                    return tmpl.replace('{{keyword}}', keyword.keyword).replace('{{hairstyle_name}}', keyword.keyword)
                return ''

            def gpt_content(system_prompt, user_prompt):
                if not user_prompt:
                    return ''
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return response.choices[0].message.content

            if content_type == 'title':
                content = gpt_content("", fill_prompt(prompt.title_prompt))
                post.generated_title = content.strip('"').strip("'")
                post.save()
                return JsonResponse({
                    'success': True,
                    'message': 'Title regenerated.',
                    'content': post.generated_title
                })

            elif content_type == 'intro':
                content = gpt_content("you are a helpful assistant that generates engaging article introductions. Do not use markdown.", fill_prompt(prompt.intro_prompt))
                post.generated_intro = content
                post.save()
                return JsonResponse({
                    'success': True,
                    'message': 'Intro regenerated.',
                    'content': post.generated_intro
                })

            elif content_type == 'style_section':
                content = gpt_content(
                    f"You are a helpful assistant that generates detailed style descriptions. Use <h2> HTML headings for each style section, followed by paragraphs. Do not use markdown. Generate exactly {version_count} unique hairstyles.",
                    fill_prompt(prompt.style_prompt)
                )
                post.generated_style_section = content
                post.save()
                return JsonResponse({
                    'success': True,
                    'message': 'Style section regenerated.',
                    'content': post.generated_style_section
                })

            elif content_type == 'conclusion':
                content = gpt_content("You are a helpful assistant that generates article conclusions. Do not use markdown.", fill_prompt(prompt.conclusion_prompt))
                post.generated_conclusion = content
                post.save()
                return JsonResponse({
                    'success': True,
                    'message': 'Conclusion regenerated.',
                    'content': post.generated_conclusion
                })

            elif content_type == 'meta_title':
                meta_prompt = fill_prompt(prompt.meta_data_prompt)
                meta_json = gpt_content("Return JSON with meta_title and meta_description.", meta_prompt)
                try:
                    meta_data = json.loads(meta_json)
                    post.meta_title = meta_data.get('meta_title', '')
                except Exception:
                    post.meta_title = meta_json
                post.save()
                return JsonResponse({
                    'success': True,
                    'message': 'Meta title regenerated.',
                    'meta_title': post.meta_title
                })

            elif content_type == 'meta_description':
                meta_prompt = fill_prompt(prompt.meta_data_prompt)
                meta_json = gpt_content("Return JSON with meta_title and meta_description.", meta_prompt)
                try:
                    meta_data = json.loads(meta_json)
                    post.meta_description = meta_data.get('meta_description', '')
                except Exception:
                    post.meta_description = meta_json
                post.save()
                return JsonResponse({
                    'success': True,
                    'message': 'Meta description regenerated.',
                    'meta_description': post.meta_description
                })
            elif content_type == 'featured_image':
                post.featured_image_status = 'in_process'
                post.save()
                threading.Thread(target=generate_post_images_task, args=(post.id,), kwargs={'only_featured': True}).start()
                return JsonResponse({
                    'success': True,
                    'message': 'Featured image regeneration started.',
                    'status': post.featured_image_status
                })

            elif content_type == 'style_images':
                post.style_images_status = 'in_process'
                post.save()
                threading.Thread(target=generate_post_images_task, args=(post.id,), kwargs={'only_style': True}).start()
                return JsonResponse({
                    'success': True,
                    'message': 'Style images regeneration started.',
                    'status': post.style_images_status
                })

            else:
                return JsonResponse({'success': False, 'message': 'Invalid content type.'})

        return JsonResponse({'success': False, 'message': 'Invalid request.'})


    def changelist_view(self, request, extra_context=None):
        if extra_context is None:
            extra_context = {}
        from .models import Prompt
        extra_context['prompts'] = Prompt.objects.all()
        return super().changelist_view(request, extra_context=extra_context)

    def push_to_wordpress(self, request, post_id):
        import requests
        from django.conf import settings
        from django.http import JsonResponse
        from .models import Post
        post = Post.objects.get(pk=post_id)

        # 1. Upload the featured image to WordPress
        media_id = None
        if post.featured_image:
            image_path = post.featured_image.path
            image_name = post.featured_image.name.split("/")[-1]
            media_headers = {
                "Authorization": f"Bearer {settings.WORDPRESS_JWT_TOKEN}",
                "Content-Disposition": f'attachment; filename="{image_name}"',
                "Content-Type": "image/png" if image_name.endswith(".png") else "image/jpeg",
            }
            with open(image_path, "rb") as img:
                media_response = requests.post(
                    f"{settings.WORDPRESS_API_URL.rsplit('/', 1)[0]}/media",
                    headers=media_headers,
                    data=img,
                )
            media_response.raise_for_status()
            media_id = media_response.json()["id"]

        # 2. Prepare post data
        title_content_plain = post.generated_title or ""
        combined_content = f"{post.generated_intro or ''}\n\n{post.generated_style_section or ''}\n\n{post.generated_conclusion or ''}"
        post_data = {
            "title": title_content_plain,
            "content": combined_content,
            "status": "draft",
        }
        if media_id:
            post_data["featured_media"] = media_id

        post_headers = {
            "Authorization": f"Bearer {settings.WORDPRESS_JWT_TOKEN}",
            "Content-Type": "application/json",
        }

        # 3. Create the post in WordPress
        response = requests.post(
            settings.WORDPRESS_API_URL,
            headers=post_headers,
            json=post_data,
        )
        response.raise_for_status()
        return JsonResponse(response.json())

    def generate_content_for_selected(self, request, queryset):
        from WordAI_Publisher.tasks import generate_post_images_task
        from .models import Prompt, ModelInfo, Post
        import threading

        client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        models = list(ModelInfo.objects.all())
        count = 0
        skipped = []

        for keyword in queryset:
            prompt = Prompt.objects.filter(prompt_id=keyword.prompt_id).first()
            if not prompt or not models:
                skipped.append(keyword.keyword)
                continue
            model_info = random.choice(models)

            def fill_prompt(tmpl):
                if tmpl:
                    return tmpl.replace('{{keyword}}', keyword.keyword).replace('{{hairstyle_name}}', keyword.keyword)
                return ''
            title_prompt = fill_prompt(prompt.title_prompt)
            intro_prompt = fill_prompt(prompt.intro_prompt)
            style_prompt = fill_prompt(prompt.style_prompt)
            conclusion_prompt = fill_prompt(prompt.conclusion_prompt)
            meta_data_prompt = fill_prompt(prompt.meta_data_prompt)

            def gpt_content(system_prompt, user_prompt):
                if not user_prompt:
                    return ''
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return response.choices[0].message.content

            generated_title = gpt_content("", title_prompt)
            if generated_title:
                generated_title = generated_title.strip().strip('"').strip("'")
            generated_intro = gpt_content("You are a helpful assistant that generates engaging article introductions. Do not use markdown.", intro_prompt)
            generated_style_section = gpt_content(
                "You are a helpful assistant that generates detailed style descriptions. Use <h2> HTML headings for each style section, followed by paragraphs. Do not use markdown.",
                style_prompt
            )
            generated_conclusion = gpt_content("You are a helpful assistant that generates article conclusions. Do not use markdown.", conclusion_prompt)
            meta_title = ''
            meta_description = ''
            if meta_data_prompt:
                meta_json = gpt_content("You are an SEO assistant that generates concise meta titles and descriptions in JSON format.", meta_data_prompt)
                try:
                    meta_data = json.loads(meta_json)
                    meta_title = meta_data.get('meta_title', '')
                    meta_description = meta_data.get('meta_description', '')
                except Exception:
                    meta_title = ''
                    meta_description = meta_json

            post = Post.objects.create(
                keyword=keyword,
                prompt=prompt,
                model_info=model_info,
                version=1,
                generated_title=generated_title,
                generated_intro=generated_intro,
                generated_style_section=generated_style_section,
                generated_conclusion=generated_conclusion,
                meta_title=meta_title,
                meta_description=meta_description,
                status='draft',
                content_generated='completed',
                featured_image_status='in_process',
                style_images_status='in_process',
            )
            post.save()

            # Start image generation in a background thread (no Celery)
            threading.Thread(target=generate_post_images_task, args=(post.id,)).start()

            count += 1

        self.message_user(
            request,
            f"Content generated for {count} keywords. Skipped: {', '.join(skipped)}"
        )

    generate_content_for_selected.short_description = "Generate Content for selected keywords"

@admin.register(Post)
class PostAdmin(admin.ModelAdmin):
    change_form_template = "admin/WordAI_Publisher/post_change_form.html"
    form = PostAdminForm
    class Media:
            css = {
                'all': ('WordAI_Publisher/css/main.css',)
            }

    list_display = (
        'id', 'keyword',
        'content_generated_icon',  # Use icon method instead of raw field
        'featured_image_status_icon',
        'style_images_status_icon',
        'status', 'created_at',
        'preview_button', 'push_to_wordpress_button',
    )

    search_fields = ('keyword__keyword', 'prompt__prompt_id', 'model_info__model_id')
    list_filter = ('status', 'created_at')
    actions = ['push_selected_to_wordpress', 'regenerate_featured_image', 'regenerate_style_images']

    def get_keyword(self, obj):
        return obj.keyword.keyword
    get_keyword.short_description = 'Keyword'

    def preview_button(self, obj):
            return format_html(
            '<a class="button preview-btn" href="{}" data-object-id="{}" target="_blank">Preview</a>',
            reverse('admin:post_preview', args=[obj.pk]),
            obj.pk
        )
    preview_button.short_description = "Preview"

    def push_to_wordpress_button(self, obj):
        return format_html(
            '<a class="button" href="{}">Push to WordPress</a>',
            reverse('admin:push_post_to_wordpress', args=[obj.pk])
        )
    push_to_wordpress_button.short_description = "Push to WordPress"
    
    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('preview/<int:post_id>/', self.admin_site.admin_view(self.preview_view), name='post_preview'),
            path('push-to-wordpress/<int:post_id>/', self.admin_site.admin_view(self.push_to_wordpress), name='push_post_to_wordpress'),
        ]
        return custom_urls + urls

    def preview_view(self, request, post_id):
        post = get_object_or_404(Post, pk=post_id)
        style_sections = []

        # Safely decode style_images JSON
        style_images = post.style_images
        if isinstance(style_images, str):
            try:
                style_images = json.loads(style_images)
            except json.JSONDecodeError:
                style_images = {}

        # Normalize keys by stripping whitespace
        style_images_clean = {k.strip(): v for k, v in style_images.items()}

        # Only build style sections if we have a generated style section
        if post.generated_style_section:
            headings = re.findall(r'(<h2>.*?</h2>)', post.generated_style_section, re.DOTALL)
            contents = re.split(r'<h2>.*?</h2>', post.generated_style_section)[1:]

            for i, heading in enumerate(headings):
                style_name = re.sub(r'<.*?>', '', heading).strip()

                # Default to placeholder
                image_url = '/static/WordAI_Publisher/img/placeholder.png'

                # If images are done generating, use actual image if available
                if post.style_images_status != 'in_process':
                    image_url = style_images_clean.get(style_name, image_url)

                # Build section
                section_content = contents[i] if i < len(contents) else ""
                style_sections.append({
                    'heading': mark_safe(heading),
                    'image_url': image_url,
                    'content': mark_safe(section_content),
                })

        return render(request, 'admin/WordAI_Publisher/post_preview.html', {
            'post': post,
            'style_sections': style_sections,
        })
        
    def push_to_wordpress(self, request, post_id):
        post = get_object_or_404(Post, pk=post_id)
        self._push_post_to_wordpress(post, request)
        return redirect(request.META.get('HTTP_REFERER', '/admin/WordAI_Publisher/post/'))

    def push_selected_to_wordpress(self, request, queryset):
        success_count = 0
        for post in queryset:
            result = self._push_post_to_wordpress(post, request, silent=True)
            if result:
                success_count += 1
        self.message_user(request, f"{success_count} posts pushed to WordPress as draft!")
    push_selected_to_wordpress.short_description = "Push selected posts to WordPress"

    def _push_post_to_wordpress(self, post, request=None, silent=False):
        title_content_plain = re.sub(r'<[^>]*>', '', post.generated_title or '').strip()
        combined_content = f"{post.generated_intro or ''}\n\n{post.generated_style_section or ''}\n\n{post.generated_conclusion or ''}"
        wordpress_api_url = settings.WORDPRESS_API_URL

        # === STEP 1: Upload Featured Image ===
        image_id = None
        try:
            image_path = post.featured_image.path
            media_headers = {
                "Authorization": f"Bearer {settings.WORDPRESS_JWT_TOKEN}",
                "Content-Disposition": f'attachment; filename="{os.path.basename(image_path)}"',
                "Content-Type": "image/jpeg",
                'User-Agent': 'Mozilla/5.0',
                'Accept': 'application/json',
            }

            with open(image_path, "rb") as img:
                response = requests.post(settings.WORDPRESS_API_URL_MEDIA, headers=media_headers, data=img)

            if response.status_code == 201:
                image_id = response.json()['id']
            elif not silent:
                self.message_user(request, f"Featured image upload failed: {response.status_code} {response.text}", level=messages.ERROR)

        except Exception as e:
            if not silent:
                self.message_user(request, f"Image upload exception: {e}", level=messages.ERROR)

        # === STEP 2: Upload Gallery Style Images ===
        style_images_data = post.style_images or {}
        gallery_media_ids = []
        threads = []

        def upload_image_thread(style_name, image_path, collected_ids):
            try:
                media_headers_thread = {
                    "Authorization": f"Bearer {settings.WORDPRESS_JWT_TOKEN}",
                    "Content-Type": "image/png",
                    "Content-Disposition": f'attachment; filename="{os.path.basename(image_path)}"',
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "application/json",
                    "Referer": "https://ieghpqrlf6.wpdns.site/",
                }

                with open(image_path, "rb") as img:
                    response = requests.post(settings.WORDPRESS_API_URL_MEDIA, headers=media_headers_thread, data=img)

                if response.status_code == 201:
                    collected_ids.append(response.json()['id'])
                elif not silent:
                    print(f"Failed to upload {style_name}: {response.text}")

            except Exception as e:
                if not silent:
                    print(f"Error uploading {style_name}: {str(e)}")

        for style_name, image_url_path in style_images_data.items():
            relative_path = image_url_path.replace('/media/', '')
            image_path = os.path.join(settings.MEDIA_ROOT, relative_path)
            thread = threading.Thread(target=upload_image_thread, args=(style_name, image_path, gallery_media_ids))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # === STEP 3: Create Post ===
        post_headers = {
            "Authorization": f"Bearer {settings.WORDPRESS_JWT_TOKEN}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        }

        post_data = {
            "title": title_content_plain,
            "content": combined_content,
            "status": "draft",
            "acf": {
                "ai_title": post.generated_title,
                "ai_intro": post.generated_intro,
                "ai_style": post.generated_style_section,
                "ai_conclusion": post.generated_conclusion,
                "style_images": gallery_media_ids
            },
            "featured_media": image_id if image_id else None
        }

        try:
            response = requests.post(wordpress_api_url, headers=post_headers, json=post_data)
            if response.status_code == 201:
                post.status = 'publish'
                post.save()
                if not silent:
                    self.message_user(request, "Post pushed to WordPress as draft!", level=messages.SUCCESS)
                return True
            else:
                if not silent:
                    error_message = response.json().get('message', response.text)
                    self.message_user(request, f"Failed to push post: {error_message}", level=messages.ERROR)

        except requests.exceptions.RequestException as e:
            if not silent:
                self.message_user(request, f"Network/WordPress API Error: {e}", level=messages.ERROR)

        return False

    def content_generated_icon(self, obj):
        if obj.content_generated == 'completed':
            return format_html('<span style="color:green;">&#10004;</span>')  # ✔️
        else:
            return format_html('<span style="color:red;">&#10008;</span>')    # ❌
    content_generated_icon.short_description = "Content Generated"

    def featured_image_status_icon(self, obj):
        if obj.featured_image_status == 'completed':
            return format_html('<span style="color:green;">&#10004;</span>')
        elif obj.featured_image_status == 'in_process':
            return format_html('<span style="color:orange;">&#8987;</span>')  # ⏳
        else:
            return format_html('<span style="color:red;">&#10008;</span>')
    featured_image_status_icon.short_description = "Featured Image Status"

    def style_images_status_icon(self, obj):
        if obj.style_images_status == 'completed':
            return format_html('<span style="color:green;">&#10004;</span>')
        elif obj.style_images_status == 'in_process':
            return format_html('<span style="color:orange;">&#8987;</span>')
        else:
            return format_html('<span style="color:red;">&#10008;</span>')
    style_images_status_icon.short_description = "Style Images Status"


    def regenerate_featured_image(self, request, queryset):
        from WordAI_Publisher.tasks import generate_post_images_task
        count = 0
        for post in queryset:
            post.featured_image_status = 'in_process'
            post.save()
            # Start featured image generation in a background thread
            threading.Thread(
                target=generate_post_images_task,
                args=(post.id,),
                kwargs={'only_featured': True}
            ).start()
            count += 1
        self.message_user(
            request,
            f"✅ Regeneration started for featured images of {count} post(s). It is running in the background.",
            level=messages.SUCCESS
        )
    regenerate_featured_image.short_description = "Regenerate Featured Image"

    def regenerate_style_images(self, request, queryset):
        from WordAI_Publisher.tasks import generate_post_images_task
        count = 0
        for post in queryset:
            post.style_images_status = 'in_process'
            post.save()
            # Start style images generation in a background thread
            threading.Thread(
                target=generate_post_images_task,
                args=(post.id,),
                kwargs={'only_style': True}
            ).start()
            count += 1
        self.message_user(
            request,
            f"✅ Regeneration started for style images of {count} post(s). It is running in the background.",
            level=messages.SUCCESS
        )
    regenerate_style_images.short_description = "Regenerate Style Images"

def extract_styles_by_h2(style_section_html):
    if not isinstance(style_section_html, str):
        return []
    pattern = r'<h2>(.*?)</h2>'
    return re.findall(pattern, style_section_html)

