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

from .models import Keyword, Prompt, ModelInfo, Post
from .admin_forms import CSVUploadForm

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

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('upload-csv/', self.admin_site.admin_view(self.upload_csv)),
            path('generate-content/<int:pk>/', self.admin_site.admin_view(self.generate_content_view), name='generate_content'),
            path('ajax-generate-versions/', self.admin_site.admin_view(self.ajax_generate_versions), name='ajax_generate_versions'),
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
                    Keyword.objects.create(
                        keyword=row.get('keyword', ''),
                        prompt_id=row.get('prompt_id', '')
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

            # --- FEATURED IMAGE GENERATION (USING IMAGE PROMPT FROM DB) ---
            featured_image_url = None
            try:
                image_prompt = prompt.image_prompt or ''
                # Replace all variables in the image prompt
                image_prompt = image_prompt.replace('{{hairstyle_name}}', keyword.keyword)
                if model_info:
                    image_prompt = image_prompt.replace('{{ethnicity}}', model_info.ethnicity or '')
                    image_prompt = image_prompt.replace('{{skin_tone}}', model_info.skin_tone or '')
                    image_prompt = image_prompt.replace('{{hair_texture}}', model_info.hair_texture or '')
                    image_prompt = image_prompt.replace('{{face_shape}}', model_info.face_shape or '')
                print(f"[DEBUG] Final featured image prompt: {image_prompt}")
                # Use the new OpenAI API call for image generation (base64 response)
                featured_img_response = client.responses.create(
                    model="gpt-4.1-mini",
                    input=image_prompt,
                    tools=[{"type": "image_generation"}],
                )
                image_data = [
                    output.result
                    for output in featured_img_response.output
                    if output.type == "image_generation_call"
                ]
                if image_data:
                    image_base64 = image_data[0]
                    image_name = f"{keyword.keyword.replace(' ', '_')}_featured_{random.randint(1000,9999)}.png"
                    relative_path = f"featured_images/{image_name}"
                    full_path = os.path.join(settings.MEDIA_ROOT, relative_path)
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    with open(full_path, "wb") as f:
                        f.write(base64.b64decode(image_base64))
                    featured_image_url = f"{settings.MEDIA_URL}{relative_path}"
                else:
                    featured_image_url = None
            except Exception as e:
                featured_image_url = None

            # Generate content for each section
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

            # Save as Post (without style_images for now)
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
            )
            # Attach the featured image to the post's ImageField
            if featured_image_url:
                with open(full_path, 'rb') as f:
                    post.featured_image.save(image_name, ContentFile(f.read()), save=True)

            # --- STYLE IMAGES GENERATION (SAVED LOCALLY) ---
            style_names = extract_styles_by_h2(generated_style_section or "")
            style_images = {}

            for style_name in style_names:
                # Use the image_prompt from the Prompt table and replace variables
                style_img_prompt = prompt.image_prompt or ''
                style_img_prompt = style_img_prompt.replace('{{hairstyle_name}}', style_name)
                if model_info:
                    style_img_prompt = style_img_prompt.replace('{{ethnicity}}', model_info.ethnicity or '')
                    style_img_prompt = style_img_prompt.replace('{{skin_tone}}', model_info.skin_tone or '')
                    style_img_prompt = style_img_prompt.replace('{{hair_texture}}', model_info.hair_texture or '')
                    style_img_prompt = style_img_prompt.replace('{{face_shape}}', model_info.face_shape or '')

                print(f"[DEBUG] Final style image prompt: {style_img_prompt}")
                try:
                    style_img_response = client.responses.create(
                        model="gpt-4.1-mini",
                        input=style_img_prompt,
                        tools=[{"type": "image_generation"}],
                    )
                    # Extract base64 image data from the response
                    image_data = [
                        output.result
                        for output in style_img_response.output
                        if output.type == "image_generation_call"
                    ]
                    if image_data:
                        image_base64 = image_data[0]
                        safe_style_name = "".join(c if c.isalnum() else "_" for c in style_name)
                        style_img_name = f"{safe_style_name}_style_{post.id}_{random.randint(1000,9999)}.png"
                        style_relative_path = f"style_images/{style_img_name}"
                        style_full_path = os.path.join(settings.MEDIA_ROOT, style_relative_path)
                        os.makedirs(os.path.dirname(style_full_path), exist_ok=True)
                        with open(style_full_path, "wb") as f:
                            f.write(base64.b64decode(image_base64))
                        style_images[style_name] = f"{settings.MEDIA_URL}{style_relative_path}"
                    else:
                        style_images[style_name] = None
                except Exception as e:
                    style_images[style_name] = None

            post.style_images = style_images
            post.save()

            return JsonResponse({
                'success': True,
                'message': 'Post generated with featured image and style images saved locally',
                'post_id': post.id,
                'featured_image': post.featured_image.url if post.featured_image else None,
                'style_images': post.style_images or {}
            })
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

class PostAdmin(admin.ModelAdmin):
    readonly_fields = ('featured_image_preview', 'style_images_preview',)
    fields = (
        'keyword', 'prompt', 'model_info', 'version',
        'generated_title', 'generated_intro', 'generated_style_section', 'generated_conclusion',
        'meta_title', 'meta_description', 'status',
        'featured_image', 'featured_image_preview',
        'style_images_preview',
    )

    def featured_image_preview(self, obj):
        if obj.featured_image:
            return format_html(
                '<img src="{}" style="max-width: 300px; max-height: 200px;" />',
                obj.featured_image.url
            )
        return "-"
    featured_image_preview.short_description = "Featured Image"

    def style_images_preview(self, obj):
        if not obj.style_images:
            return "-"
        html = ""
        for style, url in obj.style_images.items():
            if url:
                html += f"<div><strong>{style}</strong><br><img src='{url}' style='max-width:150px; margin-bottom:10px;'/></div>"
        return format_html(html)
    style_images_preview.short_description = "Style Images"

def extract_styles_by_h2(style_section_html):
    if not isinstance(style_section_html, str):
        return []
    pattern = r'<h2>(.*?)</h2>'
    return re.findall(pattern, style_section_html)
