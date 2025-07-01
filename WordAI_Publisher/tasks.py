import os
import base64
import random
import re
from django.conf import settings
from .models import Post
import openai
import logging

logger = logging.getLogger(__name__)

def generate_post_images_task(post_id, only_featured=False, only_style=False):
    logger.info(f"Started image generation for post {post_id} (only_featured={only_featured}, only_style={only_style})")
    try:
        post = Post.objects.get(pk=post_id)
        prompt = post.prompt
        model_info = post.model_info
        client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

        # === FEATURED IMAGE GENERATION ===
        if only_featured or not (only_featured or only_style):
            post.featured_image_status = 'in_process'
            post.save(update_fields=['featured_image_status'])
            try:
                image_prompt = prompt.image_prompt or ''
                image_prompt = image_prompt.replace('{{hairstyle_name}}', post.keyword.keyword)
                if model_info:
                    image_prompt = image_prompt.replace('{{ethnicity}}', model_info.ethnicity or '')
                    image_prompt = image_prompt.replace('{{skin_tone}}', model_info.skin_tone or '')
                    image_prompt = image_prompt.replace('{{hair_texture}}', model_info.hair_texture or '')
                    image_prompt = image_prompt.replace('{{face_shape}}', model_info.face_shape or '')

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
                    image_name = f"{post.keyword.keyword.replace(' ', '_')}_featured_{random.randint(1000,9999)}.png"
                    relative_path = f"featured_images/{image_name}"
                    full_path = os.path.join(settings.MEDIA_ROOT, relative_path)
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    with open(full_path, "wb") as f:
                        f.write(base64.b64decode(image_base64))
                    # Save to ImageField
                    with open(full_path, "rb") as f:
                        post.featured_image.save(image_name, f, save=True)
                    post.featured_image_status = 'completed'
                else:
                    post.featured_image_status = 'not_generated'
            except Exception as e:
                post.featured_image_status = 'not_generated'
            post.save(update_fields=['featured_image', 'featured_image_status'])

        # === STYLE IMAGES GENERATION ===
        if only_style or not (only_featured or only_style):
            post.style_images_status = 'in_process'
            post.save(update_fields=['style_images_status'])

            style_names = extract_styles_by_h2(post.generated_style_section or "")
            style_images = {}

            for style_name in style_names:
                try:
                    style_img_prompt = prompt.image_prompt or ''
                    style_img_prompt = style_img_prompt.replace('{{hairstyle_name}}', style_name)
                    if model_info:
                        style_img_prompt = style_img_prompt.replace('{{ethnicity}}', model_info.ethnicity or '')
                        style_img_prompt = style_img_prompt.replace('{{skin_tone}}', model_info.skin_tone or '')
                        style_img_prompt = style_img_prompt.replace('{{hair_texture}}', model_info.hair_texture or '')
                        style_img_prompt = style_img_prompt.replace('{{face_shape}}', model_info.face_shape or '')

                    style_img_response = client.responses.create(
                        model="gpt-4.1-mini",
                        input=style_img_prompt,
                        tools=[{"type": "image_generation"}],
                    )
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
            post.style_images_status = 'completed'
            post.save(update_fields=['style_images', 'style_images_status'])

    except Exception as e:
        # Optionally log the error
        print(f"Error in generate_post_images_task for post {post_id}: {e}")
    logger.info(f"Finished image generation for post {post_id}")

def extract_styles_by_h2(style_section_html):
    if not isinstance(style_section_html, str):
        return []
    pattern = r'<h2>(.*?)</h2>'
    return re.findall(pattern, style_section_html) 