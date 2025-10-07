import os
import re
import random
import base64
from bs4 import BeautifulSoup
import logging
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
import openai
from .models import Post, ModelInfo
from itertools import cycle
import sys

# Configure logger to output to terminal
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


def extract_styles_by_h2(style_section_html):
    styles = []
    if isinstance(style_section_html, str):
        soup = BeautifulSoup(style_section_html, 'html.parser')
        for h2 in soup.find_all('h2'):
            styles.append(h2.get_text(strip=True))
    logger.info(f"[extract_styles_by_h2] Extracted styles: {styles}")
    return styles


def _fill_image_placeholders(tmpl: str, keyword: str, mi: ModelInfo, style_desc: str = "") -> str:
    """
    Replace prompt placeholders with Post + ModelInfo values.
    Supports the same placeholders you used for featured/style images.
    """
    if not tmpl:
        return ""
    out = tmpl.replace('{{keyword}}', keyword)
    out = out.replace('{{style_image_description}}', style_desc or '')

    # ModelInfo-backed placeholders (gracefully empty if None)
    repls = {
        '{{ethnicity}}': getattr(mi, 'ethnicity', None),
        '{{skin_tone}}': getattr(mi, 'skin_tone', None),
        '{{hair_texture}}': getattr(mi, 'hair_texture', None),
        '{{face_shape}}': getattr(mi, 'face_shape', None),
        '{{tshirt}}': getattr(mi, 'tshirt', None),
        '{{eye_color}}': getattr(mi, 'eye_color', None),
        '{{hair_color}}': getattr(mi, 'hair_color', None),
        '{{build_description}}': getattr(mi, 'build_description', None),
        '{{expression_description}}': getattr(mi, 'expression_description', None),
        '{{wardrobe_color}}': getattr(mi, 'wardrobe_color', None),
        '{{wardrobe_item}}': getattr(mi, 'wardrobe_item', None),
        '{{grooming_description}}': getattr(mi, 'grooming_description', None),
        '{{brand}}': getattr(mi, 'brand', None),
    }
    for k, v in repls.items():
        out = out.replace(k, v or '')
    return out


def generate_post_images_task(
    post_id,
    only_featured=False,
    only_style=False,
    specific_style=None,
    featured_prompt_text=None,
    style_prompts_override=None
):
    logger.info(f"Started image generation for post {post_id} (only_featured={only_featured}, only_style={only_style}, specific_style={specific_style})")
    try:
        post = Post.objects.get(pk=post_id)
        prompt = post.prompt

        model_infos = ModelInfo.objects.all()
        first_model_info = model_infos.first()

        client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

        # === FEATURED IMAGE ===
        if only_featured or not (only_featured or only_style):
            post.featured_image_status = 'in_process'
            post.save(update_fields=['featured_image_status'])
            try:
                image_prompt = featured_prompt_text or prompt.featured_image_prompt or ''
                print(image_prompt)
                image_prompt = image_prompt.replace('{{keyword}}', post.keyword.keyword)
                if first_model_info:
                    image_prompt = image_prompt.replace('{{ethnicity}}', first_model_info.ethnicity or '')
                    image_prompt = image_prompt.replace('{{style_image_description}}', '')
                    image_prompt = image_prompt.replace('{{skin_tone}}', first_model_info.skin_tone or '')
                    image_prompt = image_prompt.replace('{{hair_texture}}', first_model_info.hair_texture or '')
                    image_prompt = image_prompt.replace('{{face_shape}}', first_model_info.face_shape or '')
                    image_prompt = image_prompt.replace('{{tshirt}}', first_model_info.tshirt or '')
                    image_prompt = image_prompt.replace('{{eye_color}}', first_model_info.eye_color or '')

                    # New fields
                    image_prompt = image_prompt.replace('{{hair_color}}', first_model_info.hair_color or '')
                    image_prompt = image_prompt.replace('{{build_description}}', first_model_info.build_description or '')
                    image_prompt = image_prompt.replace('{{expression_description}}', first_model_info.expression_description or '')
                    image_prompt = image_prompt.replace('{{wardrobe_color}}', first_model_info.wardrobe_color or '')
                    image_prompt = image_prompt.replace('{{wardrobe_item}}', first_model_info.wardrobe_item or '')
                    image_prompt = image_prompt.replace('{{grooming_description}}', first_model_info.grooming_description or '')
                    image_prompt = image_prompt.replace('{{brand}}', first_model_info.brand or '')

                logger.info(f"GPT-4.1-mini prompt for featured image: {image_prompt}")

                post.featured_prompt_text = image_prompt

                result = client.images.generate(
                    model="gpt-image-1",
                    prompt=image_prompt,
                    size="1536x1024",
                    quality="high"
                )

                # If SDK supports .json(); otherwise, stringify safely
                try:
                    logger.info(f"Full GPT response for featured image: {result.json()}")
                except Exception:
                    logger.info(f"Full GPT response for featured image (raw): {result}")

                image_base64 = result.data[0].b64_json

                if image_base64:
                    image_data = base64.b64decode(image_base64)
                    image_name = f"{post.keyword.keyword.replace(' ', '_')}_featured_{random.randint(1000,9999)}.png"
                    content_file = ContentFile(image_data)
                    post.featured_image.save(image_name, content_file, save=True)
                    post.featured_image_status = 'completed'
                    logger.info(f"Featured image saved for post {post_id}")
                else:
                    post.featured_image_status = 'not_generated'
                    logger.warning("No image data returned by GPT for featured image.")

            except Exception as e:
                logger.exception(f"Featured image generation failed for post {post_id}: {e}")
                post.featured_image_status = 'not_generated'

            post.save(update_fields=['featured_image', 'featured_image_status', 'featured_prompt_text'])

        # === STYLE IMAGES ===
        if only_style or not (only_featured or only_style):
            post.style_images_status = 'in_process'
            post.save(update_fields=['style_images_status'])

            style_image_descriptions = post.style_image_descriptions or {}
            style_names = list(style_image_descriptions.keys())

            if not style_names:
                logger.warning(f"No style image descriptions found for post {post_id}. Using fallback style.")
                style_names = ["Generic Hairstyle"]
                style_image_descriptions["Generic Hairstyle"] = "This is a placeholder description for a generic hairstyle."

            if specific_style:
                style_names = [specific_style]

            style_images = post.style_images or {}
            style_prompt_dict = post.style_prompts or {}

            model_info_cycle = cycle(model_infos)

            for idx, style_name in enumerate(style_names):
                try:
                    current_model_info = next(model_info_cycle)

                    if style_prompt_dict.get(style_name):
                        style_img_prompt = style_prompt_dict[style_name]
                    else:
                        style_img_prompt = (style_prompts_override or {}).get(style_name) if style_prompts_override else None
                        if not style_img_prompt:
                            style_img_prompt = prompt.image_prompt or ''

                        style_img_prompt = style_img_prompt.replace('{{keyword}}', post.keyword.keyword)
                        style_img_prompt = style_img_prompt.replace('{{style_image_description}}', style_image_descriptions.get(style_name, ''))
                        style_img_prompt = style_img_prompt.replace('{{ethnicity}}', current_model_info.ethnicity or '')
                        style_img_prompt = style_img_prompt.replace('{{skin_tone}}', current_model_info.skin_tone or '')
                        style_img_prompt = style_img_prompt.replace('{{hair_texture}}', current_model_info.hair_texture or '')
                        style_img_prompt = style_img_prompt.replace('{{face_shape}}', current_model_info.face_shape or '')
                        style_img_prompt = style_img_prompt.replace('{{tshirt}}', current_model_info.tshirt or '')
                        style_img_prompt = style_img_prompt.replace('{{eye_color}}', current_model_info.eye_color or '')

                        # New fields
                        style_img_prompt = style_img_prompt.replace('{{hair_color}}', current_model_info.hair_color or '')
                        style_img_prompt = style_img_prompt.replace('{{build_description}}', current_model_info.build_description or '')
                        style_img_prompt = style_img_prompt.replace('{{expression_description}}', current_model_info.expression_description or '')
                        style_img_prompt = style_img_prompt.replace('{{wardrobe_color}}', current_model_info.wardrobe_color or '')
                        style_img_prompt = style_img_prompt.replace('{{wardrobe_item}}', current_model_info.wardrobe_item or '')
                        style_img_prompt = style_img_prompt.replace('{{grooming_description}}', current_model_info.grooming_description or '')
                        style_img_prompt = style_img_prompt.replace('{{brand}}', current_model_info.brand or '')

                        style_prompt_dict[style_name] = style_img_prompt

                    style_img_response = client.images.generate(
                        model="gpt-image-1",
                        prompt=style_img_prompt,
                        size="1536x1024",
                        quality="high"
                    )

                    logger.info(f"[{post_id}] GPT-Image-1 response for style '{style_name}': {style_img_response}")

                    image_base64 = style_img_response.data[0].b64_json

                    if image_base64:
                        safe_style_name = "".join(c if c.isalnum() else "_" for c in style_name)
                        style_img_file = f"{safe_style_name}_style_{post.id}_{random.randint(1000,9999)}.png"
                        image_data = base64.b64decode(image_base64)
                        style_file = ContentFile(image_data)

                        style_s3_path = f"style_images/{style_img_file}"
                        default_storage.save(style_s3_path, style_file)

                        style_url = f"https://{getattr(settings, 'AWS_STORAGE_BUCKET_NAME', 'your-bucket')}.s3.{getattr(settings, 'AWS_S3_REGION_NAME', 'us-east-1')}.amazonaws.com/{style_s3_path}"
                        style_images[style_name] = style_url

                        logger.info(f"[{post_id}] Saved style image '{style_name}' to S3 at {style_url}")
                    else:
                        style_images[style_name] = None
                        logger.warning(f"[{post_id}] No image data returned by GPT for style '{style_name}'")

                except Exception as e:
                    logger.exception(f"[{post_id}] Style image generation failed for '{style_name}': {e}")
                    style_images[style_name] = None

            post.style_images = style_images
            post.style_prompts = style_prompt_dict
            post.style_images_status = 'completed'
            post.save(update_fields=['style_images', 'style_images_status', 'style_prompts'])

        # === SECTION IMAGES (8 new images) ===
        # Generate when running "styles" or when running everything (i.e., not only_featured)
        if only_style or not (only_featured or only_style):
            logger.info(f"[{post_id}] Generating section images (8)")

            # Map: (post_field_to_store_url, prompt_field_on_prompt_model, safe_prefix)
            section_specs = [
                ('generated_quick_style_snapshot_image',        'quick_style_snapshot_image_prompt',        'quick_style_snapshot'),
                ('generated_packing_essentials_checklist_image','packing_essentials_checklist_image_prompt','packing_essentials'),
                ('generated_daytime_outfits_image',             'daytime_outfits_image_prompt',             'daytime_outfits'),
                ('generated_evening_and_nightlife_image',       'evening_and_nightlife_image_prompt',       'evening_nightlife'),
                ('generated_outdoor_activities_image',          'outdoor_activities_image_prompt',          'outdoor_activities'),
                ('generated_seasonal_variations_image',         'seasonal_variations_image_prompt',         'seasonal_variations'),
                ('generated_style_tips_for_blending_image',     'style_tips_for_blending_image_prompt',     'style_tips_blending'),
                ('generated_destination_specific_extras_image', 'destination_specific_extras_image_prompt', 'destination_extras'),
            ]

            # Rotate through ModelInfo objects for variety, like you do for styles
            model_info_cycle = cycle(model_infos if model_infos else [first_model_info])

            update_fields = []
            for post_field, prompt_field, prefix in section_specs:
                try:
                    mi = next(model_info_cycle) or first_model_info
                    raw_tmpl = getattr(prompt, prompt_field, '') or ''
                    if not raw_tmpl.strip():
                        logger.warning(f"[{post_id}] Missing prompt for section '{post_field}' ({prompt_field}); skipping.")
                        continue

                    filled_prompt = _fill_image_placeholders(
                        raw_tmpl,
                        keyword=post.keyword.keyword,
                        mi=mi,
                        style_desc=""  # not used for section images
                    )

                    logger.info(f"[{post_id}] Section prompt ({post_field}): {filled_prompt}")

                    resp = client.images.generate(
                        model="gpt-image-1",
                        prompt=filled_prompt,
                        size="1536x1024",     # keep consistent with your featured/style assets
                        quality="high"
                    )

                    image_base64 = resp.data[0].b64_json if resp and resp.data else None
                    if not image_base64:
                        logger.warning(f"[{post_id}] No image data for section '{post_field}'")
                        setattr(post, post_field, None)
                        update_fields.append(post_field)
                        continue

                    image_data = base64.b64decode(image_base64)
                    safe_kw = "".join(c if c.isalnum() else "_" for c in post.keyword.keyword)
                    filename = f"{safe_kw}_{prefix}_{post.id}_{random.randint(1000,9999)}.png"
                    s3_path = f"section_images/{filename}"

                    default_storage.save(s3_path, ContentFile(image_data))

                    section_url = (
                        f"https://{getattr(settings, 'AWS_STORAGE_BUCKET_NAME', 'your-bucket')}"
                        f".s3.{getattr(settings, 'AWS_S3_REGION_NAME', 'us-east-1')}.amazonaws.com/{s3_path}"
                    )

                    setattr(post, post_field, section_url)
                    update_fields.append(post_field)
                    logger.info(f"[{post_id}] Saved section image '{post_field}' to {section_url}")

                except Exception as e:
                    logger.exception(f"[{post_id}] Section image generation failed for '{post_field}': {e}")
                    setattr(post, post_field, None)
                    update_fields.append(post_field)

            if update_fields:
  
                post.save(update_fields=update_fields)
             # always persist the status afterwards
            post.modular_images_status = 'completed'
            post.save(update_fields=['modular_images_status'])
            
    except Exception as e:
        logger.exception(f"Error in generate_post_images_task for post {post_id}: {e}")
    logger.info(f"Finished image generation for post {post_id}")
