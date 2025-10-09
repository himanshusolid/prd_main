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
    Supports the same placeholders you used for featured/style/section images.
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


# Map each modular section image field -> (Prompt model field, filename prefix)
SECTION_MAP = {
    'generated_quick_style_snapshot_image':         ('quick_style_snapshot_image_prompt',         'quick_style_snapshot'),
    'generated_packing_essentials_checklist_image': ('packing_essentials_checklist_image_prompt', 'packing_essentials'),
    'generated_daytime_outfits_image':              ('daytime_outfits_image_prompt',               'daytime_outfits'),
    'generated_evening_and_nightlife_image':        ('evening_and_nightlife_image_prompt',         'evening_nightlife'),
    'generated_outdoor_activities_image':           ('outdoor_activities_image_prompt',            'outdoor_activities'),
    'generated_seasonal_variations_image':          ('seasonal_variations_image_prompt',           'seasonal_variations'),
    'generated_style_tips_for_blending_image':      ('style_tips_for_blending_image_prompt',       'style_tips_blending'),
    'generated_destination_specific_extras_image':  ('destination_specific_extras_image_prompt',   'destination_extras'),
}


def _resolve_section_raw_prompt(post: Post, prompt_obj, post_field: str, prompt_field: str):
    """
    Resolve the raw prompt for a section with JSON-first fallback, then prompt table.

    Preference order:
      1) post.extra_image_saved_prompts[post_field]  (if your model has this optional field)
      2) post.extra_image_used_prompts[post_field]
      3) getattr(prompt_obj, prompt_field) from the Prompt table

    Returns: (raw_prompt: str, source: str)
    """
    # 1) Optional per-post saved JSON (if you add/use it)
    saved_json = getattr(post, 'extra_image_saved_prompts', None) or {}
    raw = saved_json.get(post_field)
    if isinstance(raw, str) and raw.strip():
        return raw, 'post_json_saved'

    # 2) Previously used per-post JSON (what you save after generation)
    used_json = getattr(post, 'extra_image_used_prompts', None) or {}
    raw = used_json.get(post_field)
    if isinstance(raw, str) and raw.strip():
        return raw, 'post_json_used'

    # 3) Prompt table fallback
    raw = getattr(prompt_obj, prompt_field, '') or ''
    if raw.strip():
        return raw, 'prompt_table'

    return '', 'missing'


def generate_post_images_task(
    post_id,
    only_featured: bool = False,
    only_style: bool = False,
    only_sections: bool = False,      # NEW: generate only the 8 modular sections (or a subset via `sections`)
    specific_style: str = None,
    featured_prompt_text: str = None,
    style_prompts_override: dict = None,
    sections: list = None,            # NEW: list of post field names (from SECTION_MAP keys). None => all 8
    image_size: str = "1536x1024",
):
    logger.info(
        f"Started image generation for post {post_id} "
        f"(only_featured={only_featured}, only_style={only_style}, only_sections={only_sections}, "
        f"specific_style={specific_style}, sections={sections})"
    )
    try:
        post = Post.objects.get(pk=post_id)
        prompt = post.prompt

        model_infos = ModelInfo.objects.all()
        first_model_info = model_infos.first()

        client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

        # Decide which blocks to run
        run_featured  = only_featured or (not only_featured and not only_style and not only_sections)
        run_style     = only_style    or (not only_featured and not only_style and not only_sections)
        run_sections  = only_sections or (not only_featured and not only_style and not only_sections)

        # === FEATURED IMAGE ===
        if run_featured:
            post.featured_image_status = 'in_process'
            post.save(update_fields=['featured_image_status'])
            try:
                image_prompt = featured_prompt_text or (getattr(prompt, 'featured_image_prompt', '') or '')
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

                logger.info(f"[{post_id}] Featured prompt: {image_prompt}")

                # Save the actual prompt used for featured
                post.featured_prompt_text = image_prompt

                result = client.images.generate(
                    model="gpt-image-1",
                    prompt=image_prompt,
                    size=image_size,
                    quality="high"
                )
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
                    logger.info(f"[{post_id}] Featured image saved")
                else:
                    post.featured_image_status = 'not_generated'
                    logger.warning(f"[{post_id}] No data for featured image")

            except Exception as e:
                logger.exception(f"[{post_id}] Featured image generation failed: {e}")
                post.featured_image_status = 'not_generated'

            post.save(update_fields=['featured_image', 'featured_image_status', 'featured_prompt_text'])

        # === STYLE IMAGES (gallery) ===
        if run_style:
            post.style_images_status = 'in_process'
            post.save(update_fields=['style_images_status'])

            style_image_descriptions = post.style_image_descriptions or {}
            style_names = list(style_image_descriptions.keys())

            if not style_names:
                logger.warning(f"[{post_id}] No style image descriptions; using fallback")
                style_names = ["Generic Hairstyle"]
                style_image_descriptions["Generic Hairstyle"] = "This is a placeholder description for a generic hairstyle."

            if specific_style:
                style_names = [specific_style]

            style_images = post.style_images or {}
            style_prompt_dict = post.style_prompts or {}

            model_info_cycle = cycle(model_infos if model_infos else [first_model_info])

            for style_name in style_names:
                try:
                    current_model_info = next(model_info_cycle) or first_model_info

                    if style_prompt_dict.get(style_name):
                        style_img_prompt = style_prompt_dict[style_name]
                    else:
                        style_img_prompt = (style_prompts_override or {}).get(style_name) if style_prompts_override else None
                        if not style_img_prompt:
                            style_img_prompt = getattr(prompt, 'image_prompt', '') or ''

                        style_img_prompt = style_img_prompt.replace('{{keyword}}', post.keyword.keyword)
                        style_img_prompt = style_img_prompt.replace('{{style_image_description}}', style_image_descriptions.get(style_name, ''))
                        style_img_prompt = style_img_prompt.replace('{{ethnicity}}', (getattr(current_model_info, 'ethnicity', None) or ''))
                        style_img_prompt = style_img_prompt.replace('{{skin_tone}}', (getattr(current_model_info, 'skin_tone', None) or ''))
                        style_img_prompt = style_img_prompt.replace('{{hair_texture}}', (getattr(current_model_info, 'hair_texture', None) or ''))
                        style_img_prompt = style_img_prompt.replace('{{face_shape}}', (getattr(current_model_info, 'face_shape', None) or ''))
                        style_img_prompt = style_img_prompt.replace('{{tshirt}}', (getattr(current_model_info, 'tshirt', None) or ''))
                        style_img_prompt = style_img_prompt.replace('{{eye_color}}', (getattr(current_model_info, 'eye_color', None) or ''))
                        # New fields
                        style_img_prompt = style_img_prompt.replace('{{hair_color}}', (getattr(current_model_info, 'hair_color', None) or ''))
                        style_img_prompt = style_img_prompt.replace('{{build_description}}', (getattr(current_model_info, 'build_description', None) or ''))
                        style_img_prompt = style_img_prompt.replace('{{expression_description}}', (getattr(current_model_info, 'expression_description', None) or ''))
                        style_img_prompt = style_img_prompt.replace('{{wardrobe_color}}', (getattr(current_model_info, 'wardrobe_color', None) or ''))
                        style_img_prompt = style_img_prompt.replace('{{wardrobe_item}}', (getattr(current_model_info, 'wardrobe_item', None) or ''))
                        style_img_prompt = style_img_prompt.replace('{{grooming_description}}', (getattr(current_model_info, 'grooming_description', None) or ''))
                        style_img_prompt = style_img_prompt.replace('{{brand}}', (getattr(current_model_info, 'brand', None) or ''))

                        style_prompt_dict[style_name] = style_img_prompt

                    style_img_response = client.images.generate(
                        model="gpt-image-1",
                        prompt=style_img_prompt,
                        size=image_size,
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

                        style_url = (
                            f"https://{getattr(settings, 'AWS_STORAGE_BUCKET_NAME', 'your-bucket')}"
                            f".s3.{getattr(settings, 'AWS_S3_REGION_NAME', 'us-east-1')}.amazonaws.com/{style_s3_path}"
                        )
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

        # === SECTION / MODULAR IMAGES (8 images; or a subset if `sections` provided) ===
        if run_sections:
            logger.info(f"[{post_id}] Generating modular section images")

            # Decide which status field name to use (prefer new modular_images_status)
            status_field = 'modular_images_status' if hasattr(post, 'modular_images_status') else 'section_images_status'

            setattr(post, status_field, 'in_process')
            post.save(update_fields=[status_field])

            try:
                # What to generate?
                if sections:
                    to_generate = [s for s in sections if s in SECTION_MAP]
                else:
                    to_generate = list(SECTION_MAP.keys())

                model_info_cycle = cycle(model_infos if model_infos else [first_model_info])

                update_fields = []
                used_updates = {}  # { post_field: filled_prompt }

                for post_field in to_generate:
                    try:
                        prompt_field, prefix = SECTION_MAP[post_field]
                        mi = next(model_info_cycle) or first_model_info

                        # JSON-first, then Prompt table
                        raw_tmpl, src = _resolve_section_raw_prompt(post, prompt, post_field, prompt_field)
                        if not raw_tmpl.strip():
                            logger.warning(f"[{post_id}] Missing prompt for section '{post_field}' ({prompt_field}); source={src}; skipping.")
                            setattr(post, post_field, None)
                            update_fields.append(post_field)
                            continue

                        filled_prompt = _fill_image_placeholders(
                            raw_tmpl,
                            keyword=post.keyword.keyword,
                            mi=mi,
                            style_desc=""
                        )
                        logger.info(f"[{post_id}] Section prompt source={src} ({post_field}): {filled_prompt}")

                        resp = client.images.generate(
                            model="gpt-image-1",
                            prompt=filled_prompt,
                            size=image_size,
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

                        # record the exact USED prompt (plain string) keyed by section field name
                        used_updates[post_field] = filled_prompt

                    except Exception as e:
                        logger.exception(f"[{post_id}] Section image generation failed for '{post_field}': {e}")
                        setattr(post, post_field, None)
                        update_fields.append(post_field)

                # merge + persist the USED prompts JSON (as plain strings)
                if used_updates:
                    post.refresh_from_db(fields=["extra_image_used_prompts"])
                    current = post.extra_image_used_prompts or {}
                    current.update(used_updates)  # { "generated_xxx_image": "filled prompt" }
                    post.extra_image_used_prompts = current
                    update_fields.append("extra_image_used_prompts")

                if update_fields:
                    update_fields = list(dict.fromkeys(update_fields))  # dedupe
                    post.save(update_fields=update_fields)

                # Mark COMPLETED
                setattr(post, status_field, 'completed')
                post.save(update_fields=[status_field])

            except Exception as e:
                logger.exception(f"[{post_id}] Section images block failed: {e}")
                setattr(post, status_field, 'not_generated')
                post.save(update_fields=[status_field])

    except Exception as e:
        logger.exception(f"Error in generate_post_images_task for post {post_id}: {e}")

    logger.info(f"Finished image generation for post {post_id}")
