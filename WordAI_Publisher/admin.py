from django.contrib import admin, messages
from django.urls import path, reverse
from django.shortcuts import render, redirect, get_object_or_404
from django.template.response import TemplateResponse
from django.http import HttpResponse, JsonResponse, HttpResponseBadRequest
from django.utils.html import format_html
from django.utils.safestring import mark_safe
from django import forms
from django.conf import settings
from django.db import models
from django.db.models import JSONField  # ✅ Works with MySQL (Django 3.1+)
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from django.http import HttpResponseRedirect
from WordAI_Publisher.models import GenerationJob  # adjust import path if different

import base64
import csv
import io
import json
import mimetypes
import openai
import os
import random
import re
import requests
import threading
import time
from urllib.parse import urlparse

import markdown  # markdown library
from ckeditor_uploader.widgets import CKEditorUploadingWidget
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

from WordAI_Publisher.tasks import generate_post_images_task
from .models import Keyword, Prompt, ModelInfo, Post, GenerationJob
from .admin_forms import CSVUploadForm, PostAdminForm

# =============== Simple FIFO queue worker (single-thread) ===============
_worker_guard = threading.Lock()
_worker_thread = None
    # Whitelist of the 8 new section image prompt keys
# Map prompt keys -> section image field names
SECTION_PROMPT_KEYS = {
    "quick_style_snapshot_image_prompt",
    "packing_essentials_checklist_image_prompt",
    "daytime_outfits_image_prompt",
    "evening_and_nightlife_image_prompt",
    "outdoor_activities_image_prompt",
    "seasonal_variations_image_prompt",
    "style_tips_for_blending_image_prompt",
    "destination_specific_extras_image_prompt",
}

PROMPT_TO_SECTION_FIELD = {
    "quick_style_snapshot_image_prompt":          "generated_quick_style_snapshot_image",
    "packing_essentials_checklist_image_prompt":  "generated_packing_essentials_checklist_image",
    "daytime_outfits_image_prompt":               "generated_daytime_outfits_image",
    "evening_and_nightlife_image_prompt":         "generated_evening_and_nightlife_image",
    "outdoor_activities_image_prompt":            "generated_outdoor_activities_image",
    "seasonal_variations_image_prompt":           "generated_seasonal_variations_image",
    "style_tips_for_blending_image_prompt":       "generated_style_tips_for_blending_image",
    "destination_specific_extras_image_prompt":   "generated_destination_specific_extras_image",
}


def _ensure_worker_running():
    """Start a single background worker to drain the queue."""
    global _worker_thread
    with _worker_guard:
        if not _worker_thread or not _worker_thread.is_alive():
            _worker_thread = threading.Thread(target=_queue_runner, daemon=True)
            _worker_thread.start()

def _queue_runner():
    """Continuously pull the oldest queued job and process it until empty."""
    while True:
        job = GenerationJob.objects.filter(status='queued').order_by('created_at').first()
        if not job:
            break
        _run_generation_job(job.id)


# =============== LLM helpers ===============
def _gpt_client():
    return openai.OpenAI(api_key=settings.OPENAI_API_KEY)

def gpt_content(client, system_prompt, user_prompt):
    if not user_prompt:
        return ''
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt or ""},
            {"role": "user", "content": user_prompt or ""}
        ]
    )
    return (resp.choices[0].message.content or "").strip()


# =============== parsing, uniquifying, top-up helpers ===============
def _slugify_title(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").strip().lower()
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", "-", s).strip("-")
    return s

def extract_styles_from_html(html: str):
    """
    Parse <h2>Title</h2> blocks and the content following each until the next <h2>.
    Robust to attributes and case.
    Returns a list of {"style_name": ..., "html": "<h2>..</h2>..."} preserving order.
    """
    pattern = re.compile(
        r"<h2\b[^>]*>(.*?)<\/h2>(.*?)(?=<h2\b|$)",
        flags=re.DOTALL | re.IGNORECASE
    )
    matches = pattern.findall(html or "")
    blocks = []
    for raw_title, content in matches:
        # Strip any markup inside title and unescape entities
        clean_title = re.sub(r"<[^>]+>", "", raw_title or "")
        clean_title = (clean_title or "").strip()
        if not clean_title:
            continue
        content = (content or "").strip()
        blocks.append({
            "style_name": clean_title,
            "html": f"<h2>{clean_title}</h2>{content}"
        })
    return blocks

def uniquify_titles(items):
    """
    If a style name repeats, append " (2)", " (3)", ... so dict keys stay unique.
    """
    seen = {}
    out = []
    for it in items:
        base = (it.get("style_name") or "").strip()
        if not base:
            continue
        if base in seen:
            seen[base] += 1
            new_name = f"{base} ({seen[base]})"
        else:
            seen[base] = 1
            new_name = base
        out.append({**it, "style_name": new_name})
    return out

def _titles_set(items):
    return {it["style_name"] for it in items if it.get("style_name")}

def top_up_missing_styles(client, system_prompt, base_style_prompt, have_titles, need_count):
    """
    If the model returned fewer than requested, ask for ONLY the missing count,
    excluding titles we already have.
    """
    if need_count <= 0:
        return []

    exclude_list = "\n".join(f"- {t}" for t in sorted(have_titles)) if have_titles else "- (none)"
    user_prompt = (
        f"{base_style_prompt}\n\n"
        f"IMPORTANT RULES:\n"
        f"- Do NOT repeat any of these existing style names:\n{exclude_list}\n"
        f"- Generate exactly {need_count} additional, unique hairstyles.\n"
        f"- Format: Use <h2>Title</h2> followed by paragraphs for each style. No markdown."
    )

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    more_html = (resp.choices[0].message.content or "").strip()
    return extract_styles_from_html(more_html)


def _run_generation_job(job_id: int):
    """Process a single queued job."""
    client = _gpt_client()

    job = GenerationJob.objects.select_related('keyword', 'prompt').get(pk=job_id)
    if not job.keyword or not job.prompt:
        job.status = 'failed'
        job.error = 'Keyword or Prompt missing'
        job.finished_at = timezone.now()
        job.save(update_fields=['status', 'error', 'finished_at'])
        return

    try:
        job.status = 'running'
        job.started_at = timezone.now()
        job.save(update_fields=['status', 'started_at'])

        # choose a model_info like you do today
        models_list = list(ModelInfo.objects.all())
        if not models_list:
            raise RuntimeError("No ModelInfo records available")
        model_info = random.choice(models_list)

        # --------- helpers ----------
        def fill_prompt(tmpl: str) -> str:
            if tmpl:
                return (tmpl
                        .replace('{{keyword}}', job.keyword.keyword or '')
                        .replace('{{hairstyle_name}}', job.keyword.keyword or '')
                        .replace('{{version_count}}', str(job.version_count))
                        .replace('{{template_type}}', str(getattr(job, 'template_type', '') or ''))
                        .replace('{{season}}',        str(getattr(job, 'season', '') or ''))
                        .replace('{{year}}',          str(getattr(job, 'year', '') or '')))
            return ''

        def ensure_h2(title: str, html: str) -> str:
            if "<h2" in (html or "").lower():
                return html
            return f"<h2>{title}</h2>\n{html or ''}"

        # System prompts for different section types
        PROSE_SYS = (
            "Return raw HTML only. Never use Markdown or code fences. "
            "You are a travel and fashion editor. Return clean HTML only. "
            "Use <h2> for the exact section title and <p> paragraphs for content. "
            "No markdown."
        )
        CHECKLIST_SYS = (
            "Return raw HTML only. Never use Markdown or code fences. "
            "You are a travel and fashion editor. Return clean HTML only. "
            "Use <h2> for the exact section title and an unordered checklist using <ul><li> items. "
            "No markdown."
        )

        def gen_section(title: str, user_prompt: str, mode: str = "prose") -> str:
            if not user_prompt or not user_prompt.strip():
                return ""
            sys = CHECKLIST_SYS if mode == "checklist" else PROSE_SYS
            body = gpt_content(client, sys, user_prompt) or ""
            return ensure_h2(title, body)

        pr = job.prompt

        # ---------------------------------------------------------------------
        # INDIVIDUAL FLOW
        # ---------------------------------------------------------------------
        if job.prompt_type == 'individual':

            # =================== MODULAR BRANCH USING 8 PROMPTS ===================
            if str(getattr(job, 'template_type', '')).lower() == 'modular':
                # 1) Read and fill all eight prompts
                p_quick   = fill_prompt(getattr(pr, 'quick_style_snapshot_prompt', ''))
                p_pack    = fill_prompt(getattr(pr, 'packing_essentials_checklist_prompt', ''))
                p_day     = fill_prompt(getattr(pr, 'daytime_outfits_prompt', ''))
                p_eve     = fill_prompt(getattr(pr, 'evening_and_nightlife_prompt', ''))
                p_outdoor = fill_prompt(getattr(pr, 'outdoor_activities_prompt', ''))
                p_season  = fill_prompt(getattr(pr, 'seasonal_variations_prompt', ''))
                p_blend   = fill_prompt(getattr(pr, 'style_tips_for_blending_prompt', ''))
                p_extra   = fill_prompt(getattr(pr, 'destination_specific_extras_prompt', ''))

                # 2) Generate each section
                sec_quick  = gen_section("Quick Style Snapshot", p_quick, mode="prose")
                sec_pack   = gen_section("Packing Essentials Checklist", p_pack, mode="checklist")
                sec_day    = gen_section("Daytime Outfits", p_day, mode="prose")
                sec_eve    = gen_section("Evening and Nightlife", p_eve, mode="prose")
                sec_out    = gen_section("Outdoor Activities", p_outdoor, mode="prose")
                sec_season = gen_section("Seasonal Variations", p_season, mode="prose")
                sec_blend  = gen_section("Style Tips for Blending", p_blend, mode="prose")
                sec_extra  = gen_section("Destination Specific Extras", p_extra, mode="prose")

                # 3) Assemble in the intended order
                modular_html = "\n\n".join([
                    s for s in [
                        sec_quick, sec_pack, sec_day, sec_eve,
                        sec_out, sec_season, sec_blend, sec_extra
                    ] if s
                ])

                # 4) You can still generate the usual title/intro/conclusion/meta
                title_prompt      = fill_prompt(getattr(pr, 'title_prompt', ''))
                intro_prompt      = fill_prompt(getattr(pr, 'intro_prompt', ''))
                conclusion_prompt = fill_prompt(getattr(pr, 'conclusion_prompt', ''))
                meta_data_prompt  = fill_prompt(getattr(pr, 'meta_data_prompt', ''))

                generated_title = gpt_content(client, "", title_prompt) or ""
                generated_title = generated_title.strip().strip('"').strip("'")

                generated_intro = gpt_content(
                    client,
                    "You are a helpful assistant that generates engaging article introductions. Do not use markdown.",
                    intro_prompt
                ) or ""

                generated_conclusion = gpt_content(
                    client,
                    "You are a helpful assistant that generates article conclusions. Do not use markdown.",
                    conclusion_prompt
                ) or ""

                meta_title = ''
                meta_description = ''
                if meta_data_prompt:
                    meta_json = gpt_content(client, "Return JSON with meta_title and meta_description.", meta_data_prompt)
                    try:
                        meta_data = json.loads(meta_json or "{}")
                        meta_title = meta_data.get('meta_title', '')
                        meta_description = meta_data.get('meta_description', '')
                    except Exception:
                        meta_title = ''
                        meta_description = meta_json or ''

                # 5) Save. Reuse generated_style_section to store the modular HTML,
                #    or add a new column if you prefer (generated_packing_guide).
                post = Post.objects.create(
                    keyword=job.keyword,
                    prompt=pr,
                    model_info=model_info,
                    version=job.version_count,
                    generated_title=generated_title,
                    generated_intro=generated_intro,
                    # ── Modular sections mapped 1:1 ──
                    generated_quick_style_snapshot=sec_quick or "",
                    generated_packing_essentials_checklist=sec_pack or "",
                    generated_daytime_outfits=sec_day or "",
                    generated_evening_and_nightlife=sec_eve or "",
                    generated_outdoor_activities=sec_out or "",
                    generated_seasonal_variations=sec_season or "",
                    generated_style_tips_for_blending=sec_blend or "",
                    generated_destination_specific_extras=sec_extra or "",
                    generated_conclusion=generated_conclusion,
                    meta_title=meta_title,
                    meta_description=meta_description,
                    status='draft',
                    content_generated='completed',
                    featured_image_status='in_process',
                    style_images_status='Not generated',
                    modular_images_status = 'in_process'

                )
                post.save()
                threading.Thread(target=generate_post_images_task, args=(post.id,), daemon=True).start()

            # =================== NON-MODULAR INDIVIDUAL FLOW (UNCHANGED) ===================
            else:
                title_prompt = fill_prompt(getattr(pr, 'title_prompt', ''))
                intro_prompt = fill_prompt(getattr(pr, 'intro_prompt', ''))
                style_prompt = fill_prompt(getattr(pr, 'style_prompt', ''))
                conclusion_prompt = fill_prompt(getattr(pr, 'conclusion_prompt', ''))
                meta_data_prompt = fill_prompt(getattr(pr, 'meta_data_prompt', ''))

                generated_title = gpt_content(client, "", title_prompt)
                if generated_title:
                    generated_title = generated_title.strip().strip('"').strip("'")

                generated_intro = gpt_content(
                    client,
                    "You are a helpful assistant that generates engaging article introductions. Do not use markdown.",
                    intro_prompt
                )

                style_sys = (
                    "You are a helpful assistant that generates detailed style descriptions. "
                    "Use <h2> HTML headings for each style section, followed by paragraphs. "
                    "Do not use markdown."
                )
                generated_style_section = gpt_content(
                    client,
                    style_sys + f" Generate exactly {job.version_count} unique hairstyles.",
                    style_prompt
                )

                style_blocks = extract_styles_from_html(generated_style_section)
                missing = job.version_count - len(style_blocks)
                if missing > 0:
                    more_blocks = top_up_missing_styles(
                        client=client,
                        system_prompt=style_sys,
                        base_style_prompt=style_prompt,
                        have_titles=_titles_set(style_blocks),
                        need_count=missing
                    )
                    style_blocks.extend(more_blocks)

                style_blocks = uniquify_titles(style_blocks)
                if len(style_blocks) > job.version_count:
                    style_blocks = style_blocks[:job.version_count]

                style_image_descriptions = []
                for block in style_blocks:
                    style_name = block["style_name"]
                    image_desc = gpt_content(
                        client,
                        "You are an editorial stylist creating image descriptions for a fashion AI. "
                        "Write a visual description of the haircut below in 35–60 words. "
                        "Include hair length, texture, shape, sides, top, and camera angle.",
                        f"Hairstyle: {style_name}"
                    )
                    style_image_descriptions.append({
                        "style_name": style_name,
                        "image_style_description": (image_desc or "").strip()
                    })
                style_dict = {item["style_name"]: item["image_style_description"] for item in style_image_descriptions}
                generated_style_section = "\n\n".join(b["html"] for b in style_blocks)

                generated_conclusion = gpt_content(
                    client,
                    "You are a helpful assistant that generates article conclusions. Do not use markdown.",
                    conclusion_prompt
                )

                meta_title = ''
                meta_description = ''
                if meta_data_prompt:
                    meta_json = gpt_content(client, "Return JSON with meta_title and meta_description.", meta_data_prompt)
                    try:
                        meta_data = json.loads(meta_json)
                        meta_title = meta_data.get('meta_title', '')
                        meta_description = meta_data.get('meta_description', '')
                    except Exception:
                        meta_title = ''
                        meta_description = meta_json

                post = Post.objects.create(
                    keyword=job.keyword,
                    prompt=pr,
                    model_info=model_info,
                    version=job.version_count,
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
                    style_image_descriptions=style_dict
                )
                post.save()
                threading.Thread(target=generate_post_images_task, args=(post.id,), daemon=True).start()

        # ---------------------------------------------------------------------
        # MASTER PROMPT FLOW (UNCHANGED)
        # ---------------------------------------------------------------------
        else:
            template_type = str(getattr(job, 'template_type', '') or '').strip().lower()

            # ========= SPECIAL CASE: MASTER + FOOD LOVE =========
            if template_type in ('food love', 'food_love', 'foodlove'):
                # 1) Generate full HTML from the master prompt
                foodlove_sys = (
                    "Return raw HTML only. Never use Markdown or code fences. "
                    "You are an experienced food editor writing for a modern lifestyle site. "
                    "Produce a complete article in clean HTML with semantic structure: "
                    "<h1> (optional), <h2> section headings, <p> paragraphs, and lists where useful. "
                    "Do not include scripts, styles, or external assets."
                )
                master_prompt_text = fill_prompt(getattr(pr, 'master_prompt', ''))
                full_html = gpt_content(client, foodlove_sys, master_prompt_text) or ""

                # 2) Meta data (reuse the SAME feature/path used elsewhere)
                meta_title = ''
                meta_description = ''
                meta_data_prompt = fill_prompt(getattr(pr, 'meta_data_prompt', ''))
                if meta_data_prompt:
                    meta_json = gpt_content(
                        client,
                        "Return JSON with meta_title and meta_description.",
                        meta_data_prompt
                    )
                    try:
                        meta_data = json.loads(meta_json or "{}")
                        meta_title = meta_data.get('meta_title', '') or ''
                        meta_description = meta_data.get('meta_description', '') or ''
                    except Exception:
                        # Fallback: if model returned plain text, keep it in description
                        meta_title = ''
                        meta_description = (meta_json or '').strip()

                # 3) (Optional) Try to pull a title from the HTML's first <h1> if present
                #    This is safe and doesn't conflict with meta fields.
                def _extract_h1(html: str) -> str:
                    import re
                    m = re.search(r'<h1[^>]*>(.*?)</h1>', html, flags=re.I | re.S)
                    return (m.group(1).strip() if m else '')

                generated_title = _extract_h1(full_html)

                # 4) Create post (NO images)
                post = Post.objects.create(
                    keyword=job.keyword,
                    prompt=pr,
                    model_info=model_info,
                    version=job.version_count,
                    generated_title=generated_title,
                    generated_intro="",                       # not needed; content is in full HTML
                    generated_style_section=full_html,        # store the whole HTML here
                    generated_conclusion="",                  # not needed; already in HTML
                    meta_title=meta_title,
                    meta_description=meta_description,
                    status='draft',
                    content_generated='completed',
                    featured_image_status='Not generated',
                    style_images_status='Not generated',
                    # If your model has modular_images_status, keep it consistent:
                    # modular_images_status='Not generated'
                )
                post.save()

                # 5) Finish job WITHOUT launching image thread
                job.post = post
                job.status = 'done'
                job.finished_at = timezone.now()
                job.save(update_fields=['post', 'status', 'finished_at'])
                return

            # ========= DEFAULT MASTER (existing JSON flow) =========
            system_prompt = (
                "You are a professional SEO and content writer who returns STRICT JSON output for a blog article."
                " Always return exactly this JSON structure: "
                "{"
                "  \"meta_title\": \"string\","
                "  \"meta_description\": \"string\","
                "  \"title\": \"string\","
                "  \"introduction\": [\"string\", ...],"
                "  \"styles\": [\"string\", ...],"
                "  \"final_thoughts\": [\"string\", ...]"
                " }"
                " Important JSON rules:"
                "- Use only double quotes (\"\") for JSON strings."
                "- NEVER add markdown code fences."
                "- Do not add any markdown or text outside the JSON object."
                "- Arrays must contain only flat strings."
            )
            master_prompt_text = fill_prompt(getattr(pr, 'master_prompt', ''))
            json_text = gpt_content(_gpt_client(), system_prompt, master_prompt_text)
            structured = json.loads(json_text)

            post = Post.objects.create(
                keyword=job.keyword,
                prompt=pr,
                model_info=model_info,
                version=job.version_count,
                generated_title=structured.get("title", ""),
                generated_intro="".join(structured.get("introduction", [])),
                generated_style_section="".join(structured.get("styles", [])),
                generated_conclusion="".join(structured.get("final_thoughts", [])),
                meta_title=structured.get("meta_title", ""),
                meta_description=structured.get("meta_description", ""),
                status='draft',
                content_generated='completed',
                featured_image_status='in_process',
                style_images_status='in_process',
            )
            post.save()
            threading.Thread(target=generate_post_images_task, args=(post.id,), daemon=True).start()

        # Done
        job.post = post
        job.status = 'done'
        job.finished_at = timezone.now()
        job.save(update_fields=['post', 'status', 'finished_at'])

    except Exception as e:
        job.status = 'failed'
        job.error = str(e)
        job.finished_at = timezone.now()
        job.save(update_fields=['status', 'error', 'finished_at'])

# =============================================================================
# S3/Storage + WordPress Helpers
# =============================================================================

def _wp_media_endpoint():
    """
    Returns the WP media endpoint. Uses settings.WORDPRESS_API_URL_MEDIA if set.
    Otherwise derives it from WORDPRESS_API_URL (.../wp-json/wp/v2/posts -> .../media).
    """
    media_url = getattr(settings, "WORDPRESS_API_URL_MEDIA", None)
    if media_url:
        return media_url
    base = settings.WORDPRESS_API_URL.rsplit('/posts', 1)[0]
    return f"{base}/media"

def _wp_headers():
    # IMPORTANT: do NOT set Content-Type for file uploads; requests will set multipart/form-data.
    return {
        "Authorization": f"Bearer {settings.WORDPRESS_JWT_TOKEN}",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0",
    }

def _guess_ct(filename: str) -> str:
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"

def _open_from_storage_or_url(name_or_url: str):
    """
    Returns (fileobj, filename) where fileobj is a readable binary stream.
    Supports:
      - Absolute http(s) URLs (public or signed S3)
      - MEDIA_URL-prefixed URLs
      - Storage-relative keys (private S3 or other backends)
    """
    if not name_or_url:
        raise ValueError("Empty image path/url")

    # Absolute URL
    if isinstance(name_or_url, str) and name_or_url.startswith(("http://", "https://")):
        resp = requests.get(name_or_url, stream=True, timeout=60)
        resp.raise_for_status()
        filename = os.path.basename(urlparse(name_or_url).path) or "upload.bin"
        bio = io.BytesIO(resp.content)
        bio.seek(0)
        return bio, filename

    # MEDIA_URL-prefixed URL or storage key
    storage_key = str(name_or_url).lstrip("/")
    if getattr(settings, "MEDIA_URL", None):
        media_path = urlparse(settings.MEDIA_URL).path
        url_path = urlparse(storage_key).path
        if url_path.startswith(media_path):
            storage_key = url_path[len(media_path):].lstrip("/")

    fobj = default_storage.open(storage_key, "rb")
    filename = os.path.basename(storage_key) or "upload.bin"
    return fobj, filename
# =============================================================================
# Admin: ModelInfo
# =============================================================================

@admin.register(ModelInfo)
class ModelInfoAdmin(admin.ModelAdmin):
    list_display = (
        'model_id', 'ethnicity', 'skin_tone', 'hair_texture', 'face_shape',
        'tshirt', 'eye_color', 'hair_color', 'created_at'
    )
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
                        face_shape=row.get('face_shape', ''),
                        tshirt=row.get('tshirt', ''),
                        eye_color=row.get('eye_color', ''),
                        hair_color=row.get('hair_color', ''),
                        build_description=row.get('build_description', ''),
                        expression_description=row.get('expression_description', ''),
                        wardrobe_color=row.get('wardrobe_color', ''),
                        wardrobe_item=row.get('wardrobe_item', ''),
                        grooming_description=row.get('grooming_description', ''),
                        brand=row.get('brand', ''),
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


# =============================================================================
# Admin: Prompt
# =============================================================================

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
            'fields': (
                'master_prompt', 'title_prompt', 'intro_prompt', 'style_prompt',

                'quick_style_snapshot_prompt', 'quick_style_snapshot_image_prompt',
                'daytime_outfits_prompt', 'daytime_outfits_image_prompt',
                'evening_and_nightlife_prompt', 'evening_and_nightlife_image_prompt',
                'outdoor_activities_prompt', 'outdoor_activities_image_prompt',
                'seasonal_variations_prompt', 'seasonal_variations_image_prompt',
                'packing_essentials_checklist_prompt', 'packing_essentials_checklist_image_prompt',
                'style_tips_for_blending_prompt', 'style_tips_for_blending_image_prompt',
                'destination_specific_extras_prompt', 'destination_specific_extras_image_prompt',

                'conclusion_prompt', 'meta_data_prompt',
                'featured_image_prompt', 'image_prompt',
            ),
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


# =============================================================================
# Forms
# =============================================================================

class KeywordAdminForm(forms.ModelForm):
    class Meta:
        model = Keyword
        fields = '__all__'


# =============================================================================
# Admin: Keyword
# =============================================================================

@admin.register(Keyword)
class KeywordAdmin(admin.ModelAdmin):
    form = KeywordAdminForm
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
            path('ajax-regenerate-single-style/', self.admin_site.admin_view(self.ajax_regenerate_single_style), name='ajax_regenerate_single_style'),
            path('save-style-prompt/', self.admin_site.admin_view(self.save_style_prompt), name='save_style_prompt'),
            path('ajax-bulk-generate/', self.admin_site.admin_view(self.ajax_bulk_generate), name='ajax_bulk_generate'),
        ]
        return custom_urls + urls


    def changelist_view(self, request, extra_context=None):
        extra_context = extra_context or {}
        # for the template’s dropdown
        extra_context['prompts'] = Prompt.objects.all().order_by('prompt_id')
        # (optional) if your JS builds options from JSON
        extra_context['prompts_json'] = json.dumps(
            list(Prompt.objects.values('id', 'prompt_id', 'title_prompt').order_by('prompt_id'))
        )
        return super().changelist_view(request, extra_context=extra_context)

    # ---------------------------
    # CSV Upload (Keywords)
    # ---------------------------
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

    # ---------------------------
    # UI Helpers & AJAX Handlers
    # ---------------------------
    def generate_content_button(self, obj):
        return format_html(
            '<a class="button" href="{}">Generate Content</a>&nbsp;',
            reverse('admin:generate_content', args=[obj.pk])
        )
    generate_content_button.short_description = "Actions"
    generate_content_button.allow_tags = True
    
    def ajax_bulk_generate(self, request):
        """
        Enqueue bulk generation jobs instead of running them inline.
        Expects JSON: { 
            "keyword_ids":[...], 
            "prompt_id":"...", 
            "prompt_type":"individual|master", 
            "version_count":N,
            "template_type":"regular|modular",
            "season":"spring|summer|fall|winter",
            "year":YYYY
        }
        Returns an immediate thank-you message with current time and job ids.
        """
        if request.method != 'POST':
            return JsonResponse({'success': False, 'message': 'Invalid method'}, status=405)

        # Parse JSON body (or fallback to form-encoded)
        try:
            payload = json.loads(request.body.decode() or "{}")
        except Exception:
            return JsonResponse({'success': False, 'message': 'Bad JSON'}, status=400)

        keyword_ids = payload.get('keyword_ids') or request.POST.getlist('keyword_ids[]') or []
        prompt_id = (payload.get('prompt_id') or request.POST.get('prompt_id') or '').strip()
        prompt_type = (payload.get('prompt_type') or request.POST.get('prompt_type') or 'individual').strip().lower()
        version_count = int(payload.get('version_count') or request.POST.get('version_count') or 1)
        template_type = (payload.get('template_type') or request.POST.get('template_type') or 'regular').strip().lower()
        season = (payload.get('season') or request.POST.get('season') or '').strip().lower() or None
        year = payload.get('year') or request.POST.get('year')

        # ---- Validation ----
        if not keyword_ids:
            return JsonResponse({'success': False, 'message': 'Missing keyword_ids'}, status=400)
        if not prompt_id:
            return JsonResponse({'success': False, 'message': 'Missing prompt_id'}, status=400)
        if prompt_type not in ('individual', 'master'):
            return JsonResponse({'success': False, 'message': 'prompt_type must be \"individual\" or \"master\"'}, status=400)
        if template_type not in ('regular', 'modular'):
            return JsonResponse({'success': False, 'message': 'template_type must be \"regular\" or \"modular\"'}, status=400)
        if version_count < 1:
            return JsonResponse({'success': False, 'message': 'version_count must be >= 1'}, status=400)

        # if template_type == 'modular':
        #     if not season:
        #         return JsonResponse({'success': False, 'message': 'season is required for modular template'}, status=400)
        #     if year:
        #         try:
        #             year = int(year)
        #         except (TypeError, ValueError):
        #             return JsonResponse({'success': False, 'message': 'year must be an integer'}, status=400)

        # Resolve prompt
        pr = Prompt.objects.filter(prompt_id=prompt_id).first()
        if not pr:
            return JsonResponse({'success': False, 'message': 'Prompt not found'}, status=404)

        created_jobs, errors = [], []

        for kid in keyword_ids:
            try:
                kw = Keyword.objects.get(pk=kid)
            except Keyword.DoesNotExist:
                errors.append({'id': kid, 'error': 'Keyword not found'})
                continue

            job = GenerationJob.objects.create(
                keyword=kw,
                prompt=pr,
                prompt_type=prompt_type,
                version_count=version_count,
                template_type=template_type,
                season=season,
                year=year,
                created_by=request.user if getattr(request, "user", None) and request.user.is_authenticated else None,
                status='queued',
            )
            created_jobs.append(job.id)

        # Start/ensure the single FIFO worker is running
        _ensure_worker_running()

        return JsonResponse({
            'success': True,
            'queued': True,
            'job_ids': created_jobs,
            'errors': errors,
            'message': f"✅ Thanks! {len(created_jobs)} request(s) queued at {timezone.now().strftime('%Y-%m-%d %H:%M:%S')} and will run one by one."
        }, status=200)


    def save_style_prompt(self, request):
        """Save a style/section/featured prompt (JSON body)."""
        if request.method != "POST":
            return HttpResponseBadRequest("Only POST requests allowed.")

        # Parse JSON safely
        try:
            if not request.body:
                return JsonResponse({"success": False, "error": "Empty body"}, status=400)
            data = json.loads(request.body.decode() or "{}")
        except Exception as e:
            return JsonResponse({"success": False, "error": f"Invalid JSON: {e}"}, status=400)

        style_name = (data.get("style_name") or "").strip()
        content    = data.get("content")
        object_id  = data.get("object_id")

        if not style_name or content is None or not object_id:
            return JsonResponse({"success": False, "error": "Missing one or more required fields."}, status=400)

        obj = get_object_or_404(Post, pk=object_id)

        try:
            # 1) Featured image prompt (simple text field)
            if style_name == "featured_image":
                obj.featured_prompt_text = content
                obj.save(update_fields=["featured_prompt_text"])
                return JsonResponse({"success": True})

            # 2) NEW: 8 modular section image prompts -> save to extra_image_used_prompts (keyed by section field)
            if style_name in SECTION_PROMPT_KEYS:
                section_field = PROMPT_TO_SECTION_FIELD.get(style_name)
                if not section_field:
                    return JsonResponse({"success": False, "error": "Unknown section prompt key."}, status=400)

                current = obj.extra_image_used_prompts or {}
                if not isinstance(current, dict):
                    current = {}

                # Store as plain string per your requirement:
                #   { "generated_daytime_outfits_image": "the filled prompt" }
                current[section_field] = content
                obj.extra_image_used_prompts = current
                obj.save(update_fields=["extra_image_used_prompts"])
                return JsonResponse({"success": True})

            # 3) Default: legacy/style images behavior (kept as-is)
            prompts = obj.style_prompts or {}
            if not isinstance(prompts, dict):
                prompts = {}
            prompts[style_name] = content
            obj.style_prompts = prompts
            obj.save(update_fields=["style_prompts"])
            return JsonResponse({"success": True})

        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)}, status=500)



    def ajax_regenerate_single_style(self, request):
        if request.method == 'POST':
            data = json.loads(request.body.decode())
            post_id = data.get('post_id')
            style_name = data.get('style_name')

            post = get_object_or_404(Post, pk=post_id)
            post.style_images_status = 'in_process'
            post.save()

            threading.Thread(
                target=generate_post_images_task,
                args=(post.id,),
                kwargs={'only_style': True, 'specific_style': style_name}
            ).start()

            return JsonResponse({'success': True, 'message': f'Style "{style_name}" regeneration started.'})
        return JsonResponse({'success': False, 'message': 'Invalid request.'})

    def ajax_generate_versions(self, request):
        """
        Create a GenerationJob for a single keyword/prompt selection.

        Updates:
        - Normalizes `template_type` to a canonical slug: 'regular' | 'modular' | 'food_love'
        - Validates against the normalized set to avoid case/value mismatches ('Food Love' vs 'food_love')
        - Leaves the rest of the logic unchanged
        """
        if request.method != 'POST':
            return JsonResponse({'success': False, 'message': 'Invalid request method.'}, status=405)

        # ---- Helpers ----
        def _normalize_template_type(val: str) -> str:
            v = (val or '').strip().lower()
            # collapse separators to space
            v = v.replace('_', ' ').replace('-', ' ')
            if v in ('food love', 'foodlove'):
                return 'food_love'
            if v == 'modular':
                return 'modular'
            if v == 'regular':
                return 'regular'
            return v  # unknown

        # ---- Parse body (JSON or form) ----
        keyword_id = prompt_id = prompt_type = None
        version_count = None
        template_type = season = year = None

        try:
            if request.content_type and request.content_type.startswith('application/json'):
                raw = (request.body or b'').decode('utf-8') or '{}'
                data = json.loads(raw)

                keyword_id    = data.get('keyword_id')
                prompt_id     = (data.get('prompt_id') or '').strip()
                prompt_type   = (data.get('prompt_type') or 'individual').strip().lower()
                version_count = int(data.get('version_count') or 1)
                template_type = _normalize_template_type(data.get('template_type') or 'regular')
                season        = (data.get('season') or '').strip().lower() or None
                year          = data.get('year')
            else:
                keyword_id    = request.POST.get('keyword_id')
                prompt_id     = (request.POST.get('prompt_id') or '').strip()
                prompt_type   = (request.POST.get('prompt_type') or 'individual').strip().lower()
                version_count = int(request.POST.get('version_count') or 1)
                template_type = _normalize_template_type(request.POST.get('template_type') or 'regular')
                season        = (request.POST.get('season') or '').strip().lower() or None
                year          = request.POST.get('year')
        except (ValueError, json.JSONDecodeError) as e:
            return JsonResponse({'success': False, 'message': f'Bad request body: {e}'}, status=400)

        # ---- Validate fields early ----
        if not keyword_id:
            return JsonResponse({'success': False, 'message': 'Missing keyword_id'}, status=400)
        try:
            keyword_id = int(keyword_id)
        except (TypeError, ValueError):
            return JsonResponse({'success': False, 'message': 'keyword_id must be an integer'}, status=400)

        if not prompt_id:
            return JsonResponse({'success': False, 'message': 'Missing prompt_id'}, status=400)

        if prompt_type not in ('individual', 'master'):
            return JsonResponse({'success': False, 'message': 'prompt_type must be "individual" or "master"'}, status=400)

        if version_count < 1:
            return JsonResponse({'success': False, 'message': 'version_count must be >= 1'}, status=400)

        # Accept normalized values only
        allowed_templates = ('regular', 'modular', 'food_love')
        if template_type not in allowed_templates:
            return JsonResponse(
                {
                    'success': False,
                    'message': 'template_type must be one of: "regular", "modular", "food_love"'
                },
                status=400
            )

        # If you ever need seasonal gating for modular, uncomment this block:
        # if template_type == 'modular':
        #     if not season:
        #         return JsonResponse({'success': False, 'message': 'season is required for modular template'}, status=400)
        #     if year:
        #         try:
        #             year = int(year)
        #         except (TypeError, ValueError):
        #             return JsonResponse({'success': False, 'message': 'year must be an integer'}, status=400)

        # ---- Lookups ----
        keyword = get_object_or_404(Keyword, pk=keyword_id)
        prompt = get_object_or_404(Prompt, prompt_id=prompt_id)

        # ---- Create job ----
        job = GenerationJob.objects.create(
            keyword=keyword,
            prompt=prompt,
            prompt_type=prompt_type,
            version_count=version_count,
            template_type=template_type,  # canonical slug now
            season=season,
            year=year,
            created_by=request.user if request.user.is_authenticated else None,
            status='queued',
        )

        _ensure_worker_running()

        return JsonResponse({
            'success': True,
            'queued': True,
            'job_id': job.id,
            'message': f"✅ Thanks! Your request was queued at {timezone.now().strftime('%Y-%m-%d %H:%M:%S')}."
        }, status=200)

    def ajax_regenerate_versions(self, request):
        if request.method == 'POST':
            data = json.loads(request.body.decode())
            post_id = data.get('post_id')
            content_type = data.get('content_type')
            job = GenerationJob.objects.select_related('keyword', 'prompt').get(post_id=post_id)
            post = get_object_or_404(Post, pk=post_id)
            keyword = post.keyword
            prompt = post.prompt
            model_info = post.model_info
            version_count = post.version
            client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

            def fill_prompt(tmpl):
                if tmpl:
                    return (tmpl
                            .replace('{{keyword}}', keyword.keyword)
                            .replace('{{hairstyle_name}}', keyword.keyword)
                            .replace('{{version_count}}', str(version_count))
                            .replace('{{template_type}}', str(getattr(job, 'template_type', '') or ''))
                            .replace('{{season}}',        str(getattr(job, 'season', '') or ''))
                            .replace('{{year}}',          str(getattr(job, 'year', '') or '')))
 
                return ''

            def extract_styles_from_html(html):
                pattern = r"<h2>(.*?)<\/h2>(.*?)(?=<h2>|$)"
                matches = re.findall(pattern, html, re.DOTALL)
                return [{"style_name": title.strip(), "html": f"<h2>{title.strip()}</h2>{content.strip()}"} for title, content in matches]

            def gpt_content(system_prompt, user_prompt):
                if not user_prompt:
                    return ''
                response = client.chat.completions.create(
                    model="gpt-4o",
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
                return JsonResponse({'success': True, 'message': 'Title regenerated.', 'content': post.generated_title})

            elif content_type == 'intro':
                content = gpt_content(
                    "you are a helpful assistant that generates engaging article introductions. Do not use markdown.",
                    fill_prompt(prompt.intro_prompt)
                )
                post.generated_intro = content
                post.save()
                return JsonResponse({'success': True, 'message': 'Intro regenerated.', 'content': post.generated_intro})
            elif content_type == 'generated_quick_style_snapshot':
                content = gpt_content(
                    "Return raw HTML only. Never use Markdown or code fences. ",
                    fill_prompt(getattr(prompt, 'quick_style_snapshot_prompt', ''))
                )
                post.generated_quick_style_snapshot = content
                post.save(update_fields=['generated_quick_style_snapshot'])
                return JsonResponse({'success': True, 'message': 'Quick Style Snapshot regenerated.', 'content': post.generated_quick_style_snapshot})

            elif content_type == 'generated_packing_essentials_checklist':
                content = gpt_content(
                    "Return raw HTML only. Never use Markdown or code fences. ",
                    fill_prompt(getattr(prompt, 'packing_essentials_checklist_prompt', ''))
                )
                post.generated_packing_essentials_checklist = content
                post.save(update_fields=['generated_packing_essentials_checklist'])
                return JsonResponse({'success': True, 'message': 'Packing Essentials Checklist regenerated.', 'content': post.generated_packing_essentials_checklist})

            elif content_type == 'generated_daytime_outfits':
                content = gpt_content(
                    "Return raw HTML only. Never use Markdown or code fences. ",
                    fill_prompt(getattr(prompt, 'daytime_outfits_prompt', ''))
                )
                post.generated_daytime_outfits = content
                post.save(update_fields=['generated_daytime_outfits'])
                return JsonResponse({'success': True, 'message': 'Daytime Outfits regenerated.', 'content': post.generated_daytime_outfits})

            elif content_type == 'generated_evening_and_nightlife':
                content = gpt_content(
                    "Return raw HTML only. Never use Markdown or code fences. ",
                    fill_prompt(getattr(prompt, 'evening_and_nightlife_prompt', ''))
                )
                post.generated_evening_and_nightlife = content
                post.save(update_fields=['generated_evening_and_nightlife'])
                return JsonResponse({'success': True, 'message': 'Evening and Nightlife regenerated.', 'content': post.generated_evening_and_nightlife})

            elif content_type == 'generated_outdoor_activities':
                content = gpt_content(
                    "Return raw HTML only. Never use Markdown or code fences. ",
                    fill_prompt(getattr(prompt, 'outdoor_activities_prompt', ''))
                )
                post.generated_outdoor_activities = content
                post.save(update_fields=['generated_outdoor_activities'])
                return JsonResponse({'success': True, 'message': 'Outdoor Activities regenerated.', 'content': post.generated_outdoor_activities})

            elif content_type == 'generated_seasonal_variations':
                content = gpt_content(
                    "Return raw HTML only. Never use Markdown or code fences. ",
                    fill_prompt(getattr(prompt, 'seasonal_variations_prompt', ''))
                )
                post.generated_seasonal_variations = content
                post.save(update_fields=['generated_seasonal_variations'])
                return JsonResponse({'success': True, 'message': 'Seasonal Variations regenerated.', 'content': post.generated_seasonal_variations})

            elif content_type == 'generated_style_tips_for_blending':
                content = gpt_content(
                    "Return raw HTML only. Never use Markdown or code fences. ",
                    fill_prompt(getattr(prompt, 'style_tips_for_blending_prompt', ''))
                )
                post.generated_style_tips_for_blending = content
                post.save(update_fields=['generated_style_tips_for_blending'])
                return JsonResponse({'success': True, 'message': 'Style Tips for Blending regenerated.', 'content': post.generated_style_tips_for_blending})

            elif content_type == 'generated_destination_specific_extras':
                content = gpt_content(
                   "Return raw HTML only. Never use Markdown or code fences. ",
                    fill_prompt(getattr(prompt, 'destination_specific_extras_prompt', ''))
                )
                post.generated_destination_specific_extras = content
                post.save(update_fields=['generated_destination_specific_extras'])
                return JsonResponse({'success': True, 'message': 'Destination Specific Extras regenerated.', 'content': post.generated_destination_specific_extras})

            elif content_type == 'style_section':
                # --- Detect template type from latest GenerationJob ---
                gj = (
                    GenerationJob.objects
                    .filter(post_id=post.id)
                    .order_by('-finished_at', '-started_at', '-id')
                    .values('template_type')
                    .first()
                )
                tmpl = ((gj or {}).get('template_type') or (getattr(post, 'template_type', '') or '')).strip().lower()
                tmpl_norm = tmpl.replace('_', ' ').replace('-', ' ')
                is_food_love = (tmpl_norm in ('food love', 'foodlove'))

                # --- Default Food Love system prompt ---
                foodlove_sys = (
                    "Return raw HTML only. Never use Markdown or code fences. "
                    "You are an experienced food editor writing for a modern lifestyle site. "
                    "Produce a complete article in clean HTML with semantic structure: "
                    "<h1> (optional), <h2> section headings, <p> paragraphs, and lists where useful. "
                    "Do not include scripts, styles, or external assets."
                )

                # --- Generate content ---
                if not is_food_love:
                    # =========================
                    # NORMAL BEHAVIOR (existing)
                    # =========================
                    content = gpt_content(
                        (
                            "You are a helpful assistant that generates detailed style descriptions. "
                            "Use <h2> HTML headings for each style section, followed by paragraphs. "
                            f"Do not use markdown. Generate exactly {version_count} unique hairstyles."
                        ),
                        fill_prompt(prompt.style_prompt)
                    ) or ""

                    style_blocks = extract_styles_from_html(content)
                    style_image_descriptions = []
                    for block in style_blocks:
                        style_name = block.get("style_name", "")
                        if not style_name:
                            continue
                        image_desc = gpt_content(
                            "You are an editorial stylist creating image descriptions for a fashion AI. "
                            "Write a visual description of the haircut below in 35–60 words. "
                            "Include hair length, texture, shape, sides, top, and camera angle.",
                            f"Hairstyle: {style_name}"
                        )
                        style_image_descriptions.append({
                            "style_name": style_name,
                            "image_style_description": (image_desc or "").strip()
                        })

                    style_dict = {item["style_name"]: item["image_style_description"] for item in style_image_descriptions}

                    post.style_image_descriptions = style_dict
                    post.generated_style_section = content
                    post.style_images_status = 'in_process'
                    post.style_images = {}
                    post.style_prompts = None
                    post.save()

                    threading.Thread(
                        target=generate_post_images_task,
                        args=(post.id,),
                        kwargs={'only_style': True},
                        daemon=True
                    ).start()

                    return JsonResponse({'success': True, 'message': 'Style section regenerated.', 'content': post.generated_style_section})

                else:
                    # =========================
                    # FOOD LOVE BEHAVIOR
                    # =========================
                    content = gpt_content(
                        foodlove_sys,
                        fill_prompt(prompt.master_prompt or prompt.style_prompt)
                    ) or ""

                    post.generated_style_section = content
                    post.style_image_descriptions = {}
                    post.style_images_status = 'Not generated'
                    post.style_images = {}
                    post.style_prompts = None
                    post.save()

                    return JsonResponse({
                        'success': True,
                        'message': 'Style section regenerated (Food Love: HTML only, no images).',
                        'content': post.generated_style_section
                    })

            elif content_type == 'conclusion':
                content = gpt_content(
                    "You are a helpful assistant that generates article conclusions. Do not use markdown.",
                    fill_prompt(prompt.conclusion_prompt)
                )
                post.generated_conclusion = content
                post.save()
                return JsonResponse({'success': True, 'message': 'Conclusion regenerated.', 'content': post.generated_conclusion})

            elif content_type == 'meta_title':
                meta_prompt = fill_prompt(prompt.meta_data_prompt)
                meta_json = gpt_content("Return JSON with meta_title and meta_description.", meta_prompt)
                try:
                    meta_data = json.loads(meta_json)
                    post.meta_title = meta_data.get('meta_title', '')
                except Exception:
                    post.meta_title = meta_json
                post.save()
                return JsonResponse({'success': True, 'message': 'Meta title regenerated.', 'meta_title': post.meta_title})

            elif content_type == 'meta_description':
                meta_prompt = fill_prompt(prompt.meta_data_prompt)
                meta_json = gpt_content("Return JSON with meta_title and meta_description.", meta_prompt)
                try:
                    meta_data = json.loads(meta_json)
                    post.meta_description = meta_data.get('meta_description', '')
                except Exception:
                    post.meta_description = meta_json
                post.save()
                return JsonResponse({'success': True, 'message': 'Meta description regenerated.', 'meta_description': post.meta_description})

            elif content_type == 'featured_image':
                post.featured_image_status = 'in_process'
                post.save()
                threading.Thread(
                    target=generate_post_images_task,
                    args=(post.id,),
                    kwargs={'only_featured': True, 'featured_prompt_text': post.featured_prompt_text}
                ).start()
                return JsonResponse({'success': True, 'message': 'Featured image regeneration started.', 'status': post.featured_image_status})
            
            elif content_type == 'generated_quick_style_snapshot_image':
                status_field = 'modular_images_status' if hasattr(post, 'modular_images_status') else 'section_images_status'
                setattr(post, status_field, 'in_process')
                setattr(post, content_type, None)  # 👈 clear only this field
                post.save(update_fields=[content_type, status_field])
                threading.Thread(
                    target=generate_post_images_task,
                    args=(post.id,),
                    kwargs={'only_sections': True, 'sections': ['generated_quick_style_snapshot_image']},
                ).start()
                return JsonResponse({'success': True, 'message': 'Quick Style Snapshot image regeneration started.', 'status': getattr(post, status_field)})

            elif content_type == 'generated_packing_essentials_checklist_image':
                status_field = 'modular_images_status' if hasattr(post, 'modular_images_status') else 'section_images_status'
                setattr(post, status_field, 'in_process')
                setattr(post, content_type, None)  # 👈 clear only this field
                post.save(update_fields=[content_type, status_field])
                threading.Thread(
                    target=generate_post_images_task,
                    args=(post.id,),
                    kwargs={'only_sections': True, 'sections': ['generated_packing_essentials_checklist_image']},
                ).start()
                return JsonResponse({'success': True, 'message': 'Packing Essentials image regeneration started.', 'status': getattr(post, status_field)})

            elif content_type == 'generated_daytime_outfits_image':
                status_field = 'modular_images_status' if hasattr(post, 'modular_images_status') else 'section_images_status'
                setattr(post, status_field, 'in_process')
                setattr(post, content_type, None)  # 👈 clear only this field
                post.save(update_fields=[content_type, status_field])
                threading.Thread(
                    target=generate_post_images_task,
                    args=(post.id,),
                    kwargs={'only_sections': True, 'sections': ['generated_daytime_outfits_image']},
                ).start()
                return JsonResponse({'success': True, 'message': 'Daytime Outfits image regeneration started.', 'status': getattr(post, status_field)})

            elif content_type == 'generated_evening_and_nightlife_image':
                status_field = 'modular_images_status' if hasattr(post, 'modular_images_status') else 'section_images_status'
                setattr(post, status_field, 'in_process')
                setattr(post, content_type, None)  # 👈 clear only this field
                post.save(update_fields=[content_type, status_field])
                threading.Thread(
                    target=generate_post_images_task,
                    args=(post.id,),
                    kwargs={'only_sections': True, 'sections': ['generated_evening_and_nightlife_image']},
                ).start()
                return JsonResponse({'success': True, 'message': 'Evening & Nightlife image regeneration started.', 'status': getattr(post, status_field)})

            elif content_type == 'generated_outdoor_activities_image':
                status_field = 'modular_images_status' if hasattr(post, 'modular_images_status') else 'section_images_status'
                setattr(post, status_field, 'in_process')
                setattr(post, content_type, None)  # 👈 clear only this field
                post.save(update_fields=[content_type, status_field])
                threading.Thread(
                    target=generate_post_images_task,
                    args=(post.id,),
                    kwargs={'only_sections': True, 'sections': ['generated_outdoor_activities_image']},
                ).start()
                return JsonResponse({'success': True, 'message': 'Outdoor Activities image regeneration started.', 'status': getattr(post, status_field)})

            elif content_type == 'generated_seasonal_variations_image':
                status_field = 'modular_images_status' if hasattr(post, 'modular_images_status') else 'section_images_status'
                setattr(post, status_field, 'in_process')
                setattr(post, content_type, None)  # 👈 clear only this field
                post.save(update_fields=[content_type, status_field])
                threading.Thread(
                    target=generate_post_images_task,
                    args=(post.id,),
                    kwargs={'only_sections': True, 'sections': ['generated_seasonal_variations_image']},
                ).start()
                return JsonResponse({'success': True, 'message': 'Seasonal Variations image regeneration started.', 'status': getattr(post, status_field)})

            elif content_type == 'generated_style_tips_for_blending_image':
                status_field = 'modular_images_status' if hasattr(post, 'modular_images_status') else 'section_images_status'
                setattr(post, status_field, 'in_process')
                setattr(post, content_type, None)  # 👈 clear only this field
                post.save(update_fields=[content_type, status_field])
                threading.Thread(
                    target=generate_post_images_task,
                    args=(post.id,),
                    kwargs={'only_sections': True, 'sections': ['generated_style_tips_for_blending_image']},
                ).start()
                return JsonResponse({'success': True, 'message': 'Style Tips for Blending image regeneration started.', 'status': getattr(post, status_field)})

            elif content_type == 'generated_destination_specific_extras_image':
                status_field = 'modular_images_status' if hasattr(post, 'modular_images_status') else 'section_images_status'
                setattr(post, status_field, 'in_process')
                setattr(post, content_type, None)  # 👈 clear only this field
                post.save(update_fields=[content_type, status_field])
                threading.Thread(
                    target=generate_post_images_task,
                    args=(post.id,),
                    kwargs={'only_sections': True, 'sections': ['generated_destination_specific_extras_image']},
                ).start()
                return JsonResponse({'success': True, 'message': 'Destination-Specific Extras image regeneration started.', 'status': getattr(post, status_field)})


            elif content_type == 'style_images':
                post.style_images_status = 'in_process'
                post.style_images = {}
                post.style_prompts = None
                post.save()
                threading.Thread(
                    target=generate_post_images_task,
                    args=(post.id,),
                    kwargs={'only_style': True, 'style_prompts': post.style_prompts}
                ).start()
                return JsonResponse({'success': True, 'message': 'Style images regeneration started.', 'status': post.style_images_status})

            else:
                return JsonResponse({'success': False, 'message': 'Invalid content type.'})

        return JsonResponse({'success': False, 'message': 'Invalid request.'})

    # ---------------------------
    # Content Generation View
    # ---------------------------
    def generate_content_view(self, request, pk):
        keyword = get_object_or_404(Keyword, pk=pk)
        prompt_obj = Prompt.objects.filter(prompt_id=keyword.prompt_id).first()
        model_info = ModelInfo.objects.filter(model_id=keyword.model_id).first()

        # Prompt templates (with fallbacks)
        title_prompt_template = prompt_obj.title_prompt if prompt_obj and prompt_obj.title_prompt else ""
        intro_prompt_template = prompt_obj.intro_prompt if prompt_obj and prompt_obj.intro_prompt else ""
        style_section_prompt_template = prompt_obj.style_prompt if prompt_obj and prompt_obj.style_prompt else ""
        conclusion_prompt_template = prompt_obj.conclusion_prompt if prompt_obj and prompt_obj.conclusion_prompt else ""
        meta_data_prompt_template = prompt_obj.meta_data_prompt if prompt_obj and prompt_obj.meta_data_prompt else ""

        # Replace placeholders
        def repl(t):
            if not t:
                return ""
            return (t.replace('{{keyword}}', keyword.keyword or '')
                     .replace('{{hairstyle_name}}', keyword.keyword or ''))

        title_prompt = repl(title_prompt_template)
        intro_prompt = repl(intro_prompt_template)
        style_section_prompt = repl(style_section_prompt_template)
        conclusion_prompt = repl(conclusion_prompt_template)
        meta_data_prompt = repl(meta_data_prompt_template)

        image_prompt = prompt_obj.image_prompt if prompt_obj else 'N/A'
        if image_prompt and keyword.keyword:
            image_prompt = image_prompt.replace('{{hairstyle_name}}', keyword.keyword)
        if image_prompt and model_info:
            image_prompt = (image_prompt
                            .replace('{{ethnicity}}', model_info.ethnicity or '')
                            .replace('{{skin_tone}}', model_info.skin_tone or '')
                            .replace('{{hair_texture}}', model_info.hair_texture or '')
                            .replace('{{face_shape}}', model_info.face_shape or ''))

        wordpress_post_status = None

        if request.method == "POST":
            action = request.POST.get("action")
            content_type = request.POST.get("content_type")

            if action == "generate_text":
                client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
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
                    return JsonResponse({'success': False, 'message': 'Invalid content type provided.'})

                try:
                    if content_type == 'meta_seo':
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": full_prompt}
                            ]
                        )
                        raw_meta_output = response.choices[0].message.content
                        try:
                            meta_data = json.loads(raw_meta_output)
                            meta_title = meta_data.get('meta_title', '').replace('\\"', '"').strip('"')
                            meta_description = meta_data.get('meta_description', '').replace('\\"', '"').strip('"')
                            if keyword.keyword:
                                meta_title = meta_title.replace('{{keyword}}', keyword.keyword)
                                meta_description = meta_description.replace('{{hairstyle_name}}', keyword.keyword)
                        except json.JSONDecodeError:
                            meta_title, meta_description = "", raw_meta_output

                        keyword.meta_title = meta_title
                        keyword.meta_description = meta_description
                        keyword.save()
                        return JsonResponse({
                            'success': True,
                            'message': 'Meta Title and Description generated successfully!',
                            'meta_title': meta_title,
                            'meta_description': meta_description,
                        })
                    else:
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": full_prompt}
                            ]
                        )
                        raw_generated_content = response.choices[0].message.content
                        generated_content = raw_generated_content.replace('\\"', '"').strip('"')
                        html_content = markdown.markdown(generated_content)
                        setattr(keyword, field_to_update, html_content)
                        keyword.save()
                        return JsonResponse({
                            "success": True,
                            "message": f"{content_type.replace('_', ' ').title()} generated successfully!",
                            "content": html_content
                        })
                except openai.OpenAIError as e:
                    return JsonResponse({'success': False, 'message': f'ChatGPT API Error: {e}'})
                except requests.exceptions.RequestException as e:
                    return JsonResponse({'success': False, 'message': f'Network/WordPress API Error: {e}'})

            elif action == "generate_image":
                client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
                try:
                    response = client.images.generate(
                        model="dall-e-2",
                        prompt=image_prompt,
                        size="256x256",
                        n=1,
                    )
                    image_url = response.data[0].url
                    keyword.generated_image_url = image_url
                    keyword.save()
                    return JsonResponse({'success': True, 'message': 'Image generated successfully!', 'image_url': image_url})
                except openai.OpenAIError as e:
                    return JsonResponse({'success': False, 'message': f'DALL-E API Error: {e}'})

            elif action == "push_to_wordpress":
                # Collect CKEditor sections
                title_content = request.POST.get('title_content', '')
                intro_content = request.POST.get('intro_content', '')
                style_section_content = request.POST.get('style_section_content', '')
                conclusion_content = request.POST.get('conclusion_content', '')

                title_content_plain = re.sub(r'<[^>]*>', '', title_content).strip()
                combined_content = f"{intro_content}\n\n{style_section_content}\n\n{conclusion_content}"

                wordpress_api_url = settings.WORDPRESS_API_URL
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {settings.WORDPRESS_JWT_TOKEN}",
                    "Accept": "application/json",
                    "User-Agent": "Mozilla/5.0",
                }

                try:
                    base_wordpress_url = settings.WORDPRESS_API_URL.split('/wp-json/')[0]
                    user_check_url = f"{base_wordpress_url}/wp-json/wp/v2/users/me"
                    user_response = requests.get(user_check_url, headers=headers, timeout=60)
                    if user_response.status_code != 200:
                        return JsonResponse({
                            'success': False,
                            'message': f'WordPress user verification failed. Status: {user_response.status_code}, Error: {user_response.text}'
                        })

                    post_data = {
                        "title": title_content_plain,
                        "content": combined_content,
                        "status": "publish",
                    }
                    response = requests.post(wordpress_api_url, headers=headers, json=post_data, timeout=120)
                    if response.status_code == 201:
                        return JsonResponse({'success': True, 'message': 'Content pushed to WordPress successfully!'})
                    else:
                        try:
                            error_message = response.json().get('message', response.text)
                        except Exception:
                            error_message = response.text
                        return JsonResponse({'success': False, 'message': f'Failed to push content to WordPress. Status: {response.status_code}, Error: {error_message}'})
                except requests.exceptions.RequestException as e:
                    return JsonResponse({'success': False, 'message': f'Network/WordPress API Error: {e}'})

            elif action == "generate_all_content":
                client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
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
                            model="gpt-4o",
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
                                all_generated_data['meta_title'] = "Error generating meta title"
                                all_generated_data['meta_description'] = generated_content
                        else:
                            generated_content = generated_content.replace('\\"', '"').strip('"')
                            html_content = markdown.markdown(generated_content)
                            setattr(keyword, f'generated_{c_type}', html_content)
                            all_generated_data[c_type] = html_content
                    except openai.OpenAIError as e:
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

        form = KeywordAdminForm(instance=keyword)
        context = self.admin_site.each_context(request)
        context.update({
            'opts': self.model._meta,
            'app_label': self.model._meta.app_label,
            'original': keyword,
            'title': f'Generate Content for: {keyword.keyword}',
            'image_prompt': prompt_obj.image_prompt if prompt_obj else 'N/A',
            'generated_image_url': keyword.generated_image_url,
            'wordpress_post_status': None,
            'meta_title': keyword.meta_title,
            'meta_description': keyword.meta_description,
            'generated_title': keyword.generated_title,
            'generated_intro': keyword.generated_intro,
            'generated_style_section': keyword.generated_style_section,
            'generated_conclusion': keyword.generated_conclusion,
            'title_prompt': title_prompt,
            'intro_prompt': intro_prompt,
            'style_section_prompt': style_section_prompt,
            'conclusion_prompt': conclusion_prompt,
            'form': form,
        })
        return TemplateResponse(request, "admin/WordAI_Publisher/generated_content.html", context)

    # ---------------------------
    # Bulk Generation Action
    # ---------------------------
    def generate_content_for_selected(self, request, queryset):
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
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return response.choices[0].message.content

            generated_title = gpt_content("", title_prompt)
            if generated_title:
                generated_title = generated_title.strip().strip('"').strip("'")
            generated_intro = gpt_content(
                "You are a helpful assistant that generates engaging article introductions. Do not use markdown.",
                intro_prompt
            )
            generated_style_section = gpt_content(
                "You are a helpful assistant that generates detailed style descriptions. "
                "Use <h2> HTML headings for each style section, followed by paragraphs. Do not use markdown.",
                style_prompt
            )
            generated_conclusion = gpt_content(
                "You are a helpful assistant that generates article conclusions. Do not use markdown.",
                conclusion_prompt
            )
            meta_title = ''
            meta_description = ''
            if meta_data_prompt:
                meta_json = gpt_content(
                    "You are an SEO assistant that generates concise meta titles and descriptions in JSON format.",
                    meta_data_prompt
                )
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

            threading.Thread(target=generate_post_images_task, args=(post.id,)).start()
            count += 1

        self.message_user(
            request,
            f"Content generated for {count} keywords. Skipped: {', '.join(skipped)}"
        )
    generate_content_for_selected.short_description = "Generate Content for selected keywords"


# =============================================================================
# Admin: Post
# =============================================================================

@admin.register(Post)
class PostAdmin(admin.ModelAdmin):
    change_form_template = "admin/WordAI_Publisher/post_change_form.html"
    form = PostAdminForm

    class Media:
        css = {'all': ('WordAI_Publisher/css/main.css',)}

    list_display = (
        'id', 'keyword',
        'content_generated_icon',
        'featured_image_status_icon',
        'style_images_status_icon',
        'modular_images_status_icon',
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

    # ---------------------------
    # Preview
    # ---------------------------
    def preview_view(self, request, post_id):
        post = get_object_or_404(Post, pk=post_id)

        # --- Find the most recent job tied to this post and normalize template_type ---
        last_job = (
            GenerationJob.objects
            .filter(post=post)
            .order_by('-finished_at', '-created_at')
            .first()
        )
        raw_tpl = ((last_job.template_type if last_job else '') or '').strip().lower()
        raw_tpl = raw_tpl.replace('-', ' ').replace('_', ' ')
        if raw_tpl in ('food love', 'foodlove'):
            job_template_type = 'food_love'
        elif raw_tpl == 'modular':
            job_template_type = 'modular'
        else:
            job_template_type = 'regular'

        # --- Build style_sections only for non–Food Love ---
        style_sections = []
        if job_template_type != 'food_love':
            # Safely decode style_images JSON (if present)
            style_images = getattr(post, 'style_images', {})  # may be dict or JSON string or None
            if isinstance(style_images, str):
                try:
                    style_images = json.loads(style_images or "{}")
                except json.JSONDecodeError:
                    style_images = {}

            # Normalize keys for lookup consistency
            style_images_clean = { (k or '').strip(): v for k, v in (style_images or {}).items() }

            # Split generated_style_section into <h2> blocks
            if post.generated_style_section:
                headings = re.findall(r'(<h2>.*?</h2>)', post.generated_style_section, flags=re.DOTALL|re.IGNORECASE)
                contents = re.split(r'<h2>.*?</h2>', post.generated_style_section, flags=re.DOTALL|re.IGNORECASE)[1:]

                for i, heading in enumerate(headings):
                    style_name = re.sub(r'<.*?>', '', heading, flags=re.DOTALL).strip()
                    # Default placeholder unless images are still in process
                    image_url = '/static/WordAI_Publisher/img/placeholder.png'
                    if getattr(post, 'style_images_status', '') != 'in_process':
                        image_url = style_images_clean.get(style_name, image_url)

                    section_content = contents[i] if i < len(contents) else ""
                    style_sections.append({
                        'heading': mark_safe(heading),
                        'image_url': image_url,
                        'content': mark_safe(section_content),
                    })

        # Render preview template with the template type from GenerationJob
        return render(
            request,
            'admin/WordAI_Publisher/post_preview.html',
            {
                'post': post,
                'style_sections': style_sections,
                'job_template_type': job_template_type,  # template branches on this
            }
        )
        # ---------------------------
        # Push to WordPress (S3-safe)
    # ---------------------------
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
        self.message_user(request, f"{success_count} posts pushed to WordPress!")
    push_selected_to_wordpress.short_description = "Push selected posts to WordPress"

    def _push_post_to_wordpress(self, post, request=None, silent=False):
        import os, re, json, threading, requests
        from django.contrib import messages

        # === Basics ===
        title_content_plain = re.sub(r'<[^>]*>', '', post.generated_title or '').strip()
        combined_content = f"{post.generated_intro or ''}\n\n{post.generated_style_section or ''}\n\n{post.generated_conclusion or ''}"

        # Determine template_type from latest GenerationJob (fallback to post.template_type)
        tmpl_type = ''
        gj = (
            GenerationJob.objects
            .filter(post_id=post.id)
            .order_by('-finished_at', '-started_at', '-id')
            .values('template_type')
            .first()
        )
        if gj and gj.get('template_type'):
            tmpl_type = (gj['template_type'] or '').strip().lower()
        else:
            tmpl_type = (getattr(post, 'template_type', '') or '').strip().lower()

        norm = tmpl_type.replace('-', ' ').replace('_', ' ')
        is_foodlove = norm in ('food love', 'foodlove')
        is_modular = (norm == 'modular')
        print('template_type:', tmpl_type, 'is_modular:', is_modular, 'is_foodlove:', is_foodlove)

        # === Select endpoints + token from env-backed settings ===
        if is_foodlove:
            wordpress_api_url = getattr(settings, 'FOODLOVE_API_URL', '')
            media_endpoint     = getattr(settings, 'FOODLOVE_API_URL_MEDIA', '')
            jwt_token          = getattr(settings, 'FOODLOVE_JWT_TOKEN', '')
        else:
            wordpress_api_url = getattr(settings, 'WORDPRESS_API_URL', '')
            media_endpoint     = getattr(settings, 'WORDPRESS_API_URL_MEDIA', '')
            jwt_token          = getattr(settings, 'WORDPRESS_JWT_TOKEN', '')

        if not wordpress_api_url or not media_endpoint:
            if not silent:
                self.message_user(
                    request,
                    "Missing WordPress API env vars for this template type.",
                    level=messages.ERROR
                )
            return False

        # Build headers for the chosen site
        post_headers = {
            "Authorization": f"Bearer {jwt_token}" if jwt_token else "",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0",
        }
        media_headers = {
            "Authorization": f"Bearer {jwt_token}" if jwt_token else "",
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        }

        # === STEP 1: Upload Featured Image (allowed for both sites) ===
        image_id = None
        try:
            if post.featured_image:
                filename = os.path.basename(post.featured_image.name) or "featured.jpg"
                content_type = _guess_ct(filename) or "image/jpeg"
                with post.featured_image.open("rb") as fh:
                    files = {"file": (filename, fh, content_type)}
                    r = requests.post(media_endpoint, headers=media_headers, files=files, timeout=120)
                if r.status_code == 201:
                    image_id = r.json().get('id')
                elif not silent:
                    self.message_user(
                        request,
                        f"Featured image upload failed: {r.status_code} {r.text}",
                        level=messages.ERROR
                    )
        except Exception as e:
            if not silent:
                self.message_user(request, f"Image upload exception: {e}", level=messages.ERROR)

        # =========================
        # FOOD LOVE BRANCH (simple)
        # =========================
        if is_foodlove:
            # Push everything into default editor. No ACF/meta/gallery/modular images.
            post_payload = {
                "title": title_content_plain,
                "status": "draft",
                "content": combined_content,
            }
            if image_id:
                post_payload["featured_media"] = image_id

            try:
                resp = requests.post(wordpress_api_url, headers=post_headers, json=post_payload, timeout=120)
                if resp.status_code == 201:
                    post.status = 'draft'
                    post.save(update_fields=['status'])
                    if not silent:
                        self.message_user(request, "Post pushed to Food Love!", level=messages.SUCCESS)
                    return True
                else:
                    if not silent:
                        try:
                            error_message = resp.json().get('message', resp.text)
                        except Exception:
                            error_message = resp.text
                        self.message_user(request, f"Failed to push to Food Love: {error_message}", level=messages.ERROR)
                    return False
            except requests.exceptions.RequestException as e:
                if not silent:
                    self.message_user(request, f"Network/WordPress API Error (Food Love): {e}", level=messages.ERROR)
                return False

        # ============================================
        # FASHIONBEANS BRANCH (your existing behavior)
        # ============================================

        # === STEP 2: Upload Gallery Style Images (SKIP for modular) ===
        gallery_media_ids = []
        if not is_modular:
            style_images_data = post.style_images or {}   # {style_name: key_or_url}
            threads = []
            _silent = False  # local control within this scope

            def upload_image_thread(style_name, key_or_url, collected_ids):
                try:
                    fobj, filename = _open_from_storage_or_url(key_or_url)
                    try:
                        content_type = _guess_ct(filename) or "image/jpeg"
                        files = {"file": (os.path.basename(filename), fobj, content_type)}
                        rr = requests.post(media_endpoint, headers=media_headers, files=files, timeout=120)
                    finally:
                        try:
                            fobj.close()
                        except Exception:
                            pass

                    if rr.status_code == 201:
                        collected_ids.append(rr.json().get("id"))
                    elif not _silent:
                        msg = rr.text
                        try:
                            msg = rr.json()
                        except Exception:
                            pass
                        print(f"[Gallery] Failed to upload {style_name}: {rr.status_code} {msg}")
                except Exception as e:
                    if not _silent:
                        print(f"[Gallery] Error uploading {style_name}: {e}")

            if isinstance(style_images_data, dict):
                for style_name, key_or_url in style_images_data.items():
                    t = threading.Thread(target=upload_image_thread, args=(style_name, key_or_url, gallery_media_ids))
                    t.start()
                    threads.append(t)

            for t in threads:
                t.join()

        # === STEP 2b: Upload Modular Section Images (ONLY for modular) ===
        modular_image_ids = {}  # {acf_field_name: media_id}
        if is_modular:
            modular_sources = {
                "generated_quick_style_snapshot_image":        getattr(post, "generated_quick_style_snapshot_image", None),
                "generated_packing_essentials_checklist_image":getattr(post, "generated_packing_essentials_checklist_image", None),
                "generated_daytime_outfits_image":             getattr(post, "generated_daytime_outfits_image", None),
                "generated_evening_and_nightlife_image":       getattr(post, "generated_evening_and_nightlife_image", None),
                "generated_outdoor_activities_image":          getattr(post, "generated_outdoor_activities_image", None),
                "generated_seasonal_variations_image":         getattr(post, "generated_seasonal_variations_image", None),
                "generated_style_tips_for_blending_image":     getattr(post, "generated_style_tips_for_blending_image", None),
                "generated_destination_specific_extras_image": getattr(post, "generated_destination_specific_extras_image", None),
            }

            def upload_modular_image(acf_field_name, key_or_url, out_map):
                """Uploads one modular image to WP media; stores media ID in out_map[acf_field_name]."""
                if not key_or_url:
                    out_map[acf_field_name] = None
                    return
                try:
                    fobj, filename = _open_from_storage_or_url(key_or_url)
                    try:
                        content_type = _guess_ct(filename) or "image/jpeg"
                        files = {"file": (os.path.basename(filename), fobj, content_type)}
                        rr = requests.post(media_endpoint, headers=media_headers, files=files, timeout=120)
                    finally:
                        try:
                            fobj.close()
                        except Exception:
                            pass

                    if rr.status_code == 201:
                        out_map[acf_field_name] = rr.json().get("id")
                    else:
                        out_map[acf_field_name] = None
                        if not silent:
                            try:
                                msg = rr.json().get('message', rr.text)
                            except Exception:
                                msg = rr.text
                            print(f"[Modular] Failed to upload {acf_field_name}: {rr.status_code} {msg}")
                except Exception as e:
                    out_map[acf_field_name] = None
                    if not silent:
                        print(f"[Modular] Error uploading {acf_field_name}: {e}")

            # Upload sequentially (threading is fine too)
            for acf_name, src in modular_sources.items():
                upload_modular_image(acf_name, src, modular_image_ids)

        # === STEP 3: Build payload (different ACF for modular) ===
        # (FashionBeans payload unchanged)
        acf_data = {
            "ai_title": post.generated_title,
            "ai_intro": post.generated_intro,
            "ai_conclusion": post.generated_conclusion,
        }

        if is_modular:
            acf_data.update({
                "field_66bb21792e74b": "single-modular-template",
                "generated_quick_style_snapshot": post.generated_quick_style_snapshot,
                "generated_packing_essentials_checklist": post.generated_packing_essentials_checklist,
                "generated_daytime_outfits": post.generated_daytime_outfits,
                "generated_evening_and_nightlife": post.generated_evening_and_nightlife,
                "generated_outdoor_activities": post.generated_outdoor_activities,
                "generated_seasonal_variations": post.generated_seasonal_variations,
                "generated_style_tips_for_blending": post.generated_style_tips_for_blending,
                "generated_destination_specific_extras": post.generated_destination_specific_extras,
                # include uploaded modular media IDs
                "generated_quick_style_snapshot_image":        modular_image_ids.get("generated_quick_style_snapshot_image"),
                "generated_packing_essentials_checklist_image":modular_image_ids.get("generated_packing_essentials_checklist_image"),
                "generated_daytime_outfits_image":             modular_image_ids.get("generated_daytime_outfits_image"),
                "generated_evening_and_nightlife_image":       modular_image_ids.get("generated_evening_and_nightlife_image"),
                "generated_outdoor_activities_image":          modular_image_ids.get("generated_outdoor_activities_image"),
                "generated_seasonal_variations_image":         modular_image_ids.get("generated_seasonal_variations_image"),
                "generated_style_tips_for_blending_image":     modular_image_ids.get("generated_style_tips_for_blending_image"),
                "generated_destination_specific_extras_image": modular_image_ids.get("generated_destination_specific_extras_image"),
            })
        else:
            # Regular article: single style section + gallery image IDs
            acf_data.update({
                "field_66bb21792e74b": "single-custom-template",
                "ai_style": post.generated_style_section,
                "style_images": gallery_media_ids,
            })

        post_payload = {
            "title": title_content_plain,
            "status": "draft",
            "acf": acf_data,
            "meta": {
                "rank_math_title": post.meta_title,
                "rank_math_description": post.meta_description,
            },
        }
        if image_id:
            post_payload["featured_media"] = image_id

        # === STEP 4: Create post (FashionBeans) ===
        try:
            resp = requests.post(wordpress_api_url, headers=post_headers, json=post_payload, timeout=120)
            if resp.status_code == 201:
                post.status = 'draft'  # or 'publish'
                post.save(update_fields=['status'])
                if not silent:
                    self.message_user(request, "Post pushed to WordPress!", level=messages.SUCCESS)
                ok = True
            else:
                ok = False
                if not silent:
                    try:
                        error_message = resp.json().get('message', resp.text)
                    except Exception:
                        error_message = resp.text
                    self.message_user(request, f"Failed to push post: {error_message}", level=messages.ERROR)
        except requests.exceptions.RequestException as e:
            ok = False
            if not silent:
                self.message_user(request, f"Network/WordPress API Error: {e}", level=messages.ERROR)

        return ok

    # ---------------------------
    # Icons
    # ---------------------------
    def content_generated_icon(self, obj):
        if obj.content_generated == 'completed':
            return format_html('<span style="color:green;">&#10004;</span>')
        else:
            return format_html('<span style="color:red;">&#10008;</span>')
    content_generated_icon.short_description = "Content Generated"

    def featured_image_status_icon(self, obj):
        if obj.featured_image_status == 'completed':
            return format_html('<span style="color:green;">&#10004;</span>')
        elif obj.featured_image_status == 'in_process':
            return format_html('<span style="color:orange;">&#8987;</span>')
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


    def modular_images_status_icon(self, obj):
        status = getattr(obj, 'modular_images_status', 'not_generated')
        if status == 'completed':
            return format_html('<span title="Completed" style="color:green;">&#10004;</span>')
        elif status == 'in_process':
            return format_html('<span title="In process" style="color:orange;">&#8987;</span>')
        else:
            return format_html('<span title="Not generated" style="color:red;">&#10008;</span>')
    modular_images_status_icon.short_description = "Modular Images Status"
    modular_images_status_icon.admin_order_field = 'modular_images_status'

    # ---------------------------
    # Regeneration Actions
    # ---------------------------
    def regenerate_featured_image(self, request, queryset):
        count = 0
        for post in queryset:
            post.featured_image_status = 'in_process'
            post.save()
            threading.Thread(
                target=generate_post_images_task,
                args=(post.id,),
                kwargs={'only_featured': True}
            ).start()
            count += 1
        self.message_user(
            request,
            f"✅ Regeneration started for featured images of {count} post(s).",
            level=messages.SUCCESS
        )
    regenerate_featured_image.short_description = "Regenerate Featured Image"

    def regenerate_style_images(self, request, queryset):
        count = 0
        for post in queryset:
            post.style_images_status = 'in_process'
            post.save()
            threading.Thread(
                target=generate_post_images_task,
                args=(post.id,),
                kwargs={'only_style': True}
            ).start()
            count += 1
        self.message_user(
            request,
            f"✅ Regeneration started for style images of {count} post(s).",
            level=messages.SUCCESS
        )
    regenerate_style_images.short_description = "Regenerate Style Images"


# =============================================================================
# Utilities
# =============================================================================

def extract_styles_by_h2(style_section_html):
    if not isinstance(style_section_html, str):
        return []
    pattern = r'<h2>(.*?)</h2>'
    return re.findall(pattern, style_section_html)


# make sure GenerationJob is imported:
# from .models import Keyword, Prompt, ModelInfo, Post, GenerationJob

# ---------- Actions ----------
def retry_jobs(modeladmin, request, queryset):
    updated = queryset.update(
        status='queued',
        error='',
        started_at=None,
        finished_at=None,
    )
    try:
        _ensure_worker_running()  # uses your queue worker helper
    except Exception:
        pass
    modeladmin.message_user(request, f"Re-queued {updated} job(s).")

retry_jobs.short_description = "Retry (re-queue) selected job(s)"

def cancel_jobs(modeladmin, request, queryset):
    updated = queryset.exclude(status='done').update(
        status='failed',
        error='Cancelled by admin',
        finished_at=timezone.now(),
    )
    modeladmin.message_user(request, f"Cancelled {updated} job(s).")

cancel_jobs.short_description = "Cancel (mark failed) selected job(s)"

def run_worker_now(modeladmin, request, queryset):
    try:
        _ensure_worker_running()
        modeladmin.message_user(request, "Worker started (or already running).")
    except Exception as e:
        modeladmin.message_user(request, f"Unable to start worker: {e}", level=messages.ERROR)

run_worker_now.short_description = "Start queue worker"

@admin.register(GenerationJob)
class GenerationJobAdmin(admin.ModelAdmin):
    # Optional autorefresh template; add it in step 2 or remove this line
    change_list_template = "admin/WordAI_Publisher/generationjob_changelist.html"

    list_display = (
        'id',
        'keyword_link',
        'prompt_link',
        'prompt_type',
        'version_count',
        'status_badge',
        'created_at',
        'started_at',
        'finished_at',
        'post_link',
    )
    list_filter = ('status', 'prompt_type', 'created_at')
    search_fields = ('keyword__keyword', 'prompt__prompt_id', 'post__id')
    readonly_fields = ('error', 'created_at', 'started_at', 'finished_at', 'post')

    actions = [retry_jobs, cancel_jobs, run_worker_now]
    ordering = ('created_at',)

    def keyword_link(self, obj):
        if not obj.keyword_id:
            return "-"
        url = reverse('admin:WordAI_Publisher_keyword_change', args=[obj.keyword_id])
        return format_html('<a href="{}">{}</a>', url, obj.keyword.keyword)
    keyword_link.short_description = "Keyword"

    def prompt_link(self, obj):
        if not obj.prompt_id:
            return "-"
        url = reverse('admin:WordAI_Publisher_prompt_change', args=[obj.prompt_id])
        return format_html('<a href="{}">{}</a>', url, obj.prompt.prompt_id)
    prompt_link.short_description = "Prompt"

    def post_link(self, obj):
        if not obj.post_id:
            return "-"
        url = reverse('admin:WordAI_Publisher_post_change', args=[obj.post_id])
        return format_html('<a href="{}">Post #{}</a>', url, obj.post_id)
    post_link.short_description = "Post"

    def status_badge(self, obj):
        colors = {
            'queued': '#ff9800',   # orange
            'running': '#2196f3',  # blue
            'done': '#4caf50',     # green
            'failed': '#f44336',   # red
        }
        color = colors.get(obj.status, '#777')
        return format_html('<span style="color:{};font-weight:600">{}</span>', color, obj.status)
    status_badge.short_description = "Status"

    def changelist_view(self, request, extra_context=None):
        # When no filter is applied, default to queued
        if request.method == "GET" and "status__exact" not in request.GET and not request.GET:
            return HttpResponseRedirect(f"{request.path}?status__exact=queued")
        return super().changelist_view(request, extra_context=extra_context)