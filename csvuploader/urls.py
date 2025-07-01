"""csvuploader URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from WordAI_Publisher import views  # 👈 Updated import
from django.conf import settings
from django.utils.safestring import mark_safe
from django.views.generic.base import RedirectView # Import RedirectView
from django.conf.urls.static import static # Import static

# Custom admin site configuration
admin.site.site_header = mark_safe(settings.ADMIN_SITE_HEADER)
admin.site.site_title = settings.ADMIN_SITE_TITLE
admin.site.index_title = settings.ADMIN_INDEX_TITLE

urlpatterns = [
    path('admin/', admin.site.urls),
    path('ckeditor/', include('ckeditor_uploader.urls')),
    # Redirect root URL to admin page
    path('', RedirectView.as_view(url='/admin/', permanent=False), name='index'),
    # path('upload/', views.upload_csv, name='upload_csv'),  # optional: remove or keep
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)