from django.contrib import admin
from django.urls import path, include
from core.google_auth import google_oauth_success

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('core.urls')),
    path('accounts/', include('allauth.urls')),
    path('auth/google/success/', google_oauth_success, name='google_oauth_success'),
]
