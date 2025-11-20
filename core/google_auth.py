from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect

@login_required
def google_oauth_success(request):
    """
    Handle successful Google OAuth login.
    Generate JWT tokens and redirect to frontend with tokens.
    """
    user = request.user
    
    # Generate JWT tokens
    refresh = RefreshToken.for_user(user)
    access_token = str(refresh.access_token)
    refresh_token = str(refresh)
    
    # Redirect to frontend with tokens in URL params
    frontend_url = f'http://localhost:3000/auth/google/callback?access_token={access_token}&refresh_token={refresh_token}'
    return redirect(frontend_url)
