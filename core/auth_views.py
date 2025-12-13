from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework.response import Response
from rest_framework import status
from .models import Notification


class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    """Custom serializer to add user info to token response"""
    
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        # Add custom claims if needed
        token['username'] = user.username
        return token


class CustomTokenObtainPairView(TokenObtainPairView):
    """
    Custom JWT token view that creates a login notification
    """
    serializer_class = CustomTokenObtainPairSerializer
    
    def post(self, request, *args, **kwargs):
        response = super().post(request, *args, **kwargs)
        
        # If login successful, create notification
        if response.status_code == status.HTTP_200_OK:
            # Get username from request
            username = request.data.get('username')
            if username:
                from django.contrib.auth.models import User
                try:
                    user = User.objects.get(username=username)
                    Notification.objects.create(
                        user=user,
                        message=f"👋 Welcome back, {user.first_name or user.username}! Ready to continue your learning journey?"
                    )
                except User.DoesNotExist:
                    pass
        
        return response
