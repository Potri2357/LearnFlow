# core/serializers.py
from rest_framework import serializers
from django.contrib.auth.models import User
from .models import LectureNote, Question, UserAnswer, TopicWeakness, Flashcard, StickyNote
from .models import TopicMastery, UserStreak, UserProfile, Notification

class LectureNoteSerializer(serializers.ModelSerializer):
    class Meta:
        model = LectureNote
        fields = ['id', 'title', 'subject', 'file', 'content', 'created_at', 'study_notes', 'formulas', 'key_points']
        read_only_fields = ['content', 'created_at', 'study_notes', 'formulas', 'key_points']

class QuestionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Question
        fields = [
            'id', 'lecture_note', 'topic', 'question_text',
            'option_a', 'option_b', 'option_c', 'option_d',
            'correct_option', 'explanation', 'difficulty',
            'blooms_level', 'question_type', 'is_high_yield',
            'relevance_score', 'is_starred', 'attempt_count',
            'correct_count', 'created_at',
        ]


class StickyNoteSerializer(serializers.ModelSerializer):
    class Meta:
        model = StickyNote
        fields = [
            'id', 'lecture_note', 'title', 'content', 'color',
            'note_type', 'is_pinned', 'page_number', 'source_text',
            'created_at', 'updated_at',
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']

class FlashcardSerializer(serializers.ModelSerializer):
    class Meta:
        model = Flashcard
        fields = "__all__"
        read_only_fields = ['user', 'created_at']

class UserAnswerSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserAnswer
        fields = "__all__"

class TopicMasterySerializer(serializers.ModelSerializer):
    class Meta:
        model = TopicMastery
        fields = "__all__"

class UserStreakSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserStreak
        fields = "__all__"


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name']
        read_only_fields = ['id']


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True, style={'input_type': 'password'})
    password2 = serializers.CharField(write_only=True, required=True, style={'input_type': 'password'}, label='Confirm Password')

    class Meta:
        model = User
        fields = ['username', 'email', 'password', 'password2', 'first_name', 'last_name']

    def validate(self, attrs):
        if attrs['password'] != attrs['password2']:
            raise serializers.ValidationError({"password": "Password fields didn't match."})
        return attrs

    def create(self, validated_data):
        validated_data.pop('password2')
        user = User.objects.create_user(**validated_data)
        # Create UserProfile automatically
        UserProfile.objects.create(user=user)
        return user


class UserProfileSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)

    class Meta:
        model = UserProfile
        fields = ['id', 'user', 'bio', 'avatar']


class NotificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Notification
        fields = ['id', 'message', 'is_read', 'created_at']
        read_only_fields = ['id', 'created_at']

