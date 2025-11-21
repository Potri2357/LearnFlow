from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .views import (
    upload_lecture_note, generate_questions, submit_answer, weak_topics, progress,
    generate_mcqs, get_quiz_questions, submit_mcq_answer, adaptive_quiz_start, generate_study_plan,
    analytics_for_note, recent_weak_topics, next_actions, ai_insights,
    upload_pdf, get_note_details, quiz_completed,
    RegisterView, UserProfileView, NotificationListView, NotificationMarkReadView, 
    NotificationMarkAllReadView, NotificationDeleteView, CurrentUserView
)

urlpatterns = [
    path("upload-note/", upload_lecture_note),
    path('generate-questions/<int:note_id>/', generate_questions, name='generate_questions'),
    path("submit-answer/", submit_answer),
    path("weak-topics/", weak_topics),
    path("progress/", progress),
    path("generate-mcqs/", generate_mcqs),
    path("quiz/<int:note_id>/", get_quiz_questions),
    path("submit-mcq/", submit_mcq_answer),   
    path("adaptive/quiz/start/", adaptive_quiz_start),
    path("study-plan/", generate_study_plan),
    path("analytics/<int:note_id>/", analytics_for_note),
    path("recent-weak-topics/", recent_weak_topics),
    path("next-actions/", next_actions),
    path("ai-insights/<int:note_id>/", ai_insights),
    path('upload/', upload_lecture_note, name='upload_note'),
    path('questions/<int:note_id>/generate/', generate_questions, name='generate_questions'),
    path('answer/', submit_answer, name='submit_answer'),
    path('weak-topics/', weak_topics, name='weak_topics'),
    path('progress/', progress, name='progress'),
    path('analytics/<int:note_id>/', analytics_for_note, name='analytics'),
    path('study-plan/<int:note_id>/', generate_study_plan, name='study_plan'),
    path('note-details/<int:note_id>/', get_note_details, name='note_details'),
    path('upload-pdf/', upload_pdf),
    path('quiz-completed/', quiz_completed, name='quiz_completed'),
    
    # Authentication URLs
    path('auth/register/', RegisterView.as_view(), name='register'),
    path('auth/login/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('auth/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('auth/me/', CurrentUserView.as_view(), name='current_user'),
    
    # Profile URLs
    path('profile/', UserProfileView.as_view(), name='user_profile'),
    
    # Notification URLs
    path('notifications/', NotificationListView.as_view(), name='notifications'),
    path('notifications/<int:pk>/mark-read/', NotificationMarkReadView.as_view(), name='notification_mark_read'),
    path('notifications/mark-all-read/', NotificationMarkAllReadView.as_view(), name='notification_mark_all_read'),
    path('notifications/<int:pk>/delete/', NotificationDeleteView.as_view(), name='notification_delete'),
]
