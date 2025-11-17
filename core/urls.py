from django.urls import path
from .views import upload_lecture_note, generate_questions, submit_answer, weak_topics, progress
from django.urls import path
from .views import generate_mcqs, get_quiz_questions, submit_mcq_answer ,  adaptive_quiz_start , generate_study_plan
from .views import analytics_for_note, recent_weak_topics, next_actions , ai_insights
from django.urls import path
from . import views
from django.urls import path, include

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
    path('upload/', views.upload_lecture_note, name='upload_note'),
    path('questions/<int:note_id>/generate/', views.generate_questions, name='generate_questions'),
    path('answer/', views.submit_answer, name='submit_answer'),
    path('weak-topics/', views.weak_topics, name='weak_topics'),
    path('progress/', views.progress, name='progress'),
    path('analytics/<int:note_id>/', views.analytics_for_note, name='analytics'),
    path('study-plan/<int:note_id>/', views.generate_study_plan, name='study_plan'),  # if you have this
    # optional n8n webhook receiver
    # path('webhook/n8n/<str:flow_id>/', views.n8n_webhook, name='n8n_webhook'),
]
