from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView
from .views import (
    upload_lecture_note, generate_questions, submit_answer, weak_topics, progress,
    generate_mcqs, get_quiz_questions, submit_mcq_answer, adaptive_quiz_start, generate_study_plan,
    analytics_for_note, recent_weak_topics, next_actions, ai_insights,
    upload_pdf, get_note_details, quiz_completed, user_profile, get_dashboard_stats, get_all_questions,
    RegisterView, NotificationListView, NotificationMarkReadView, 
    NotificationMarkAllReadView, NotificationDeleteView, CurrentUserView,
    LectureNoteListView, LectureNoteDetailView,
    generate_flashcards, get_flashcards, review_flashcard, update_question, summarize_lecture,
    get_weak_topic_explanation, get_lectures_by_topics, generate_lecture_study_aids,
    # Exam preparation views
    upload_exam_syllabus, upload_previous_papers, generate_exam_questions,
    get_exam_questions, update_exam_question, delete_exam_question, list_exam_syllabi,
    generate_exam_strategy, generate_video_explanation,
    sticky_notes, sticky_note_detail
)
from .ai_tutor_views import concept_coach_chat, evaluate_assignment
from .auth_views import CustomTokenObtainPairView


urlpatterns = [
    path("dashboard/stats/", get_dashboard_stats, name='dashboard_stats'),
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
    path('questions/all/', get_all_questions, name='get_all_questions'),
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
    path('auth/login/', CustomTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('auth/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('auth/me/', CurrentUserView.as_view(), name='current_user'),
    
    # Profile URLs
    path('profile/', user_profile, name='user_profile'),
    
    # Notification URLs
    path('notifications/', NotificationListView.as_view(), name='notifications'),
    path('notifications/<int:pk>/mark-read/', NotificationMarkReadView.as_view(), name='notification_mark_read'),
    path('notifications/mark-all-read/', NotificationMarkAllReadView.as_view(), name='notification_mark_all_read'),
    path('notifications/<int:pk>/delete/', NotificationDeleteView.as_view(), name='notification_delete'),
    
    # Lecture URLs
    path('lectures/', LectureNoteListView.as_view(), name='lecture_list'),
    path('lectures/<int:pk>/', LectureNoteDetailView.as_view(), name='lecture_detail'),
    path('lectures/<int:note_id>/generate-study-aids/', generate_lecture_study_aids, name='generate_study_aids'),
    
    # Sticky Notes
    path('sticky-notes/', sticky_notes, name='sticky_notes_list_create'),
    path('sticky-notes/<int:note_id>/', sticky_note_detail, name='sticky_note_detail'),

    # Question Management
    path('questions/<int:question_id>/update/', update_question, name='update_question'),

    # Flashcards
    path('flashcards/generate/', generate_flashcards, name='generate_flashcards'),
    path('flashcards/', get_flashcards, name='get_flashcards'),
    path('flashcards/<int:card_id>/review/', review_flashcard, name='review_flashcard'),
    
    # Lecture Summarization
    path('lectures/<int:note_id>/summarize/', summarize_lecture, name='summarize_lecture'),
    
    # Weak Topic Explanation
    path('weak-topic/explain/', get_weak_topic_explanation, name='weak_topic_explanation'),
    
    # Multi-Lecture Quiz
    path('lectures/by-topics/', get_lectures_by_topics, name='get_lectures_by_topics'),
    
    # Exam Preparation
    path('exam/syllabi/', list_exam_syllabi, name='list_exam_syllabi'),
    path('exam/syllabus/upload/', upload_exam_syllabus, name='upload_exam_syllabus'),
    path('exam/syllabus/<int:syllabus_id>/papers/', upload_previous_papers, name='upload_previous_papers'),
    path('exam/syllabus/<int:syllabus_id>/generate/', generate_exam_questions, name='generate_exam_questions'),
    path('exam/syllabus/<int:syllabus_id>/questions/', get_exam_questions, name='get_exam_questions'),
    path('exam/syllabus/<int:syllabus_id>/strategy/', generate_exam_strategy, name='generate_exam_strategy'),
    path('exam/question/<int:question_id>/update/', update_exam_question, name='update_exam_question'),
    path('exam/question/<int:question_id>/delete/', delete_exam_question, name='delete_exam_question'),
    
    # Video Generation
    path('video/generate/', generate_video_explanation, name='generate_video_explanation'),

    # AI Tutor & Evaluator
    path('ai-tutor/chat/', concept_coach_chat, name='concept_coach_chat'),
    path('ai-tutor/evaluate/', evaluate_assignment, name='evaluate_assignment'),
]
