from django.contrib import admin
from .models import (
    LectureNote, Question, TopicWeakness, UserAnswer, UserProgress,
    TopicMastery, UserStreak, UserProfile, Notification, StudyPlan,
    QuizAttempt, Badge, ExamSyllabus, PreviousQuestionPaper, ExamQuestion, ExamConfiguration
)


@admin.register(LectureNote)
class LectureNoteAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "created_at")


@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "lecture_note",
        "question_text",
        "correct_option",
        "difficulty",
        "created_at"
    )
    list_filter = ("lecture_note", "correct_option")
    search_fields = ("question_text",)


@admin.register(TopicWeakness)
class TopicWeaknessAdmin(admin.ModelAdmin):
    list_display = ("id", "lecture_note", "user", "topic", "weakness_score")
    list_filter = ("lecture_note", "user")
    search_fields = ("topic",)


@admin.register(UserAnswer)
class UserAnswerAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "question", "user_answer", "is_correct", "answered_at")
    list_filter = ("is_correct", "user")
    search_fields = ("user__username",)


@admin.register(UserProgress)
class UserProgressAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "total_questions", "correct_answers")
    search_fields = ("user__username",)


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "bio")
    search_fields = ("user__username",)


@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "message", "is_read", "created_at")
    list_filter = ("is_read", "created_at")
    search_fields = ("user__username", "message")
    
admin.site.register(TopicMastery)
admin.site.register(UserStreak)
admin.site.register(StudyPlan)
admin.site.register(QuizAttempt)
admin.site.register(Badge)

# Exam Preparation
admin.site.register(ExamSyllabus)
admin.site.register(PreviousQuestionPaper)
admin.site.register(ExamQuestion)
admin.site.register(ExamConfiguration)


