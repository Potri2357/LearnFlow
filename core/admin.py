# core/admin.py

from django.contrib import admin
from .models import LectureNote, Question, TopicWeakness, UserAnswer, UserProgress
from .models import TopicMastery, UserStreak
from .models import StudyPlan



@admin.register(LectureNote)
class LectureNoteAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "user", "uploaded_at")
    search_fields = ("title", "content")


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
    
admin.site.register(TopicMastery)
admin.site.register(UserStreak)
admin.site.register(StudyPlan)

