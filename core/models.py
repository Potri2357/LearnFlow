# core/models.py

from django.db import models
from django.contrib.auth.models import User

from django.db import models
from django.contrib.auth.models import User

class LectureNote(models.Model):
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to='notes/', null=True, blank=True)
    content = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title


class Question(models.Model):
    lecture_note = models.ForeignKey(LectureNote, on_delete=models.CASCADE, related_name='questions')
    topic = models.CharField(max_length=255, null=True, blank=True)  # NEW
    question_text = models.TextField()

    option_a = models.TextField(null=True, blank=True)
    option_b = models.TextField(null=True, blank=True)
    option_c = models.TextField(null=True, blank=True)
    option_d = models.TextField(null=True, blank=True)

    correct_option = models.CharField(
        max_length=1,
        choices=[("A", "A"), ("B", "B"), ("C", "C"), ("D", "D")],
        null=True,
        blank=True
    )

    explanation = models.TextField(null=True, blank=True)
    difficulty = models.FloatField(default=0.5)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.id} - {self.question_text[:70]}"


class TopicWeakness(models.Model):
    lecture_note = models.ForeignKey(LectureNote, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    topic = models.CharField(max_length=100)
    weakness_score = models.FloatField(default=0.0)

    def __str__(self):
        return f"{self.topic} ({self.weakness_score})"


class TopicMastery(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    lecture_note = models.ForeignKey(LectureNote, on_delete=models.CASCADE)
    topic = models.CharField(max_length=200)
    mastery = models.FloatField(default=0.3)  # 0..1
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("user", "lecture_note", "topic")

    def __str__(self):
        return f"{self.topic}: {self.mastery:.2f}"


class UserAnswer(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    user_answer = models.TextField()  # "A"/"B"/"C"/"D"
    is_correct = models.BooleanField(default=False)
    time_taken = models.IntegerField(default=0)  # seconds
    answered_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Answer by {self.user} to {self.question.id}"



class UserProgress(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    lecture_note = models.ForeignKey(LectureNote, on_delete=models.CASCADE, null=True)
    total_questions = models.IntegerField(default=0)
    correct_answers = models.IntegerField(default=0)

    def __str__(self):
        return f"{self.user.username} - {self.correct_answers}/{self.total_questions}"


class UserStreak(models.Model):
    """
    Tracks consecutive correct answers for a user (optionally per topic or global).
    We'll use per-user global and optional per-topic streak for adaptivity.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    topic = models.CharField(max_length=120, null=True, blank=True)  # null => global streak
    streak = models.IntegerField(default=0)
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("user", "topic")

    def __str__(self):
        return f"{self.user.username} | {self.topic or 'GLOBAL'} streak={self.streak}"
    
class StudyPlan(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    lecture_note = models.ForeignKey(LectureNote, on_delete=models.CASCADE)
    plan_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Plan for {self.user.username} at {self.created_at.strftime('%Y-%m-%d %H:%M')}"


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    bio = models.TextField(blank=True, null=True)
    avatar = models.ImageField(upload_to='avatars/', blank=True, null=True)

    def __str__(self):
        return f"Profile of {self.user.username}"


class Notification(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notifications')
    message = models.TextField()
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Notification for {self.user.username}: {self.message[:20]}"


class QuizAttempt(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='quiz_attempts')
    lecture_note = models.ForeignKey(LectureNote, on_delete=models.CASCADE)
    score = models.IntegerField()
    total_questions = models.IntegerField()
    completed_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.lecture_note.title} - {self.score}/{self.total_questions}"


class Badge(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='badges')
    name = models.CharField(max_length=100)
    description = models.TextField()
    icon = models.CharField(max_length=50, default="🏆")  # Emoji or icon name
    earned_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.name}"


# Exam Preparation Models
class ExamSyllabus(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='exam_syllabi')
    title = models.CharField(max_length=255)
    content = models.TextField()  # Extracted text from PDF or direct input
    file = models.FileField(upload_to='syllabi/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name_plural = "Exam Syllabi"
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.title} - {self.user.username}"


class PreviousQuestionPaper(models.Model):
    exam_syllabus = models.ForeignKey(ExamSyllabus, on_delete=models.CASCADE, related_name='previous_papers')
    file = models.FileField(upload_to='previous_papers/')
    content = models.TextField()  # Extracted text from PDF
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"Paper for {self.exam_syllabus.title}"


class ExamQuestion(models.Model):
    exam_syllabus = models.ForeignKey(ExamSyllabus, on_delete=models.CASCADE, related_name='exam_questions')
    question_text = models.TextField()
    answer = models.TextField()
    marks = models.IntegerField()
    priority = models.IntegerField()  # 1 = highest priority
    topic = models.CharField(max_length=255, blank=True)
    is_from_pattern = models.BooleanField(default=False)  # Based on previous papers analysis
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['priority', '-marks']

    def __str__(self):
        return f"Q{self.priority} ({self.marks}m) - {self.question_text[:50]}"


class ExamConfiguration(models.Model):
    exam_syllabus = models.ForeignKey(ExamSyllabus, on_delete=models.CASCADE, related_name='configurations')
    total_marks = models.IntegerField()
    num_questions = models.IntegerField()
    mark_distribution = models.JSONField()  # {"2": 5, "5": 3, "10": 2} - marks: count
    secure_centum_mode = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Config for {self.exam_syllabus.title} - {self.total_marks}m"

