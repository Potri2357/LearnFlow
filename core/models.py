# core/models.py

from django.db import models
from django.contrib.auth.models import User

class LectureNote(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    title = models.CharField(max_length=200)
    content = models.TextField()
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title


class Question(models.Model):
    lecture_note = models.ForeignKey(LectureNote, on_delete=models.CASCADE, related_name='questions')
    question_text = models.TextField()

    # MCQ options
    option_a = models.TextField(null=True, blank=True)
    option_b = models.TextField(null=True, blank=True)
    option_c = models.TextField(null=True, blank=True)
    option_d = models.TextField(null=True, blank=True)

    # Correct answer for MCQ
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


class UserAnswer(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    user_answer = models.TextField()  # stores "A", "B", "C", "D"
    is_correct = models.BooleanField(default=False)
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




class TopicMastery(models.Model):
    """
    Stores per-user mastery for a topic within a lecture note.
    Mastery ranges 0.0 (unknown) to 1.0 (mastered).
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    lecture_note = models.ForeignKey(LectureNote, on_delete=models.CASCADE)
    topic = models.CharField(max_length=120)
    mastery = models.FloatField(default=0.3)  # start modest
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("user", "lecture_note", "topic")

    def __str__(self):
        return f"{self.user.username} | {self.topic} = {self.mastery:.2f}"


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

