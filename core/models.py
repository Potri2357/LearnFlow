# core/models.py

from django.db import models
from django.contrib.auth.models import User

from django.db import models
from django.contrib.auth.models import User

class LectureNote(models.Model):
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    title = models.CharField(max_length=255)
    subject = models.CharField(max_length=100, blank=True, null=True)
    file = models.FileField(upload_to='notes/', null=True, blank=True)
    content = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    # AI-generated study aids
    study_notes = models.TextField(blank=True, null=True)  # Markdown formatted notes
    formulas = models.JSONField(default=list, blank=True)   # List of {name, formula, description}
    key_points = models.JSONField(default=list, blank=True) # List of key point strings

    def __str__(self):
        return self.title


NOTE_COLORS = [
    ('#FFF9C4', 'Yellow'),
    ('#BBDEFB', 'Blue'),
    ('#C8E6C9', 'Green'),
    ('#FFCCBC', 'Orange'),
    ('#E1BEE7', 'Purple'),
    ('#F8BBD9', 'Pink'),
]

class StickyNote(models.Model):
    """User-created class notes / sticky notes attached to a lecture (or standalone)."""
    NOTE_TYPES = [
        ('lecture', 'Lecture Note'),
        ('hint', 'Hint'),
        ('exam', 'Exam Note'),
        ('formula', 'Formula'),
    ]
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sticky_notes')
    lecture_note = models.ForeignKey(LectureNote, on_delete=models.CASCADE, related_name='sticky_notes', null=True, blank=True)
    title = models.CharField(max_length=255, default='Class Note')
    content = models.TextField(blank=True)
    color = models.CharField(max_length=20, default='#FFF9C4')  # Hex color
    note_type = models.CharField(max_length=20, choices=NOTE_TYPES, default='lecture')
    is_pinned = models.BooleanField(default=False)
    page_number = models.IntegerField(null=True, blank=True)  # PDF page reference
    source_text = models.TextField(blank=True, null=True)  # Original text dragged from PDF
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-is_pinned', '-updated_at']

    def __str__(self):
        return f"{self.title} ({self.user.username})"



class Question(models.Model):
    BLOOMS_LEVELS = [
        ('remember', 'Remember'),
        ('understand', 'Understand'),
        ('apply', 'Apply'),
        ('analyze', 'Analyze'),
        ('evaluate', 'Evaluate'),
        ('create', 'Create'),
    ]
    QUESTION_TYPES = [
        ('mcq', 'Multiple Choice'),
        ('true_false', 'True/False'),
        ('fill_blank', 'Fill in the Blank'),
        ('assertion_reason', 'Assertion-Reason'),
    ]
    lecture_note = models.ForeignKey(LectureNote, on_delete=models.CASCADE, related_name='questions')
    topic = models.CharField(max_length=255, null=True, blank=True)
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

    # Bloom's Taxonomy & Quality
    blooms_level = models.CharField(max_length=20, choices=BLOOMS_LEVELS, default='understand')
    question_type = models.CharField(max_length=20, choices=QUESTION_TYPES, default='mcq')
    is_high_yield = models.BooleanField(default=False)
    relevance_score = models.FloatField(default=5.0)  # AI-scored 1-10
    is_starred = models.BooleanField(default=False)

    # Attempt tracking
    attempt_count = models.IntegerField(default=0)
    correct_count = models.IntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.id} - {self.question_text[:70]}"


class Flashcard(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='flashcards')
    lecture_note = models.ForeignKey(LectureNote, on_delete=models.CASCADE, related_name='flashcards', null=True, blank=True)
    front = models.TextField()
    back = models.TextField()
    
    # SM-2 fields
    ease_factor = models.FloatField(default=2.5)
    interval = models.IntegerField(default=0)
    repetitions = models.IntegerField(default=0)
    next_review_date = models.DateTimeField(auto_now_add=True)
    
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Card for {self.user.username}: {self.front[:30]}"


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
    school = models.CharField(max_length=255, blank=True, null=True)
    grade = models.CharField(max_length=50, blank=True, null=True)
    subjects = models.JSONField(default=list, blank=True)  # List of subjects
    preferences = models.JSONField(default=dict, blank=True)  # User preferences dict

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

class AIResponseCache(models.Model):
    lecture_note = models.ForeignKey('LectureNote', on_delete=models.CASCADE, related_name='ai_caches', null=True, blank=True)
    exam_syllabus = models.ForeignKey('ExamSyllabus', on_delete=models.CASCADE, related_name='ai_caches', null=True, blank=True)
    action_type = models.CharField(max_length=50) # 'summarize', 'generate_mcqs', 'ai_insights', 'study_plan'
    response_data = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = []

    def __str__(self):
        target = self.lecture_note.title if self.lecture_note else (self.exam_syllabus.title if self.exam_syllabus else "Unknown")
        return f"Cache {self.action_type} for {target}"

