from django.db.models.signals import post_save
from django.dispatch import receiver
from django.db.models import Avg
from .models import UserAnswer, StudyPlan, Notification, TopicMastery, Question

@receiver(post_save, sender=StudyPlan)
def notify_study_plan_created(sender, instance, created, **kwargs):
    """
    Notify user when a new study plan is generated
    """
    if created:
        Notification.objects.create(
            user=instance.user,
            message=f"📊 Your personalized study plan for '{instance.lecture_note.title}' is ready! Check the Study Plan page to view it."
        )


@receiver(post_save, sender=TopicMastery)
def notify_mastery_milestone(sender, instance, **kwargs):
    """
    Notify user when they achieve high mastery in a topic
    """
    # Only notify when mastery crosses 80% threshold
    if instance.mastery >= 0.80:
        # Check if we already sent this notification
        recent_notification = Notification.objects.filter(
            user=instance.user,
            message__contains=f"mastered {instance.topic}"
        ).exists()
        
        if not recent_notification:
            Notification.objects.create(
                user=instance.user,
                message=f"🎉 Congratulations! You've mastered the topic: **{instance.topic}** with {int(instance.mastery * 100)}% mastery!"
            )


def create_quiz_completion_notification(user, lecture_note, current_score, total_questions):
    """
    Create a smart notification based on quiz performance
    Compares current score with previous attempts
    """
    # Get previous quiz scores for this lecture note
    previous_answers = UserAnswer.objects.filter(
        user=user,
        question__lecture_note=lecture_note
    ).exclude(
        # Exclude current quiz answers (last N answers)
        id__in=UserAnswer.objects.filter(
            user=user,
            question__lecture_note=lecture_note
        ).order_by('-answered_at')[:total_questions].values_list('id', flat=True)
    )
    
    # Calculate previous average score
    if previous_answers.exists():
        previous_correct = previous_answers.filter(is_correct=True).count()
        previous_total = previous_answers.count()
        previous_score = (previous_correct / previous_total * 100) if previous_total > 0 else 0
        
        # Compare scores
        score_diff = current_score - previous_score
        
        if current_score >= 80:
            # High score - Congratulations
            if score_diff > 5:
                message = f"🎉 Outstanding! You scored {current_score:.0f}% on '{lecture_note.title}' - that's {score_diff:.0f}% better than before! Keep up the excellent work! 🌟"
            else:
                message = f"🏆 Excellent work! You scored {current_score:.0f}% on '{lecture_note.title}'! You're mastering this topic! 💪"
        elif current_score >= 60:
            # Medium score
            if score_diff > 0:
                message = f"📈 Good progress! You scored {current_score:.0f}% on '{lecture_note.title}' - improved by {score_diff:.0f}%! Keep practicing to reach mastery! 💡"
            else:
                message = f"💪 You scored {current_score:.0f}% on '{lecture_note.title}'. Review the Study Plan page for personalized tips to improve! 📚"
        else:
            # Lower score - Encouragement
            if score_diff > 0:
                message = f"📚 You scored {current_score:.0f}% on '{lecture_note.title}' - showing improvement! Check the Study Plan page for targeted practice recommendations. You've got this! 🚀"
            else:
                message = f"🌱 You scored {current_score:.0f}% on '{lecture_note.title}'. Don't worry - every expert was once a beginner! Check the Study Plan page for a personalized study plan. Keep learning! 💡"
    else:
        # First attempt - Neutral encouragement
        if current_score >= 80:
            message = f"🎉 Amazing first attempt! You scored {current_score:.0f}% on '{lecture_note.title}'! You're off to a great start! 🌟"
        elif current_score >= 60:
            message = f"👍 Good first attempt! You scored {current_score:.0f}% on '{lecture_note.title}'. Check the Study Plan page to see where you can improve! 📊"
        else:
            message = f"🌱 You scored {current_score:.0f}% on '{lecture_note.title}'. Great job starting your learning journey! Visit the Study Plan page for a personalized study plan. 📚"
    
    # Create the notification
    Notification.objects.create(
        user=user,
        message=message
    )
