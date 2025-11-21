
import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from core.models import User, LectureNote, Notification
from core.signals import create_quiz_completion_notification

try:
    user = User.objects.first()
    note = LectureNote.objects.first()
    
    if not user or not note:
        print("Error: Need at least one user and one lecture note to test.")
    else:
        print(f"Testing with User: {user.username}, Note: {note.title}")
        
        # Count before
        count_before = Notification.objects.filter(user=user).count()
        print(f"Notifications before: {count_before}")
        
        # Trigger notification
        create_quiz_completion_notification(user, note, 85.0, 10)
        
        # Count after
        count_after = Notification.objects.filter(user=user).count()
        print(f"Notifications after: {count_after}")
        
        if count_after > count_before:
            new_notif = Notification.objects.filter(user=user).order_by('-created_at').first()
            print(f"SUCCESS: Created notification: {new_notif.message}")
        else:
            print("FAILURE: No notification created.")

except Exception as e:
    print(f"Exception occurred: {e}")
