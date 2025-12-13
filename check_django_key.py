import os
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from dotenv import load_dotenv
load_dotenv()

# Check what key Django is using
from core import views

print(f"Django is using key: {views.GEMINI_API_KEY[:15]}...{views.GEMINI_API_KEY[-8:]}")
print(f"Key length: {len(views.GEMINI_API_KEY)}")

# Also check .env directly
env_key = os.getenv('GEMINI_API_KEY')
print(f"\n.env file has key: {env_key[:15]}...{env_key[-8:]}")

if views.GEMINI_API_KEY == env_key:
    print("\n✅ Keys match! Django is using the latest key from .env")
else:
    print("\n⚠️ Keys DON'T match! Django server needs restart to pick up new key")
