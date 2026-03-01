from django.apps import AppConfig
import nltk
import os

class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'
    
    def ready(self):
        import core.signals  # Register signals
        
        # Ensure NLTK data is downloaded
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt')
            nltk.download('punkt_tab')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

