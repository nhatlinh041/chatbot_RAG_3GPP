from django.db import models
from django.utils import timezone

class ChatMessage(models.Model):
    user_message = models.TextField()
    bot_response = models.TextField()
    timestamp = models.DateTimeField(default=timezone.now)
    
    def __str__(self):
        return f"Chat at {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
    
    class Meta:
        ordering = ['-timestamp']
