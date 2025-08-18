from django.urls import path
from . import views

urlpatterns = [
    path('', views.chat_view, name='chat'),
    path('api/', views.ChatAPIView.as_view(), name='chat_api'),
]