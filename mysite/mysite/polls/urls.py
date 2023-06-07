from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', views.emotion_detection, name='emotion-detection'),
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

#path('', views.predict_emotion_view, name='predict-emotion'),