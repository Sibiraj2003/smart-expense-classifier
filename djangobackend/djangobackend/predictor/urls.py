from django.urls import path
from .views import PredictExpenseView, dashboard

urlpatterns = [
    path('predict/', PredictExpenseView.as_view()),
    path('dashboard/', dashboard),
]
