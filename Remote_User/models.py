from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)


class student_attendance_results(models.Model):

    ADay= models.CharField(max_length=30)
    RNo = models.CharField(max_length=30)
    Weekday= models.CharField(max_length=30)
    Student_Location= models.CharField(max_length=30)
    Location_Number= models.CharField(max_length=30)
    Country= models.CharField(max_length=30)
    Total_Student= models.CharField(max_length=30)
    Absence= models.CharField(max_length=30)
    Prediction= models.CharField(max_length=30)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



