from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
import os

# Create your views here.

# class Project(object):
#     def __init__(self,**kwargs):
#         self.image = kwargs['image']
#         self.title = kwargs['title']
#         self.github = kwargs['github']
#         self.url = kwargs['url']
#         self.category = kwargs['category']
#         self.techs = kwargs['techs']
#         self.description = kwargs['description']

def home(request):
    skill_images = ['django_image.png', 'nodejs_image.png', 'pytorch_image.png', 'flask_image.png', 'react_image.png', 'tensorflow_image.png','mysql_image.png','mongodb_image.jpg','numpy_image.png','firebase_image.webp']
    skill_images = list(map(lambda x:'images/'+x,skill_images))

    projects = [{
        'image':'images/pytorch_image.png',
        'title':'Mask-NoMask detection',
        'github':'https://github.com/nirbhay-design/mask-nomask-classification',
        'url':"https://mask-nomask-sharma59.herokuapp.com/",
        'category':'ML/DL',
        'techs':'Python, Pytorch, Flask, OpenCV',
        'description':'the project is build using pytorch library and the final trained model is then used for real time detection using openCV and also the testing can be done from web application build using flask'
    },

    {
        'image':'images/django_image.png',
        'title':'Email-app',
        'github':'https://github.com/nirbhay-design/django_email_app',
        'url':'https://email-app-sharma59.herokuapp.com/',
        'category':'web-dev',
        'techs':'Django, Python, Sqlite3',
        'description':'the project is build using Django framework in python and the database used is sqlite3, the project implements the functionalities of email application (inbox, compose, sent, deleted mails) etc.'
    },

    {
        'image':'images/react_image.png',
        'title':'PRA-Visulaizer',
        'github':'https://github.com/nirbhay-design/pra-visualizer',
        'url':'https://pra-visualizer.web.app/',
        'category':'web-dev',
        'techs':'React, Javascript, HTML/CSS, JSX',
        'description':'The project consists of visualization of various page replacement algorithms such as (FIFO, LRU, OPR) etc. given number of frames and demand pages, the app can visualize how various algorithms handle page replacement.'
    },

    {
        'image':'images/tensorflow_image.png',
        'title':'Exploring TfLite',
        'github':'https://github.com/nirbhay-design/embedded-project',
        'url':None,
        'category':'ML/DL',
        'techs':'Python, Tensorflow, TfLite',
        'description':'The project aims at exploring TfLite framework for reducing the model size, for inference purposes. The project implements 3 tasks and compared performance between tflite model and normal tensorflow model'
    },

    {
        'image':'images/flask_image.png',
        'title':'Email-app',
        'github':'https://github.com/nirbhay-design/email-service',
        'url':'https://email-services-sharma59.herokuapp.com/',
        'category':'web-dev',
        'techs':'Python, Flask, Mysql',
        'description':'The project implements email functionalities (Inbox, Compose, Sent, Deleted etc.) in Flask and use mysql as database'
    },

    {
        'image':'images/pytorch_image.png',
        'title':'CNNAlgos-Comparison',
        'github':'https://github.com/nirbhay-design/CNNAlgosComparison',
        'url':None,
        'category':'ML/DL',
        'techs':'Python, Pytorch',
        'description':'The project consists of various deep CNN architectures (coded from scratch) on Retinal eye disease dataset (kaggle), and performed a comparative study among these deep architectures'
    },

    {
        'image':'images/pytorch_image.png',
        'title':'Image Colorization',
        'github':'https://github.com/nirbhay-design/dlops-project',
        'url':None,
        'category':'ML/DL',
        'techs':'Python, Pytorch',
        'description':'The project aims at implementing Pix2Pix GAN architecture from scratch on RGB and LAB image format, to convert a black and white image to colored image'
    }

    ,

    {
        'image':'images/nodejs_image.png',
        'title':'Video-conferencing-app',
        'github':'https://github.com/nirbhay-design/CN-Project',
        'url':'https://stormy-peak-75848.herokuapp.com/',
        'category':'Web-dev',
        'techs':'Nodejs, ejs',
        'description':'The project aims at implementing live video call and chat application using nodejs and ejs'
    }
    
    ]

    skills = [{'type':'Languages','list':['Python','C/C++','HTML/CSS','Java','Javascript','Haskell','Prolog','Julia','Bash'],'image':'images/python_image.png'},
            {'type':'Developement','list':['Django','Flask','React','Nodejs','Firebase','Mongodb','Mysql','Heroku','Git/Github'],'image':'images/django_image.png'},
            {'type':'ML/Data Science','list':['Pytorch','Tensorflow','Matplotlib','Pandas','Numpy','OpenCV','Sklearn'],'image':'images/pytorch_image.png' }]
    
    experience = [
        {
            'under':'Dr. Angshuman Paul',
            'title':'Light weight CNN architecture for chest Xray diagnoses',
            'duration':'june2021-may2022',
            'image':'images/iitj.png',
            'position':'Research Intern'
        },
        {
            'under':'Pradeep Mishra, Kaushik Raghupathruni, Shogo Yamashita',
            'title':'Split Neural Network Models',
            'duration':'june2022-Today',
            'image':'images/exawizards.png',
            'position':'AI Engineer (Intern)'
        }
    ]

    return render(request,'home.html',{'projects':projects,'skills':skills,'experience':experience})

def resume(request):
    return render(request,'resume.html')