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

    projects = [
    {
        'image':'images/iisc.jpg',
        'title':'Semester-1 @ IISc',
        'github':'https://github.com/nirbhay-design/IISC_SEM1.git',
        'url':None,
        'category':'ML/DL',
        'techs':'Numpy, Python',
        'description': 'The repo contains the first semester assignments and PYQs for STOMA, LAA, DSA, CMO'    
    },

    {
        'image':'images/pytorch_image.png',
        'title':'Image Captioning using Detection Transformer (DeTR)',
        'github':'https://github.com/nirbhay-design/image-caption-detr.git',
        'url':None,
        'category':'ML/DL',
        'techs':'Pytorch, Transformers, Python',
        'description': 'Implemented modified DeTR from scratch in pytorch for image captioning task. Trained DeTR on Flickr30k dataset for 500 epochs and achieved a BLEU score of 57.36 on Flickr8k dataset'    
    }, 

    {
        'image':'images/pytorch_image.png',
        'title':'Vision Transformers Implementation',
        'github':'https://github.com/nirbhay-design/Transformers-Implementation.git',
        'url':None,
        'category':'ML/DL',
        'techs':'Pytorch, Vision Transformers, Python',
        'description': 'Implemented 11 SOTA research papers on vision transformers variants like Swin Transformer, Pyramid ViT, Convolution ViT etc. for Image Classification from scratch in pytorch'    
    },    
    
    {
        'image':'images/pytorch_image.png',
        'title':'Regularizing Federated Learning via Adversarial Model Perturbations',
        'github':'https://github.com/nirbhay-design/DAI_Project',
        'url':None,
        'category':'ML/DL',
        'techs':'FL, Pytorch, Python, AMP',
        'description': 'The project aims at implementing the state of the art methods for Federated Learning (FL) like scaffold, FedNTD, FedProx, FedAvg and regularize the client using adversarial model perturbations to reach flat minima.'
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
        'image':'images/django_image.png',
        'title':'Email-app',
        'github':'https://github.com/nirbhay-design/django_email_app',
        'url':None,
        'category':'web-dev',
        'techs':'Django, Python, Sqlite3',
        'description':'the project is build using Django framework in python and the database used is sqlite3, the project implements the functionalities of email application (inbox, compose, sent, deleted mails) etc.'
    },

    # {
    #     'image':'images/flask_image.png',
    #     'title':'Email-app',
    #     'github':'https://github.com/nirbhay-design/email-service',
    #     'url':None,
    #     'category':'web-dev',
    #     'techs':'Python, Flask, Mysql',
    #     'description':'The project implements email functionalities (Inbox, Compose, Sent, Deleted etc.) in Flask and use mysql as database'
    # },

    # {
    #     'image':'images/nodejs_image.png',
    #     'title':'Video-conferencing-app',
    #     'github':'https://github.com/nirbhay-design/CN-Project',
    #     'url':None,
    #     'category':'web-dev',
    #     'techs':'Nodejs, ejs',
    #     'description':'The project aims at implementing live video call and chat application using nodejs and ejs'
    # }

    {
        'image':'images/pytorch_image.png',
        'title':'Mask-NoMask detection',
        'github':'https://github.com/nirbhay-design/mask-nomask-classification',
        'url':None,
        'category':'ML/DL',
        'techs':'Python, Pytorch, Flask, OpenCV',
        'description':'the project is build using pytorch library and the final trained model is then used for real time detection using openCV and also the testing can be done from web application build using flask'
    },

    # {
    #     'image':'images/tensorflow_image.png',
    #     'title':'Exploring TfLite',
    #     'github':'https://github.com/nirbhay-design/embedded-project',
    #     'url':None,
    #     'category':'ML/DL',
    #     'techs':'Python, Tensorflow, TfLite',
    #     'description':'The project aims at exploring TfLite framework for reducing the model size, for inference purposes. The project implements 3 tasks and compared performance between tflite model and normal tensorflow model'
    # }
    
    ]

    skills = [{'type':'Languages','list':['Python','C/C++','HTML/CSS','Java','Javascript','Haskell','Prolog','Julia','Bash'],'image':'images/python_image.png'},
            {'type':'Developement','list':['Docker', 'Django','Flask','React','Nodejs','Firebase','Mongodb','Mysql','Heroku','Git/Github'],'image':'images/django_image.png'},
            {'type':'ML/Data Science','list':['Pytorch','Tensorflow','Matplotlib','Pandas','Numpy','OpenCV','Sklearn', "Deep Learning", "Computer Vision"],'image':'images/pytorch_image.png' }]
    
    experience = [
        {
            'under':'Dr. Deepak Mishra',
            'title':'FedAgPD: Aggregation-Assisted Proxyless Distillation',
            'duration':'August2022 - May2023',
            'image':'images/iitj.png',
            'institute':'IIT Jodhpur' ,
            'desc':"""Proposed a novel FL Framework FedAgPD to simultaneously handle model and data heterogeneity
                Leveraged Deep Mutual Learning at Client and Aggregation followed by Gaussian Noise based data free
                distillation at the Server, eliminating need of proxy dataset or GAN's
                FedAgPD achieved 2x better performance compared to SOTA FL algorithms like FedDF, FedMD, Kt-pfl"""
        },

        {
            'under':'Dr. Angshuman Paul',
            'title':'Extremely Lightweight CNN for Chest X-Ray Diagnosis',
            'duration':'June2021 - May2022',
            'image':'images/iitj.png',
            'institute':'IIT Jodhpur',
            'desc':"""Designed a novel Lightweight CNN model (ExLNet) for the abnormal detection of Chest Radiographs
Fused Squeeze and Excitation blocks with Depth-wise convolution to create DCISE layer as a component of
ExLNet, which outperforms SOTA models like Mobilenet, Shufflenet on NIH, VinBig medical datasets"""
        },

        {
            'under':'Dr. Angshuman Paul',
            'title':'Cell Detection and Classification',
            'duration':'August2022 - May2023',
            'image':'images/iitj.png',
            'institute':'IIT Jodhpur',
            'desc': """Detected and classified cells data sample into necrotic and apoptotic cells
Finetuned various SOTA object detectors such as YOLO, SSD, RetinaNet, DeTR
Achieved remarkable results using DeTR with a Mean Average Precision (MAP) of 40.0"""    
        },

        {
            'under':'Pradeep Mishra, Kaushik Raghupathruni, Shogo Yamashita',
            'title':'Split Neural Network Models',
            'duration':'June2022 - July2022',
            'image':'images/exawizards.png',
            'institute':'Exawizards',
            'desc':"""Worked on Split Neural Network ML paradigm and Splitted Mask-RCNN, FCN_Resnet50, YOLOv5 models for
Instance segmentation, segmentation, face detection tasks
Implemented Autoencoder model for efficient image compression to latent space and setup Pysyft to
communicate latents from Jetson Nano to GPU server, preserving data privacy at Jetson Nano"""        
        },
         {
            'under':'Sahil Bajaj',
            'title':'Content Generation, Avatar Generation',
            'duration':'July2023 - July2024',
            'image':'images/Faayaastu.jpg',
            'institute':'Faaya Astu',
            'desc':"""Trained Stable Diffusion ControlNet models on Lineart and Colorbox control on VastAI GPU instance and
deployed them on RunPod for more flexibility and control on print generation
Trained Low Rank Adaptation (LoRA) with Kohya_SS for custom face and background generation
Experimented with custom ComfyUI workflows with integrated ControlNet, LoRA, InstantID models
Containerised ComfyUI with Docker and deployed them as Serverless Endpoints on RunPod and exposed endpoint
APIs to AWS Lambda to create APIs for APP using AWS API gateway"""
        }
    ]

    return render(request,'home.html',{'projects':projects,'skills':skills,'experience':experience})

def resume(request):
    return render(request,'resume.html')