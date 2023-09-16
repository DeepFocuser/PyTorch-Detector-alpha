>## ***One Stage Detector***
* [***CenterNet***](https://github.com/DeepFocuser/PyTorch-Detector/tree/master/CenterNet)
* [***FaceDetector and Recognition***](#)
    * [***Face Detector based on CenterNet***](https://github.com/DeepFocuser/PyTorch-Detector/tree/master/CenterFaceNet)
    * [***Face Detector based on CenterNet FPN***](https://github.com/DeepFocuser/PyTorch-Detector/tree/master/CenterFaceNet_FPN)
    * [***Face Net***](https://github.com/DeepFocuser/PyTorch-Detector/tree/master/FaceNet)
    * [***Face Net with ArcFace Loss***](https://github.com/DeepFocuser/PyTorch-Detector/tree/master/FaceNetWithAngle)
* [***YoloV3***](https://github.com/DeepFocuser/PyTorch-Detector/tree/master/YoloV3)
* [***GaussianYoloV3***](https://github.com/DeepFocuser/PyTorch-Detector/tree/master/GaussianYoloV3)

>## ***Segmentation***
* [***Segmentation***](https://github.com/DeepFocuser/PyTorch-Detector/tree/master/segmentation)

>## ***Classification***
* [***Classification***](https://github.com/DeepFocuser/PyTorch-Detector/tree/master/classification)

>## ***Development environment***
* OS : ubuntu linux 18.04 LTS
* Graphic card / driver : Quadro RTX 5000 / 470.103.01
* miniconda version : 4.11.0
* pytorch version : 1.11.0
    * Configure Run Environment
        1. Create a virtual environment
        ```cmd
        jg@JG:~$ conda create -n pytorch python==3.8.0
        ```
        2. Install the required module 
        ```cmd
        jg@JG:~$ conda activate pytorch 
        (pytorch) jg@JG:~$ conda install pytorch torchvision cudatoolkit cudnn -c pytorch
        (pytorch) jg@JG:~$ pip install matplotlib tensorboard torchsummary plotly mlflow opencv-python tqdm PyYAML --no-cache-dir --pre --upgrade
        
        to do : torchsummary -> torchinfo 코드 
        to do : hiddenlayer -> build_graph 코드 작성
        ```
>## ***Author*** 

* medical18@naver.com / JONGGON
