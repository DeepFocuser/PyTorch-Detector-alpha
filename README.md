>## ***One Stage Detector***
* [***CenterNet***](https://github.com/DeepFocuser/PyTorch-Detector/tree/master/CenterNet)
* [***FaceDetector and Recognition***]()
    * [***Face Detector based on CenterNet***](https://github.com/DeepFocuser/PyTorch-Detector/tree/master/CenterFaceNet)
    * [***Face Detector based on CenterNet FPN***](https://github.com/DeepFocuser/PyTorch-Detector/tree/master/CenterFaceNet_FPN)
    * [***Face Net***](https://github.com/DeepFocuser/PyTorch-Detector/tree/master/FaceNet)
    * [***Face Net with ArcFace Loss***](https://github.com/DeepFocuser/PyTorch-Detector/tree/master/FaceNetWithAngle)
* [***YoloV3***](https://github.com/DeepFocuser/PyTorch-Detector/tree/master/YoloV3)
* [***GaussianYoloV3***](https://github.com/DeepFocuser/PyTorch-Detector/tree/master/GaussianYoloV3)

>## ***classification***
* [***classification***](https://github.com/DeepFocuser/PyTorch-Detector/tree/master/classification)

>## ***Development environment***
* OS : ubuntu linux 16.04 LTS
* Graphic card / driver : rtx 2080ti / 418.56
* Anaconda version : 4.7.12
* pytorch version : 1.7.0
    * Configure Run Environment
        1. Create a virtual environment
        ```cmd
        jg@JG:~$ conda create -n pytorch python==3.7.3
        ```
        2. Install the required module 
        ```cmd
        jg@JG:~$ conda activate pytorch 
        (pytorch) jg@JG:~$ conda install pytorch torchvision cudatoolkit=10.1 -c pytorch 
        (pytorch) jg@JG:~$ pip install matplotlib tensorboard torchsummary plotly mlflow opencv-python==4.1.1.26 tqdm PyYAML --pre --upgrade
        ```
>## ***Author*** 

* medical18@naver.com / JONGGON
