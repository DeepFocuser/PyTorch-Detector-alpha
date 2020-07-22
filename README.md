>## ***One Stage Detector***
* [***Center Detector***](https://github.com/DeepFocuser/PyTorch-Detector/tree/master/CenterNet)

>## ***Development environment***
* OS : ubuntu linux 16.04 LTS
* Graphic card / driver : rtx 2080ti / 418.56
* Anaconda version : 4.7.12
    * Configure Run Environment
        1. Create a virtual environment
        ```cmd
        jg@JG:~$ conda create -n pytorch python==3.7.3
        ```
        2. Install the required module
        ```cmd
        jg@JG:~$ conda activate pytorch
        (pytorch) jg@JG:~$ conda install pytorch torchvision cudatoolkit=10.1 -c pytorch 
        (pytorch) jg@JG:~$ pip install torchsummary plotly mlflow opencv-python==4.1.1.26 tqdm PyYAML --pre --upgrade
        ```

>## ***Author*** 

* medical18@naver.com / JONGGON
