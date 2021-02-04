'''
Face recognition 정의

참고 사이트
https://machinelearningmastery.com/introduction-to-deep-learning-for-face-recognition/
https://tech.kakaoenterprise.com/63

Face recognition is a broad problem of identifying or verifying people in photographs and videos.
'''
import mlflow as ml
import torch
import yaml

import test_verification
import train

# nms 구현하면 끝
stream = yaml.load(open("configs/detector.yaml", "rt", encoding='UTF8'), Loader=yaml.SafeLoader)

# dataset
parser = stream['Dataset']
train_dataset_path = parser['train']
valid_dataset_path = parser['valid']
test_dataset_path = parser['test']
test_method = parser['test_method']
threshold = parser['threshold']
test_weight_path = parser['test_weight_path']
test_save_path = parser['save_path']
save_flag = parser['save_flag']
show_flag = parser['show_flag']

# model
parser = stream['model']
training = parser["training"]
load_name = parser["load_name"]
save_period = parser["save_period"]
load_period = parser["load_period"]
input_size = parser["input_size"]
base = parser["ResNetbase"]
pretrained_base = parser["pretrained_base"]

# hyperparameters
parser = stream['hyperparameters']

image_mean = parser["image_mean"]
image_std = parser["image_std"]
embedding = parser["embedding"]
margin = parser["margin"]
semi_hard_negative = parser["semi_hard_negative"]

epoch = parser["epoch"]
batch_size = parser["batch_size"]
batch_log = parser["batch_log"]
subdivision = parser["subdivision"]
data_augmentation = parser["data_augmentation"]
num_workers = parser["num_workers"]
optimizer = parser["optimizer"]
learning_rate = parser["learning_rate"]
weight_decay = parser["weight_decay"]
decay_lr = parser["decay_lr"]
decay_step = parser["decay_step"]

# gpu vs cpu
parser = stream['context']
using_cuda = parser["using_cuda"]

parser = stream['validation']
valid_size = parser["valid_size"]
eval_period = parser["eval_period"]
tensorboard = parser["tensorboard"]

parser = stream['mlflow']
using_mlflow = parser["using_mlflow"]
run_name = parser["run_name"]

if torch.cuda.device_count() > 0 and using_cuda:
    GPU_COUNT = torch.cuda.device_count()
else:
    GPU_COUNT = 0

# window 운영체제에서 freeze support 안나오게 하려면, 아래와 같이 __name__ == "__main__" 에 해줘야함.
if __name__ == "__main__":

    print("\n실행 경로 : " + __file__)
    if training:
        if using_mlflow:
            ml.set_tracking_uri("./mlruns")  # mlruns가 기본 트래킹이다.
            ex_id = ml.set_experiment("RESNET" + str(base))
            ml.start_run(run_name=run_name, experiment_id=ex_id)

            ml.log_param("image order", "RGB")
            ml.log_param("image range before normalization", "0~1")
            ml.log_param("image mean RGB", image_mean)
            ml.log_param("image std RGB", image_std)
            ml.log_param("verification threshold", threshold)
            ml.log_param("embedding vector size", embedding)
            ml.log_param("margin", margin)
            ml.log_param("semi hard negative", semi_hard_negative)

            ml.log_param("height", input_size[0])
            ml.log_param("width", input_size[1])
            ml.log_param("pretrained_base", pretrained_base)
            ml.log_param("train dataset path", train_dataset_path)
            ml.log_param("valid dataset path", valid_dataset_path)
            ml.log_param("test dataset path", test_dataset_path)
            ml.log_param("epoch", epoch)
            ml.log_param("batch size", batch_size)
            ml.log_param("data augmentation", data_augmentation)
            ml.log_param("optimizer", optimizer)
            ml.log_param("num_workers", num_workers)

            ml.log_param("learning rate", learning_rate)
            ml.log_param("weight decay", weight_decay)
            ml.log_param("decay lr", decay_lr)
            ml.log_param("decay step", decay_step)
            ml.log_param("using_cuda", using_cuda)
            ml.log_param("save_period", save_period)

        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True # 그래프가 변하는 경우 학습 속도 느려질수 있음.
        train.run(mean=image_mean,
                  std=image_std,
                  threshold = threshold,
                  embedding=embedding,
                  margin=margin,
                  semi_hard_negative=semi_hard_negative,
                  epoch=epoch,
                  input_size=input_size,
                  batch_size=batch_size,
                  batch_log=batch_log,
                  subdivision=subdivision,
                  train_dataset_path=train_dataset_path,
                  valid_dataset_path=valid_dataset_path,
                  data_augmentation=data_augmentation,
                  num_workers=num_workers,
                  optimizer=optimizer,
                  save_period=save_period,
                  load_period=load_period,
                  learning_rate=learning_rate,
                  weight_decay=weight_decay,
                  decay_lr=decay_lr,
                  decay_step=decay_step,
                  GPU_COUNT=GPU_COUNT,
                  base=base,
                  pretrained_base=pretrained_base,

                  valid_size=valid_size,
                  eval_period=eval_period,
                  tensorboard=tensorboard,
                  using_mlflow=using_mlflow)

        if using_mlflow:
            ml.end_run()
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if test_method == "verification":
            test_verification.run(mean=image_mean,
                                  std=image_std,
                                  threshold = threshold,
                                  load_name=load_name, load_period=load_period, GPU_COUNT=GPU_COUNT,
                                  test_weight_path=test_weight_path,
                                  test_dataset_path=test_dataset_path, num_workers=num_workers,
                                  test_save_path=test_save_path,
                                  show_flag=show_flag,
                                  save_flag=save_flag)
        elif test_method == "identification":
            raise AttributeError
        else:
            raise AttributeError
