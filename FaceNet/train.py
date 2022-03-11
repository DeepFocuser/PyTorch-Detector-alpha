import glob
import logging
import os
import platform
import time

import cv2
import mlflow as ml
import numpy as np
import torch
import torchvision
from torch.nn import DataParallel
from torch.optim import Adam, RMSprop, SGD, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary as modelsummary
from tqdm import tqdm

from core import PrePostNet
from core import TripletLoss, PairwiseDistance
from core import get_resnet
from core import traindataloader, validdataloader

logfilepath = ""
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)


# 초기화 참고하기
# https://pytorch.org/docs/stable/nn.init.html?highlight=nn%20init#torch.nn.init.kaiming_normal_

def run(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        threshold =1.0,
        embedding = 128,
        epoch=100,
        input_size=[256, 256],
        batch_size=16,
        batch_log=100,
        subdivision=4,
        train_dataset_path="Dataset/train",
        valid_dataset_path="Dataset/valid",
        data_augmentation=True,
        num_workers=4,
        optimizer="ADAM",
        save_period=5,
        load_period=10,
        margin = 0.2,
        semi_hard_negative=True,
        learning_rate=0.001, decay_lr=0.999, decay_step=10,
        weight_decay=0.000001,
        GPU_COUNT=0,
        base=18,
        pretrained_base=True,
        valid_size=8,
        eval_period=5,
        tensorboard=True,
        using_mlflow=True):

    if GPU_COUNT == 0:
        device = torch.device("cpu")
    elif GPU_COUNT == 1:
        device = torch.device("cuda")
    else:
        device = [torch.device(f"cuda:{i}") for i in range(0, GPU_COUNT)]

    if isinstance(device, (list, tuple)):
        context = device[0]
    else:
        context = device

    # 운영체제 확인
    if platform.system() == "Linux":
        logging.info(f"{platform.system()} OS")
    elif platform.system() == "Windows":
        logging.info(f"{platform.system()} OS")
    else:
        logging.info(f"{platform.system()} OS")

    # free memory는 정확하지 않은 것 같고, torch.cuda.max_memory_allocated() 가 정확히 어떻게 동작하는지?
    if isinstance(device, (list, tuple)):
        for i, d in enumerate(device):
            total_memory = torch.cuda.get_device_properties(d).total_memory
            free_memory = total_memory - torch.cuda.max_memory_allocated(d)
            free_memory = round(free_memory / (1024 ** 3), 2)
            total_memory = round(total_memory / (1024 ** 3), 2)
            logging.info(f'{torch.cuda.get_device_name(d)}')
            logging.info(f'Running on {d} / free memory : {free_memory}GB / total memory {total_memory}GB')
    else:
        if GPU_COUNT == 1:
            total_memory = torch.cuda.get_device_properties(device).total_memory
            free_memory = total_memory - torch.cuda.max_memory_allocated(device)
            free_memory = round(free_memory / (1024 ** 3), 2)
            total_memory = round(total_memory / (1024 ** 3), 2)
            logging.info(f'{torch.cuda.get_device_name(device)}')
            logging.info(f'Running on {device} / free memory : {free_memory}GB / total memory {total_memory}GB')
        else:
            logging.info(f'Running on {device}')

    if GPU_COUNT > 0 and batch_size < GPU_COUNT:
        logging.info("batch size must be greater than gpu number")
        exit(0)

    if data_augmentation:
        logging.info("Using Data Augmentation")

    logging.info("training classification")
    input_shape = (1, 3) + tuple(input_size)

    train_dataloader, train_dataset = traindataloader(augmentation=data_augmentation,
                                                      path=train_dataset_path,
                                                      input_size=input_size,
                                                      batch_size=batch_size,
                                                      pin_memory=True,
                                                      num_workers=num_workers,
                                                      shuffle=True, mean=mean, std=std)

    train_update_number_per_epoch = len(train_dataloader)
    if train_update_number_per_epoch < 1:
        logging.warning("train batch size가 데이터 수보다 큼")
        exit(0)

    valid_list = glob.glob(os.path.join(valid_dataset_path, "*"))
    if valid_list:
        valid_dataloader, valid_dataset = validdataloader(path=valid_dataset_path,
                                                          input_size=input_size,
                                                          batch_size=valid_size,
                                                          num_workers=num_workers,
                                                          pin_memory=True,
                                                          shuffle=True, mean=mean, std=std)
        valid_update_number_per_epoch = len(valid_dataloader)
        if valid_update_number_per_epoch < 1:
            logging.warning("valid batch size가 데이터 수보다 큼")
            exit(0)

    optimizer = optimizer.upper()
    if pretrained_base:
        model = str(input_size[0]) + "_" + str(input_size[1]) + "_" + optimizer + "_P" + "RES" + str(base)
    else:
        model = str(input_size[0]) + "_" + str(input_size[1]) + "_" + optimizer + "RES" + str(base)

    if data_augmentation:
        model = model + '_aug'
    model = model + "_embedding" + str(embedding)

    # https://discuss.pytorch.org/t/how-to-save-the-optimizer-setting-in-a-log-in-pytorch/17187
    weight_path = os.path.join("weights", f"{model}")
    param_path = os.path.join(weight_path, f'{model}-{load_period:04d}.pt')

    start_epoch = 0
    net = get_resnet(18, pretrained=pretrained_base, embedding=embedding)

    # https://github.com/sksq96/pytorch-summary
    if GPU_COUNT == 0:
        modelsummary(net.to(context), input_shape[1:], device="cpu")
    elif GPU_COUNT > 0:
        modelsummary(net.to(context), input_shape[1:], device="cuda")

    if tensorboard:
        summary = SummaryWriter(log_dir=os.path.join("torchboard", model), max_queue=10, flush_secs=10)
        summary.add_graph(net.to(context), input_to_model=torch.ones(input_shape, device=context), verbose=False)

    if os.path.exists(param_path):
        start_epoch = load_period
        checkpoint = torch.load(param_path, map_location=context)
        if 'model_state_dict' in checkpoint:
            try:
                net.load_state_dict(checkpoint['model_state_dict'])
            except Exception as E:
                logging.info(E)
            else:
                logging.info(f"loading model_state_dict")

    if start_epoch + 1 >= epoch + 1:
        logging.info("this model has already been optimized")
        exit(0)

    net.to(context)

    if optimizer.upper() == "ADAM":
        trainer = Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
    elif optimizer.upper() == "RMSPROP":
        trainer = RMSprop(net.parameters(), lr=learning_rate, alpha=0.99, weight_decay=weight_decay, momentum=0)
    elif optimizer.upper() == "SGD":
        trainer = SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        logging.error("optimizer not selected")
        exit(0)

    if os.path.exists(param_path):
        # optimizer weight 불러오기
        checkpoint = torch.load(param_path, map_location=context)
        if 'optimizer_state_dict' in checkpoint:
            try:
                trainer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as E:
                logging.info(E)
            else:
                logging.info(f"loading optimizer_state_dict")

    if isinstance(device, (list, tuple)):
        net = DataParallel(net, device_ids=device, output_device=context, dim=0)

    PDLoss = PairwiseDistance(p = 2.0)
    TLLoss = TripletLoss(margin=margin)

    # optimizer
    # https://pytorch.org/docs/master/optim.html?highlight=lr%20sche#torch.optim.lr_scheduler.CosineAnnealingLR
    unit = 1 if (len(train_dataset) // batch_size) < 1 else len(train_dataset) // batch_size
    step = unit * decay_step
    lr_sch = lr_scheduler.StepLR(trainer, step, gamma=decay_lr, last_epoch=-1)

    # torch split이 numpy, mxnet split과 달라서 아래와 같은 작업을 하는 것
    if batch_size % subdivision == 0:
        chunk = int(batch_size) // int(subdivision)
    else:
        logging.info(f"batch_size / subdivision 이 나누어 떨어지지 않습니다.")
        logging.info(f"subdivision 을 다시 설정하고 학습 진행하세요.")
        exit(0)

    start_time = time.time()
    for i in tqdm(range(start_epoch + 1, epoch + 1, 1), initial=start_epoch + 1, total=epoch):

        loss_sum = 0
        net.train()
        time_stamp = time.time()

        # multiscale을 하게되면 여기서 train_dataloader을 다시 만드는 것이 좋겠군..
        for batch_count, (anchor, positive, negative, _, _, _) in enumerate(
                train_dataloader,
                start=1):

            trainer.zero_grad()

            anchor = anchor.to(context)
            positive = positive.to(context)
            negative = negative.to(context)

            '''
            이렇게 하는 이유?
            209 line에서 net = net.to(context)로 함
            gpu>=1 인 경우 net = DataParallel(net, device_ids=device, output_device=context, dim=0) 에서 
            output_device - gradient가 계산되는 곳을 context로 했기 때문에 아래의 target들도 context로 지정해줘야 함
            '''
            anchor_split = torch.split(anchor, chunk, dim=0)
            positive_split = torch.split(positive, chunk, dim=0)
            negative_split = torch.split(negative, chunk, dim=0)

            losses = []
            total_loss = 0.0

            for anchor_part, positive_part, negative_part in zip(
                    anchor_split,
                    positive_split,
                    negative_split):

                anchor_pred = net(anchor_part)
                positive_pred = net(positive_part)
                negative_pred = net(negative_part)

                '''
                pytorch는 trainer.step()에서 batch_size 인자가 없다.
                Loss 구현시 고려해야 한다.(mean 모드) 
                '''
                ap_select = PDLoss(anchor_pred, positive_pred)
                an_select = PDLoss(anchor_pred, negative_pred)

                if semi_hard_negative:
                    # Semi-Hard Negative triplet selection
                    # (negative_distance - positive_distance < margin) AND (positive_distance < negative_distance)
                    # https://github.com/tamerthamoqa/facenet-pytorch-vggface2/blob/master/train_triplet_loss.py
                    first_condition = (an_select - ap_select) < margin
                    second_condition = ap_select < an_select
                    all = (torch.logical_and(first_condition, second_condition))
                    valid_triplets = torch.where(all == 1)
                else:
                    # Hard Negative triplet selection
                    # (negative_distance - positive_distance < margin)
                    # https://github.com/tamerthamoqa/facenet-pytorch-vggface2/blob/master/train_triplet_loss.py
                    all = (an_select - ap_select) < margin
                    valid_triplets = torch.where(all == 1)

                triplet_loss = TLLoss(anchor_pred[valid_triplets],
                                      positive_pred[valid_triplets],
                                      negative_pred[valid_triplets])
                loss = torch.div(triplet_loss, subdivision)
                losses.append(loss.item())
                total_loss = total_loss + loss

            if total_loss.isnan():
                logging.info("loss is nan")
                loss_sum += 0
                continue
            else:
                total_loss.backward()
                trainer.step()
                lr_sch.step()
                loss_sum += sum(losses)

            if batch_count % batch_log == 0:
                logging.info(f'[Epoch {i}][Batch {batch_count}/{train_update_number_per_epoch}]'
                             f'[Speed {(anchor.shape[0]*3) / (time.time() - time_stamp):.3f} samples/sec]'
                             f'[Lr = {lr_sch.get_last_lr()}]'
                             f'[loss = {sum(losses):.3f}]')
            time_stamp = time.time()

        train_loss_mean = np.divide(loss_sum, train_update_number_per_epoch)

        logging.info(
            f"train loss : {train_loss_mean}")

        if i % save_period == 0:

            if not os.path.exists(weight_path):
                os.makedirs(weight_path)

            module = net.module if isinstance(device, (list, tuple)) else net
            pretnet = PrePostNet(net=module)  # 새로운 객체가 생성

            try:
                torch.save({
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': trainer.state_dict()}, os.path.join(weight_path, f'{model}-{i:04d}.pt'))

                # torch.jit.trace() 보다는 control-flow 연산 적용이 가능한 torch.jit.script() 을 사용하자
                # torch.jit.script

                script = torch.jit.script(module)
                script.save(os.path.join(weight_path, f'{model}-{i:04d}.jit'))

                script = torch.jit.script(pretnet)
                script.save(os.path.join(weight_path, f'{model}-prepost-{i:04d}.jit'))

                # trace = torch.jit.trace(prepostnet, torch.rand(input_shape[0], input_shape[1], input_shape[2], input_shape[3], device=context))
                # trace.save(os.path.join(weight_path, f'{model}-{i:04d}.jit'))

            except Exception as E:
                logging.error(f"pt, jit export 예외 발생 : {E}")
            else:
                logging.info("pt, jit export 성공")

        if i % eval_period == 0 and valid_list:

            loss_sum = 0

            #net.eval()

            # loss 구하기
            for (anchor, positive, negative, _, _, _) in valid_dataloader:
                anchor = anchor.to(context)
                positive = positive.to(context)
                negative = negative.to(context)

                with torch.no_grad():
                    anchor_pred = net(anchor)
                    positive_pred = net(positive)
                    negative_pred = net(negative)

                    ap_select = PDLoss(anchor_pred, positive_pred)
                    an_select = PDLoss(anchor_pred, negative_pred)

                    if semi_hard_negative:
                        # Semi-Hard Negative triplet selection
                        # (negative_distance - positive_distance < margin) AND (positive_distance < negative_distance)
                        # https://github.com/tamerthamoqa/facenet-pytorch-vggface2/blob/master/train_triplet_loss.py
                        first_condition = (an_select - ap_select) < margin
                        second_condition = ap_select < an_select
                        all = (torch.logical_and(first_condition, second_condition))
                        valid_triplets = torch.where(all == 1)
                    else:
                        # Hard Negative triplet selection
                        # (negative_distance - positive_distance < margin)
                        # https://github.com/tamerthamoqa/facenet-pytorch-vggface2/blob/master/train_triplet_loss.py
                        all = (an_select - ap_select) < margin
                        valid_triplets = torch.where(all == 1)

                    triplet_loss = TLLoss(anchor_pred[valid_triplets],
                                          positive_pred[valid_triplets],
                                          negative_pred[valid_triplets])
                    loss_sum += triplet_loss.item()

            valid_loss_mean = np.divide(loss_sum, valid_update_number_per_epoch)
            logging.info(
                f"valid loss : {valid_loss_mean}")

            if tensorboard:

                batch_image = []
                dataloader_iter = iter(valid_dataloader)
                anchor, positive, negative, anchor_path, positive_path, negative_path = next(dataloader_iter)
                anchor = anchor.to(context)
                positive = positive.to(context)
                negative = negative.to(context)

                with torch.no_grad():

                    anchor_pred = net(anchor)
                    positive_pred = net(positive)
                    negative_pred = net(negative)

                    distance_of_ap_pred = torch.nn.functional.pairwise_distance(anchor_pred, positive_pred, p=2.0)
                    distance_of_an_pred = torch.nn.functional.pairwise_distance(anchor_pred, negative_pred, p=2.0)

                    for anc, pos, neg, anc_path, pos_path, neg_path, distance_of_ap, distance_of_an in zip(anchor_pred, positive_pred, negative_pred,
                                                                                                           anchor_path, positive_path, negative_path,
                                                                                                           distance_of_ap_pred, distance_of_an_pred):

                        anchor_img = cv2.imread(anc_path, flags=-1)
                        anchor_img = cv2.resize(anchor_img, dsize=(input_size[1], input_size[0]), interpolation=1)
                        anchor_img = cv2.cvtColor(anchor_img, cv2.COLOR_BGR2RGB)

                        positive_img = cv2.imread(pos_path, flags=-1)
                        positive_img = cv2.resize(positive_img, dsize=(input_size[1], input_size[0]), interpolation=1)
                        positive_img = cv2.cvtColor(positive_img, cv2.COLOR_BGR2RGB)

                        negative_img = cv2.imread(neg_path, flags=-1)
                        negative_img = cv2.resize(negative_img, dsize=(input_size[1], input_size[0]), interpolation=1)
                        negative_img = cv2.cvtColor(negative_img, cv2.COLOR_BGR2RGB)

                        distance_of_ap = distance_of_ap.item()
                        distance_of_an = distance_of_an.item()

                        if distance_of_ap < threshold:
                            ap_color = (0, 255, 0)
                        else:
                            ap_color = (255, 0, 0)

                        if distance_of_an < threshold:
                            an_color = (255, 0, 0)
                        else:
                            an_color = (0, 255, 0)

                        ap_rect = cv2.rectangle(np.ones_like(anchor_img), (0, 0), (input_size[1], input_size[0]), ap_color, thickness=-1)
                        an_rect = cv2.rectangle(np.ones_like(anchor_img), (0, 0), (input_size[1], input_size[0]), an_color, thickness=-1)
                        ap_rect = cv2.putText(ap_rect, "Dist : " + str(round(distance_of_ap, 3)), (input_size[1]//8, input_size[0]//2), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
                        an_rect = cv2.putText(an_rect, "Dist : " + str(round(distance_of_an, 3)), (input_size[1]//8, input_size[0]//2), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)

                        hconcat_images = np.concatenate([anchor_img, positive_img, ap_rect,
                                                         anchor_img, negative_img, an_rect], axis=1)

                        # Tensorboard에 그리기 위해 (height, width, channel) -> (channel, height, width) 를한다.
                        hconcat_images = np.transpose(hconcat_images, axes=(2, 0, 1))
                        batch_image.append(hconcat_images)  # (batch, channel, height, width)

                    img_grid = torchvision.utils.make_grid(torch.as_tensor(batch_image), nrow=1)
                    summary.add_image(tag="valid_result", img_tensor=img_grid, global_step=i)

                    summary.add_scalar(tag="loss/train_loss_mean",
                                       scalar_value=train_loss_mean,
                                       global_step=i)
                    summary.add_scalar(tag="loss/valid_loss_mean",
                                       scalar_value=valid_loss_mean,
                                       global_step=i)

                    for name, param in net.named_parameters():
                        summary.add_histogram(tag=name, values=param, global_step=i)

    end_time = time.time()
    learning_time = end_time - start_time
    logging.info(f"learning time : 약, {learning_time / 3600:0.2f}H")
    logging.info("optimization completed")

    if using_mlflow:
        ml.log_metric("learning time", round(learning_time / 3600, 2))


if __name__ == "__main__":
    run(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        threshold=1.0,
        epoch=100,
        input_size=[512, 512],
        batch_size=16,
        batch_log=100,
        subdivision=4,
        train_dataset_path="Dataset/train",
        valid_dataset_path="Dataset/valid",
        data_augmentation=True,
        num_workers=4,
        optimizer="ADAM",
        save_period=5,
        load_period=10,
        margin=0.2,
        semi_hard_negative=True,
        learning_rate=0.001, decay_lr=0.999, decay_step=10,
        weight_decay=0.000001,
        GPU_COUNT=0,
        base=18,
        pretrained_base=True,
        valid_size=8,
        eval_period=5,
        tensorboard=True,
        using_mlflow=True)
