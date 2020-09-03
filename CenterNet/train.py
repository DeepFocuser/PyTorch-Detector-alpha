import glob
import logging
import os
import platform
import time
from collections import OrderedDict

import cv2
import mlflow as ml
import numpy as np
import torch
import torch.autograd as autograd
import torchvision
from torch.nn import DataParallel
from torch.optim import Adam, RMSprop, SGD, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary as modelsummary
from tqdm import tqdm

from core import CenterNet
from core import HeatmapFocalLoss, NormedL1Loss
from core import Prediction
from core import Voc_2007_AP
from core import plot_bbox, PrePostNet
from core import traindataloader, validdataloader

logfilepath = ""
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)


# 초기화 참고하기
# https://pytorch.org/docs/stable/nn.init.html?highlight=nn%20init#torch.nn.init.kaiming_normal_

def run(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        epoch=100,
        input_size=[512, 512],
        input_frame_number=2,
        batch_size=16,
        batch_log=100,
        subdivision=4,
        train_dataset_path="Dataset/train",
        valid_dataset_path="Dataset/valid",
        data_augmentation=True,
        num_workers=4,
        optimizer="ADAM",
        lambda_off=1,
        lambda_size=0.1,
        save_period=5,
        load_period=10,
        learning_rate=0.001, decay_lr=0.999, decay_step=10,
        GPU_COUNT=0,
        base=18,
        pretrained_base=True,
        valid_size=8,
        eval_period=5,
        tensorboard=True,
        valid_graph_path="valid_Graph",
        valid_html_auto_open=True,
        using_mlflow=True,
        topk=100,
        iou_thresh=0.5,
        nms=False,
        except_class_thresh=0.01,
        nms_thresh=0.5,
        plot_class_thresh=0.5):
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

    logging.info("training Center Detector")
    input_shape = (1, 3 * input_frame_number) + tuple(input_size)

    scale_factor = 4  # 고정
    logging.info(f"scale factor {scale_factor}")

    train_dataloader, train_dataset = traindataloader(augmentation=data_augmentation,
                                                      path=train_dataset_path,
                                                      input_size=input_size,
                                                      input_frame_number=input_frame_number,
                                                      batch_size=batch_size,
                                                      pin_memory=True,
                                                      num_workers=num_workers,
                                                      shuffle=True, mean=mean, std=std, scale_factor=scale_factor,
                                                      make_target=True)

    train_update_number_per_epoch = len(train_dataloader)
    if train_update_number_per_epoch < 1:
        logging.warning("train batch size가 데이터 수보다 큼")
        exit(0)

    valid_list = glob.glob(os.path.join(valid_dataset_path, "*"))
    if valid_list:
        valid_dataloader, valid_dataset = validdataloader(path=valid_dataset_path,
                                                          input_size=input_size,
                                                          input_frame_number=input_frame_number,
                                                          batch_size=valid_size,
                                                          num_workers=num_workers,
                                                          pin_memory=True,
                                                          shuffle=True, mean=mean, std=std, scale_factor=scale_factor,
                                                          make_target=True)
        valid_update_number_per_epoch = len(valid_dataloader)
        if valid_update_number_per_epoch < 1:
            logging.warning("valid batch size가 데이터 수보다 큼")
            exit(0)

    num_classes = train_dataset.num_class  # 클래스 수
    name_classes = train_dataset.classes

    optimizer = optimizer.upper()
    if pretrained_base:
        model = str(input_size[0]) + "_" + str(input_size[1]) + "_" + optimizer + "_P" + "CENTER_RES" + str(base)
    else:
        model = str(input_size[0]) + "_" + str(input_size[1]) + "_" + optimizer + "_CENTER_RES" + str(base)

    # https://discuss.pytorch.org/t/how-to-save-the-optimizer-setting-in-a-log-in-pytorch/17187
    weight_path = os.path.join("weights", f"{model}")
    param_path = os.path.join(weight_path, f'{model}-{load_period:04d}.pt')

    start_epoch = 0
    net = CenterNet(base=base,
                    input_frame_number=input_frame_number,
                    heads=OrderedDict([
                        ('heatmap', {'num_output': num_classes, 'bias': -2.19}),
                        ('offset', {'num_output': 2}),
                        ('wh', {'num_output': 2})
                    ]),
                    head_conv_channel=64,
                    pretrained=pretrained_base)

    # https://github.com/sksq96/pytorch-summary
    modelsummary(net.to(context), input_shape[1:])

    if tensorboard:
        summary = SummaryWriter(log_dir=os.path.join("torchboard", model), max_queue=10, flush_secs=10)
        summary.add_graph(net.to(context), input_to_model=torch.ones(input_shape, device=context), verbose=False)

    if os.path.exists(param_path):
        start_epoch = load_period
        checkpoint = torch.load(param_path)
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
        trainer = Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.000001)
    elif optimizer.upper() == "RMSPROP":
        trainer = RMSprop(net.parameters(), lr=learning_rate, alpha=0.99, weight_decay=0.000001, momentum=0)
    elif optimizer.upper() == "SGD":
        trainer = SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.000001)
    else:
        logging.error("optimizer not selected")
        exit(0)

    if os.path.exists(param_path):
        # optimizer weight 불러오기
        checkpoint = torch.load(param_path)
        if 'optimizer_state_dict' in checkpoint:
            try:
                trainer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as E:
                logging.info(E)
            else:
                logging.info(f"loading optimizer_state_dict")

    if isinstance(device, (list, tuple)):
        net = DataParallel(net, device_ids=device, output_device=context, dim=0)

    # optimizer
    # https://pytorch.org/docs/master/optim.html?highlight=lr%20sche#torch.optim.lr_scheduler.CosineAnnealingLR
    unit = 1 if (len(train_dataset) // batch_size) < 1 else len(train_dataset) // batch_size
    step = unit * decay_step
    lr_sch = lr_scheduler.StepLR(trainer, step, gamma=decay_lr, last_epoch=-1)

    heatmapfocalloss = HeatmapFocalLoss(from_sigmoid=True, alpha=2, beta=4)
    normedl1loss = NormedL1Loss()
    prediction = Prediction(unique_ids=name_classes, topk=topk, scale=scale_factor, nms=nms,
                            except_class_thresh=except_class_thresh, nms_thresh=nms_thresh)
    precision_recall = Voc_2007_AP(iou_thresh=iou_thresh, class_names=name_classes)

    # torch split이 numpy, mxnet split과 달라서 아래와 같은 작업을 하는 것
    if batch_size % subdivision == 0:
        chunk = int(batch_size) // int(subdivision)
    else:
        logging.info(f"batch_size / subdivision 이 나누어 떨어지지 않습니다.")
        logging.info(f"subdivision 을 다시 설정하고 학습 진행하세요.")
        exit(0)


    start_time = time.time()
    for i in tqdm(range(start_epoch + 1, epoch + 1, 1), initial=start_epoch + 1, total=epoch):

        heatmap_loss_sum = 0
        offset_loss_sum = 0
        wh_loss_sum = 0
        time_stamp = time.time()

        # multiscale을 하게되면 여기서 train_dataloader을 다시 만드는 것이 좋겠군..
        for batch_count, (image, _, heatmap_target, offset_target, wh_target, mask_target, _) in enumerate(
                train_dataloader,
                start=1):

            trainer.zero_grad()

            image = image.to(context)

            '''
            이렇게 하는 이유?
            209 line에서 net = net.to(context)로 함
            gpu>=1 인 경우 net = DataParallel(net, device_ids=device, output_device=context, dim=0) 에서 
            output_device - gradient가 계산되는 곳을 context로 했기 때문에 아래의 target들도 context로 지정해줘야 함
            '''
            heatmap_target = heatmap_target.to(context)
            offset_target = offset_target.to(context)
            wh_target = wh_target.to(context)
            mask_target = mask_target.to(context)

            image_split = torch.split(image, chunk, dim=0)
            heatmap_target_split = torch.split(heatmap_target, chunk, dim=0)
            offset_target_split = torch.split(offset_target, chunk, dim=0)
            wh_target_split = torch.split(wh_target, chunk, dim=0)
            mask_target_split = torch.split(mask_target, chunk, dim=0)

            heatmap_losses = []
            offset_losses = []
            wh_losses = []
            total_loss = []

            for image_part, heatmap_target_part, offset_target_part, wh_target_part, mask_target_part in zip(
                    image_split,
                    heatmap_target_split,
                    offset_target_split,
                    wh_target_split,
                    mask_target_split):
                heatmap_pred, offset_pred, wh_pred = net(image_part)
                '''
                pytorch는 trainer.step()에서 batch_size 인자가 없다.
                Loss 구현시 고려해야 한다.(mean 모드) 
                '''
                heatmap_loss = torch.div(heatmapfocalloss(heatmap_pred, heatmap_target_part), subdivision)
                offset_loss = torch.div(normedl1loss(offset_pred, offset_target_part, mask_target_part) * lambda_off,
                                        subdivision)
                wh_loss = torch.div(normedl1loss(wh_pred, wh_target_part, mask_target_part) * lambda_size, subdivision)

                heatmap_losses.append(heatmap_loss.item())
                offset_losses.append(offset_loss.item())
                wh_losses.append(wh_loss.item())

                total_loss.append(heatmap_loss + offset_loss + wh_loss)

            # batch size만큼 나눠줘야 하지 않나?
            autograd.backward(total_loss)

            trainer.step()
            lr_sch.step()

            heatmap_loss_sum += sum(heatmap_losses)
            offset_loss_sum += sum(offset_losses)
            wh_loss_sum += sum(wh_losses)

            if batch_count % batch_log == 0:
                logging.info(f'[Epoch {i}][Batch {batch_count}/{train_update_number_per_epoch}],'
                             f'[Speed {image.shape[0] / (time.time() - time_stamp):.3f} samples/sec],'
                             f'[Lr = {lr_sch.get_last_lr()}]'
                             f'[heatmap loss = {sum(heatmap_losses):.3f}]'
                             f'[offset loss = {sum(offset_losses):.3f}]'
                             f'[wh loss = {sum(wh_losses):.3f}]')
            time_stamp = time.time()

        train_heatmap_loss_mean = np.divide(heatmap_loss_sum, train_update_number_per_epoch)
        train_offset_loss_mean = np.divide(offset_loss_sum, train_update_number_per_epoch)
        train_wh_loss_mean = np.divide(wh_loss_sum, train_update_number_per_epoch)
        train_total_loss_mean = train_heatmap_loss_mean + train_offset_loss_mean + train_wh_loss_mean

        logging.info(
            f"train heatmap loss : {train_heatmap_loss_mean} / train offset loss : {train_offset_loss_mean} / train wh loss : {train_wh_loss_mean} / train total loss : {train_total_loss_mean}")

        if i % save_period == 0:

            if not os.path.exists(weight_path):
                os.makedirs(weight_path)

            module = net.module if isinstance(device, (list, tuple)) else net
            auxnet = Prediction(unique_ids=name_classes, topk=topk, scale=scale_factor, nms=nms, except_class_thresh=except_class_thresh,
                                nms_thresh=nms_thresh)
            prepostnet = PrePostNet(net=module, auxnet=auxnet, input_frame_number=input_frame_number)  # 새로운 객체가 생성

            try:
                torch.save({
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': trainer.state_dict()}, os.path.join(weight_path, f'{model}-{i:04d}.pt'))

                # torch.jit.trace() 보다는 control-flow 연산 적용이 가능한 torch.jit.script() 을 사용하자
                # torch.jit.script
                script = torch.jit.script(module)
                script.save(os.path.join(weight_path, f'{model}-{i:04d}.jit'))

                script = torch.jit.script(prepostnet)
                script.save(os.path.join(weight_path, f'{model}-prepost-{i:04d}.jit'))

                # # torch.jit.trace - 안 써짐
                # 오류 : Expected object of device type cuda but got device type cpu for argument #2 'other' in call to _th_fmod
                # trace = torch.jit.trace(prepostnet, torch.rand(input_shape[0], input_shape[1], input_shape[2], input_shape[3], device=context))
                # trace.save(os.path.join(weight_path, f'{model}-{i:04d}.jit'))

            except Exception as E:
                logging.error(f"pt, jit export 예외 발생 : {E}")
            else:
                logging.info("pt, jit export 성공")

        if i % eval_period == 0 and valid_list:

            heatmap_loss_sum = 0
            offset_loss_sum = 0
            wh_loss_sum = 0

            # loss 구하기
            for image, label, heatmap_target, offset_target, wh_target, mask_target, _ in valid_dataloader:
                image = image.to(context)
                label = label.to(context)
                gt_box = label[:, :, :4]
                gt_id = label[:, :, 4:5]

                heatmap_target = heatmap_target.to(context)
                offset_target = offset_target.to(context)
                wh_target = wh_target.to(context)
                mask_target = mask_target.to(context)

                heatmap_pred, offset_pred, wh_pred = net(image)
                id, score, bbox = prediction(heatmap_pred, offset_pred, wh_pred)
                precision_recall.update(pred_bboxes=bbox,
                                        pred_labels=id,
                                        pred_scores=score,
                                        gt_boxes=gt_box * scale_factor,
                                        gt_labels=gt_id)

                heatmap_loss = heatmapfocalloss(heatmap_pred, heatmap_target)
                offset_loss = normedl1loss(offset_pred, offset_target, mask_target) * lambda_off
                wh_loss = normedl1loss(wh_pred, wh_target, mask_target) * lambda_size

                heatmap_loss_sum += heatmap_loss.item()
                offset_loss_sum += offset_loss.item()
                wh_loss_sum += wh_loss.item()

            valid_heatmap_loss_mean = np.divide(heatmap_loss_sum, valid_update_number_per_epoch)
            valid_offset_loss_mean = np.divide(offset_loss_sum, valid_update_number_per_epoch)
            valid_wh_loss_mean = np.divide(wh_loss_sum, valid_update_number_per_epoch)
            valid_total_loss_mean = valid_heatmap_loss_mean + valid_offset_loss_mean + valid_wh_loss_mean

            logging.info(
                f"valid heatmap loss : {valid_heatmap_loss_mean} / valid offset loss : {valid_offset_loss_mean} / valid wh loss : {valid_wh_loss_mean} / valid total loss : {valid_total_loss_mean}")

            AP_appender = []
            round_position = 2
            class_name, precision, recall, true_positive, false_positive, threshold = precision_recall.get_PR_list()
            for j, c, p, r in zip(range(len(recall)), class_name, precision, recall):
                name, AP = precision_recall.get_AP(c, p, r)
                logging.info(f"class {j}'s {name} AP : {round(AP * 100, round_position)}%")
                AP_appender.append(AP)
            mAP_result = np.mean(AP_appender)

            logging.info(f"mAP : {round(mAP_result * 100, round_position)}%")
            precision_recall.get_PR_curve(name=class_name,
                                          precision=precision,
                                          recall=recall,
                                          threshold=threshold,
                                          AP=AP_appender, mAP=mAP_result, folder_name=valid_graph_path, epoch=i,
                                          auto_open=valid_html_auto_open)
            precision_recall.reset()

            if tensorboard:

                batch_image = []
                ground_truth_colors = {}
                for k in range(num_classes):
                    ground_truth_colors[k] = (0, 1, 0) # RGB

                dataloader_iter = iter(valid_dataloader)
                image, label, _, _, _, _, _ = next(dataloader_iter)

                image = image.to(context)
                label = label.to(context)
                gt_boxes = label[:, :, :4]
                gt_ids = label[:, :, 4:5]

                heatmap_pred, offset_pred, wh_pred = net(image)
                ids, scores, bboxes = prediction(heatmap_pred, offset_pred, wh_pred)

                for img, gt_id, gt_box, heatmap, id, score, bbox in zip(image, gt_ids, gt_boxes, heatmap_pred, ids,
                                                                        scores, bboxes):

                    split_img = torch.split(img, 3, dim=0) # numpy split과 다르네...
                    hconcat_image_list = []

                    for j, ig in enumerate(split_img):

                        ig = ig.permute((1, 2, 0)) * torch.tensor(std, device=ig.device) + torch.tensor(mean, device=ig.device)
                        ig = (ig * 255).clamp(0, 255)
                        ig = ig.to(torch.uint8)
                        ig = ig.detach().cpu().numpy().copy()

                        if j == len(split_img) - 1:  # 마지막 이미지
                            # heatmap 그리기
                            heatmap = heatmap.detach().cpu().numpy().copy()
                            heatmap = np.multiply(heatmap, 255.0)  # 0 ~ 255 범위로 바꾸기
                            heatmap = np.amax(heatmap, axis=0, keepdims=True)  # channel 축으로 가장 큰것 뽑기
                            heatmap = np.transpose(heatmap, axes=(1, 2, 0))  # (height, width, channel=1)
                            heatmap = np.repeat(heatmap, 3, axis=-1)
                            heatmap = heatmap.astype("uint8")  # float32 -> uint8
                            heatmap = cv2.resize(heatmap, dsize=(input_size[1], input_size[0]))  # 사이즈 원복
                            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

                            # ground truth box 그리기
                            ground_truth = plot_bbox(ig, gt_box * scale_factor, scores=None, labels=gt_id, thresh=None,
                                                     reverse_rgb=False,
                                                     class_names=valid_dataset.classes,
                                                     absolute_coordinates=True,
                                                     colors=ground_truth_colors)
                            # # prediction box 그리기
                            prediction_box = plot_bbox(ground_truth, bbox, scores=score, labels=id,
                                                       thresh=plot_class_thresh,
                                                       reverse_rgb=False,
                                                       class_names=valid_dataset.classes,
                                                       absolute_coordinates=True,
                                                       heatmap=heatmap)
                            hconcat_image_list.append(prediction_box)
                        else:
                            hconcat_image_list.append(ig)

                    hconcat_images = np.concatenate(hconcat_image_list, axis=1)

                    # Tensorboard에 그리기 위해 (height, width, channel) -> (channel, height, width) 를한다.
                    hconcat_images = np.transpose(hconcat_images, axes=(2, 0, 1))
                    batch_image.append(hconcat_images)  # (batch, channel, height, width)

                img_grid = torchvision.utils.make_grid(torch.as_tensor(batch_image), nrow=1)
                summary.add_image(tag="valid_result", img_tensor=img_grid, global_step=i)

                summary.add_scalar(tag="heatmap_loss/train_heatmap_loss_mean",
                                   scalar_value=train_heatmap_loss_mean,
                                   global_step=i)
                summary.add_scalar(tag="heatmap_loss/valid_heatmap_loss_mean",
                                   scalar_value=valid_heatmap_loss_mean,
                                   global_step=i)

                summary.add_scalar(tag="offset_loss/train_offset_loss_mean",
                                   scalar_value=train_offset_loss_mean,
                                   global_step=i)
                summary.add_scalar(tag="offset_loss/valid_offset_loss_mean",
                                   scalar_value=valid_offset_loss_mean,
                                   global_step=i)

                summary.add_scalar(tag="wh_loss/train_wh_loss_mean",
                                   scalar_value=train_wh_loss_mean,
                                   global_step=i)
                summary.add_scalar(tag="wh_loss/valid_wh_loss_mean",
                                   scalar_value=valid_wh_loss_mean,
                                   global_step=i)

                summary.add_scalar(tag="total_loss/train_total_loss",
                                   scalar_value = train_total_loss_mean,
                                   global_step=i)
                summary.add_scalar(tag="total_loss/valid_total_loss",
                                   scalar_value = valid_total_loss_mean,
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
        epoch=100,
        input_size=[512, 512],
        input_frame_number=2,
        batch_size=16,
        batch_log=100,
        subdivision=4,
        train_dataset_path="Dataset/train",
        valid_dataset_path="Dataset/valid",
        data_augmentation=True,
        num_workers=4,
        optimizer="ADAM",
        lambda_off=1,
        lambda_size=0.1,
        save_period=5,
        load_period=10,
        learning_rate=0.001, decay_lr=0.999, decay_step=10,
        GPU_COUNT=0,
        base=18,
        pretrained_base=True,
        valid_size=8,
        eval_period=5,
        tensorboard=True,
        valid_graph_path="valid_Graph",
        valid_html_auto_open=True,
        using_mlflow=True,
        topk=100,
        iou_thresh=0.5,
        nms=False,
        except_class_thresh=0.01,
        nms_thresh=0.5,
        plot_class_thresh=0.5)
