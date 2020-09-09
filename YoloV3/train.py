import glob
import logging
import os
import platform
import time

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

from core import TargetGenerator
from core import Voc_2007_AP
from core import Yolov3, Yolov3Loss, Prediction
from core import plot_bbox, PrePostNet
from core import traindataloader, validdataloader

logfilepath = ""
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)


def run(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        offset_alloc_size=(64, 64),
        anchors={"shallow": [(10, 13), (16, 30), (33, 23)],
                 "middle": [(30, 61), (62, 45), (59, 119)],
                 "deep": [(116, 90), (156, 198), (373, 326)]},
        epoch=100,
        input_size=[416, 416],
        input_frame_number=2,
        batch_log=100,
        batch_size=16,
        batch_interval=10,
        subdivision=4,
        train_dataset_path="Dataset/train",
        valid_dataset_path="Dataset/valid",
        multiscale=False,
        factor_scale=[13, 5],
        ignore_threshold=0.5,
        dynamic=False,
        data_augmentation=True,
        num_workers=4,
        optimizer="ADAM",
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
        multiperclass=True,
        nms_thresh=0.5,
        nms_topk=500,
        iou_thresh=0.5,
        except_class_thresh=0.05,
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

    # 입력 사이즈를 32의 배수로 지정해 버리기 - stride가 일그러지는 것을 막기 위함
    if input_size[0] % 32 != 0 and input_size[1] % 32 != 0:
        logging.info("The input size must be a multiple of 32")
        exit(0)

    if GPU_COUNT > 0 and batch_size < GPU_COUNT:
        logging.info("batch size must be greater than gpu number")
        exit(0)

    if multiscale:
        logging.info("Using MultiScale")

    if data_augmentation:
        logging.info("Using Data Augmentation")

    logging.info("training YoloV3 Detector")
    input_shape = (1, 3*input_frame_number) + tuple(input_size)

    train_dataloader, train_dataset = traindataloader(multiscale=multiscale,
                                                      factor_scale=factor_scale,
                                                      augmentation=data_augmentation,
                                                      path=train_dataset_path,
                                                      input_size=input_size,
                                                      input_frame_number=input_frame_number,
                                                      batch_size=batch_size,
                                                      pin_memory=True,
                                                      batch_interval=batch_interval,
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
                                                          input_frame_number=input_frame_number,
                                                          batch_size=valid_size,
                                                          num_workers=num_workers,
                                                          pin_memory=True,
                                                          shuffle=True, mean=mean, std=std)
        valid_update_number_per_epoch = len(valid_dataloader)
        if valid_update_number_per_epoch < 1:
            logging.warning("valid batch size가 데이터 수보다 큼")
            exit(0)

    num_classes = train_dataset.num_class  # 클래스 수
    name_classes = train_dataset.classes

    optimizer = optimizer.upper()
    if pretrained_base:
        model = str(input_size[0]) + "_" + str(input_size[1]) + "_" + optimizer + "_P" + "Res_" + str(base)+f"_{input_frame_number}frame"
    else:
        model = str(input_size[0]) + "_" + str(input_size[1]) + "_" + optimizer + "_Res_" + str(base)+f"_{input_frame_number}frame"

    # https://discuss.pytorch.org/t/how-to-save-the-optimizer-setting-in-a-log-in-pytorch/17187
    weight_path = os.path.join("weights", f"{model}")
    param_path = os.path.join(weight_path, f'{model}-{load_period:04d}.pt')

    start_epoch = 0
    net = Yolov3(base=base,
                 input_size=input_size,
                 anchors=anchors,
                 num_classes=num_classes,  # foreground만
                 pretrained=pretrained_base,
                 alloc_size=offset_alloc_size)

    # https://github.com/sksq96/pytorch-summary / because of anchor, not working
    try:
        if GPU_COUNT == 0:
            modelsummary(net.to(context), input_shape[1:], device="cpu")
        elif GPU_COUNT > 0:
            modelsummary(net.to(context), input_shape[1:], device="cuda")
    except Exception:
        logging.info("torchsummary 문제로 인해 summary 불가")

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

    targetgenerator = TargetGenerator(ignore_threshold=ignore_threshold, dynamic=dynamic, from_sigmoid=False)

    loss = Yolov3Loss(sparse_label=True,
                      from_sigmoid=False,
                      batch_axis=None,
                      num_classes=num_classes,
                      reduction="sum")

    prediction = Prediction(
        unique_ids=name_classes,
        from_sigmoid=False,
        num_classes=num_classes,
        nms_thresh=nms_thresh,
        nms_topk=nms_topk,
        except_class_thresh=except_class_thresh,
        multiperclass=multiperclass)

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

        xcyc_loss_sum = 0
        wh_loss_sum = 0
        object_loss_sum = 0
        class_loss_sum = 0
        time_stamp = time.time()

        for batch_count, (image, label, _) in enumerate(
                train_dataloader, start=1):

            _, _, height, width = image.shape

            trainer.zero_grad()
            image = image.to(context)
            label = label.to(context)
            '''
            이렇게 하는 이유?
            209 line에서 net = net.to(context)로 함
            gpu>=1 인 경우 net = DataParallel(net, device_ids=device, output_device=context, dim=0) 에서 
            output_device - gradient가 계산되는 곳을 context로 했기 때문에 아래의 target들도 context로 지정해줘야 함
            '''

            image_split = torch.split(image, chunk, dim=0)
            gt_boxes = torch.split(label[:, :, :4], chunk, dim=0)
            gt_ids = torch.split(label[:, :, 4:5], chunk, dim=0)

            xcyc_losses = []
            wh_losses = []
            object_losses = []
            class_losses = []
            total_loss = []

            for image_part, gt_boxes_part, gt_ids_part in zip(image_split, gt_boxes, gt_ids):

                output1, output2, output3, anchor1, anchor2, anchor3, offset1, offset2, offset3, stride1, stride2, stride3 = net(image_part)
                xcyc_target, wh_target, objectness, class_target, weights = targetgenerator(
                    [output1, output2, output3],
                    [anchor1, anchor2, anchor3],
                    gt_boxes_part,
                    gt_ids_part, (height, width))

                xcyc_loss, wh_loss, object_loss, class_loss = loss(output1, output2, output3, xcyc_target,
                                                                   wh_target, objectness, class_target, weights)

                xcyc_loss = torch.div(xcyc_loss, subdivision)
                wh_loss = torch.div(wh_loss, subdivision)
                object_loss = torch.div(object_loss, subdivision)
                class_loss = torch.div(class_loss, subdivision)

                xcyc_losses.append(xcyc_loss.item())
                wh_losses.append(wh_loss.item())
                object_losses.append(object_loss.item())
                class_losses.append(class_loss.item())

                total_loss.append(xcyc_loss + wh_loss + object_loss + class_loss)

            autograd.backward(total_loss)
            trainer.step()
            lr_sch.step()

            xcyc_loss_sum += sum(xcyc_losses)
            wh_loss_sum += sum(wh_losses)
            object_loss_sum += sum(object_losses)
            class_loss_sum += sum(class_losses)

            if batch_count % batch_log == 0:
                logging.info(f'[Epoch {i}][Batch {batch_count}/{train_update_number_per_epoch}],'
                             f'[Speed {image.shape[0] / (time.time() - time_stamp):.3f} samples/sec],'
                             f'[Lr = {lr_sch.get_last_lr()}]'
                             f'[xcyc loss = {sum(xcyc_losses):.3f}]'
                             f'[wh loss = {sum(wh_losses):.3f}]'
                             f'[obj loss = {sum(object_losses):.3f}]'
                             f'[class loss = {sum(class_losses):.3f}]')
            time_stamp = time.time()

        train_xcyc_loss_mean = np.divide(xcyc_loss_sum, train_update_number_per_epoch)
        train_wh_loss_mean = np.divide(wh_loss_sum, train_update_number_per_epoch)
        train_object_loss_mean = np.divide(object_loss_sum, train_update_number_per_epoch)
        train_class_loss_mean = np.divide(class_loss_sum, train_update_number_per_epoch)
        train_total_loss_mean = train_xcyc_loss_mean + train_wh_loss_mean + train_object_loss_mean + train_class_loss_mean
        logging.info(
            f"train xcyc loss : {train_xcyc_loss_mean} / "
            f"train wh loss : {train_wh_loss_mean} / "
            f"train object loss : {train_object_loss_mean} / "
            f"train class loss : {train_class_loss_mean} / "
            f"train total loss : {train_total_loss_mean}"
        )

        if i % save_period == 0:

            if not os.path.exists(weight_path):
                os.makedirs(weight_path)

            module = net.module if isinstance(device, (list, tuple)) else net
            auxnet = Prediction(
                from_sigmoid=False,
                num_classes=num_classes,
                nms_thresh=nms_thresh,
                nms_topk=nms_topk,
                except_class_thresh=except_class_thresh,
                multiperclass=multiperclass)
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

            xcyc_loss_sum = 0
            wh_loss_sum = 0
            object_loss_sum = 0
            class_loss_sum = 0

            # loss 구하기
            for image, label, _ in valid_dataloader:

                _, _, height, width = image.shape

                image = image.to(context)
                label = label.to(context)
                gt_box = label[:, :, :4]
                gt_id = label[:, :, 4:5]

                output1, output2, output3, anchor1, anchor2, anchor3, offset1, offset2, offset3, stride1, stride2, stride3 = net(
                    image)
                xcyc_target, wh_target, objectness, class_target, weights = targetgenerator(
                    [output1, output2, output3],
                    [anchor1, anchor2, anchor3],
                    gt_box,
                    gt_id, (height, width))

                id, score, bbox = prediction(output1, output2, output3, anchor1, anchor2, anchor3, offset1, offset2,
                                             offset3, stride1, stride2, stride3)

                precision_recall.update(pred_bboxes=bbox,
                                        pred_labels=id,
                                        pred_scores=score,
                                        gt_boxes=gt_box,
                                        gt_labels=gt_id)

                xcyc_loss, wh_loss, object_loss, class_loss = loss(output1, output2, output3, xcyc_target,
                                                                   wh_target, objectness,
                                                                   class_target, weights)

                xcyc_loss_sum += xcyc_loss.item()
                wh_loss_sum += wh_loss.item()
                object_loss_sum += object_loss.item()
                class_loss_sum += class_loss.item()

            valid_xcyc_loss_mean = np.divide(xcyc_loss_sum, valid_update_number_per_epoch)
            valid_wh_loss_mean = np.divide(wh_loss_sum, valid_update_number_per_epoch)
            valid_object_loss_mean = np.divide(object_loss_sum, valid_update_number_per_epoch)
            valid_class_loss_mean = np.divide(class_loss_sum, valid_update_number_per_epoch)
            valid_total_loss_mean = valid_xcyc_loss_mean + valid_wh_loss_mean + valid_object_loss_mean + valid_class_loss_mean

            logging.info(
                f"valid xcyc loss : {valid_xcyc_loss_mean} / "
                f"valid wh loss : {valid_wh_loss_mean} / "
                f"valid object loss : {valid_object_loss_mean} / "
                f"valid class loss : {valid_class_loss_mean} / "
                f"valid total loss : {valid_total_loss_mean}"
            )

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
                image, label, _ = next(dataloader_iter)

                image = image.to(context)
                label = label.to(context)
                gt_boxes = label[:, :, :4]
                gt_ids = label[:, :, 4:5]

                output1, output2, output3, anchor1, anchor2, anchor3, offset1, offset2, offset3, stride1, stride2, stride3 = net(
                    image)
                ids, scores, bboxes = prediction(output1, output2, output3, anchor1, anchor2, anchor3, offset1,
                                                 offset2, offset3, stride1, stride2, stride3)

                for img, gt_id, gt_box, id, score, bbox in zip(image, gt_ids, gt_boxes, ids, scores, bboxes):
                    split_img = torch.split(img, 3, dim=0)  # numpy split과 다르네...
                    hconcat_image_list = []

                    for j, ig in enumerate(split_img):

                        ig = ig.permute((1, 2, 0)) * torch.tensor(std, device=ig.device) + torch.tensor(mean, device=ig.device)
                        ig = (ig * 255).clamp(0, 255)
                        ig = ig.to(torch.uint8)
                        ig = ig.detach().cpu().numpy().copy()

                        if j == len(split_img)-1: # 마지막 이미지

                            # ground truth box 그리기
                            ground_truth = plot_bbox(ig, gt_box, scores=None, labels=gt_id, thresh=None,
                                                     reverse_rgb=False,
                                                     class_names=valid_dataset.classes,
                                                     absolute_coordinates=True,
                                                     colors=ground_truth_colors)
                            # # prediction box 그리기
                            prediction_box = plot_bbox(ground_truth, bbox, scores=score, labels=id,
                                                       thresh=plot_class_thresh,
                                                       reverse_rgb=False,
                                                       class_names=valid_dataset.classes,
                                                       absolute_coordinates=True,)
                            hconcat_image_list.append(prediction_box)
                        else:
                            hconcat_image_list.append(ig)

                        hconcat_images = np.concatenate(hconcat_image_list, axis=1)

                        # Tensorboard에 그리기 위해 (height, width, channel) -> (channel, height, width) 를한다.
                        hconcat_images = np.transpose(hconcat_images, axes=(2, 0, 1))
                        batch_image.append(hconcat_images)  # (batch, channel, height, width)

                img_grid = torchvision.utils.make_grid(torch.as_tensor(batch_image), nrow=1)

                summary.add_image(tag="valid_result", img_tensor=img_grid, global_step=i)

                summary.add_scalar(tag="xy_loss/train_xcyc_loss",
                                   scalar_value=train_xcyc_loss_mean,
                                   global_step=i)
                summary.add_scalar(tag="xy_loss/valid_xcyc_loss",
                                   scalar_value=valid_xcyc_loss_mean,
                                   global_step=i)

                summary.add_scalar(tag="wh_loss/train_wh_loss",
                                   scalar_value=train_wh_loss_mean,
                                   global_step=i)
                summary.add_scalar(tag="wh_loss/valid_wh_loss",
                                   scalar_value=valid_wh_loss_mean,
                                   global_step=i)

                summary.add_scalar(tag="object_loss/train_object_loss",
                                   scalar_value=train_object_loss_mean,
                                   global_step=i)
                summary.add_scalar(tag="object_loss/valid_object_loss",
                                   scalar_value=valid_object_loss_mean,
                                   global_step=i)

                summary.add_scalar(tag="class_loss/train_class_loss",
                                   scalar_value=train_class_loss_mean,
                                   global_step=i)
                summary.add_scalar(tag="class_loss/valid_class_loss",
                                   scalar_value=valid_class_loss_mean,
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
        offset_alloc_size=(64, 64),
        anchors={"shallow": [(10, 13), (16, 30), (33, 23)],
                 "middle": [(30, 61), (62, 45), (59, 119)],
                 "deep": [(116, 90), (156, 198), (373, 326)]},
        epoch=10,
        input_size=[416, 416],
        input_frame_number=2,
        batch_log=100,
        batch_size=16,
        batch_interval=10,
        subdivision=4,
        train_dataset_path="Dataset/train",
        valid_dataset_path="Dataset/valid",
        multiscale=False,
        factor_scale=[13, 5],
        ignore_threshold=0.5,
        dynamic=False,
        data_augmentation=True,
        num_workers=4,
        optimizer="ADAM",
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
        multiperclass=True,
        nms_thresh=0.5,
        nms_topk=500,
        iou_thresh=0.5,
        except_class_thresh=0.05,
        plot_class_thresh=0.5)
