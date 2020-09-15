import logging
import os
import platform

import cv2
import numpy as np
import torch
from tqdm import tqdm

from core import Voc_2007_AP
from core import Yolov3Loss, TargetGenerator, Prediction
from core import box_resize
from core import plot_bbox
from core import testdataloader

logfilepath = ""  # 따로 지정하지 않으면 terminal에 뜸
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)

def run(input_frame_number=2,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        load_name="608_608_ADAM_PDark_53", load_period=10, GPU_COUNT=0,
        test_weight_path="weights",
        test_dataset_path="Dataset/test",
        test_save_path="result",
        test_graph_path="test_Graph",
        test_html_auto_open=False,
        num_workers=4,
        npy_flag = True,
        npy_save_path="npy_result",
        show_flag=True,
        save_flag=True,
        video_flag=True,
        video_min = None,
        video_max = None,
        video_fps = 15,
        video_name = "result",
        ignore_threshold=0.5,
        dynamic=False,
        multiperclass=True,
        nms_thresh=0.5,
        nms_topk=500,
        iou_thresh=0.5,
        except_class_thresh=0.05,
        plot_class_thresh=0.5):
    if GPU_COUNT <= 0:
        device = torch.device("cpu")
    elif GPU_COUNT > 0:
        device = torch.device("cuda")

    # 운영체제 확인
    if platform.system() == "Linux":
        logging.info(f"{platform.system()} OS")
    elif platform.system() == "Windows":
        logging.info(f"{platform.system()} OS")
    else:
        logging.info(f"{platform.system()} OS")

    if GPU_COUNT > 0:
        total_memory = torch.cuda.get_device_properties(device).total_memory
        free_memory = total_memory - torch.cuda.max_memory_allocated(device)
        free_memory = round(free_memory / (1024 ** 3), 2)
        total_memory = round(total_memory / (1024 ** 3), 2)
        logging.info(f'{torch.cuda.get_device_name(device)}')
        logging.info(f'Running on {device} / free memory : {free_memory}GB / total memory {total_memory}GB')
    else:
        logging.info(f'Running on {device}')


    logging.info(f"test {load_name}")
    netheight = int(load_name.split("_")[0])
    netwidth = int(load_name.split("_")[1])
    if not isinstance(netheight, int) and not isinstance(netwidth, int):
        logging.info("height is not int")
        logging.info("width is not int")
        raise ValueError
    else:
        logging.info(f"network input size : {(netheight, netwidth)}")

    try:
        test_dataloader, test_dataset = testdataloader(path=test_dataset_path,
                                                       input_size=(netheight, netwidth),
                                                       input_frame_number=input_frame_number,
                                                       num_workers=num_workers,
                                                       mean=mean, std=std)
    except Exception:
        logging.info("The dataset does not exist")
        exit(0)

    weight_path = os.path.join(test_weight_path, load_name)
    trace_path = os.path.join(weight_path, f'{load_name}-{load_period:04d}.jit')

    test_update_number_per_epoch = len(test_dataloader)
    if test_update_number_per_epoch < 1:
        logging.warning(" test batch size가 데이터 수보다 큼 ")
        exit(0)

    num_classes = test_dataset.num_class  # 클래스 수
    name_classes = test_dataset.classes
    logging.info("jit model test")

    try:
        net = torch.jit.load(trace_path, map_location=device)
        net.eval()
    except Exception:
        # DEBUG, INFO, WARNING, ERROR, CRITICAL 의 5가지 등급
        logging.info("loading jit 실패")
        exit(0)
    else:
        logging.info("loading jit 성공")


    targetgenerator = TargetGenerator(ignore_threshold=ignore_threshold, dynamic=dynamic, from_sigmoid=False)
    loss = Yolov3Loss(sparse_label=True,
                      from_sigmoid=False,
                      batch_axis=None,
                      num_classes=num_classes,
                      reduction="sum")

    prediction = Prediction(
        from_sigmoid=False,
        num_classes=num_classes,
        nms_thresh=nms_thresh,
        nms_topk=nms_topk,
        except_class_thresh=except_class_thresh,
        multiperclass=multiperclass)

    precision_recall = Voc_2007_AP(iou_thresh=iou_thresh, class_names=name_classes)

    ground_truth_colors = {}
    for i in range(num_classes):
        ground_truth_colors[i] = (0, 1, 0)

    object_loss_sum = 0
    xcyc_loss_sum = 0
    wh_loss_sum = 0
    class_loss_sum = 0

    if video_flag:
        if not os.path.exists(test_save_path):
            os.makedirs(test_save_path)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        dataloader_iter = iter(test_dataloader)
        _, _, _, origin_image, _ = next(dataloader_iter)
        _, height, width, _ = origin_image.shape
        out = cv2.VideoWriter(os.path.join(test_save_path, f'{video_name}_{video_fps}fps.mp4'), fourcc, video_fps, (width*input_frame_number, height))
        if isinstance(video_min, str):
            video_min = 0 if video_min.upper() == "NONE" else video_min
        if isinstance(video_max, str):
            video_max = test_update_number_per_epoch if video_max.upper() == "NONE" else video_max

    for i, (image, label, name, origin_img, origin_box) in tqdm(enumerate(test_dataloader), total=test_update_number_per_epoch):

        _, height, width, _ = origin_img.shape
        logging.info(f"real input size : {(height, width)}")

        image = image.to(device)
        label = label.to(device)
        gt_boxes = label[:, :, :4]
        gt_ids = label[:, :, 4:5]
        
        with torch.no_grad():
            output1, output2, output3, \
            anchor1, anchor2, anchor3, \
            offset1, offset2, offset3, \
            stride1, stride2, stride3 = net(image)

            ids, scores, bboxes = prediction(output1, output2, output3, anchor1, anchor2, anchor3, offset1, offset2,
                                             offset3, stride1, stride2, stride3)

        precision_recall.update(pred_bboxes=bboxes,
                                pred_labels=ids,
                                pred_scores=scores,
                                gt_boxes=gt_boxes,
                                gt_labels=gt_ids)

        bboxes = box_resize(bboxes[0].detach().cpu().numpy().copy(), (netwidth, netheight), (width, height))

        # 출력

        split_path = name[0].split("/")

        if npy_flag:

            npy_path = os.path.join(npy_save_path, split_path[-3], split_path[-2])
            if not os.path.exists(npy_path):
                os.makedirs(npy_path)

            ids_npy = ids.detach().cpu().numpy().copy()[0]  # (22743, 1)
            except_ids_index = np.where(ids_npy != -1)
            scores_npy = scores.detach().cpu().numpy().copy()[0][except_ids_index]
            scores_npy = np.expand_dims(scores_npy, axis=-1)

            xmin, ymin, xmax, ymax = np.split(bboxes, 4, axis=-1)

            xmin = xmin[except_ids_index]
            ymin = ymin[except_ids_index]
            xmax = xmax[except_ids_index]
            ymax = ymax[except_ids_index]
            bboxes_npy = np.stack([xmin, ymin, xmax, ymax], axis=-1)

            # 순서 scores, bboxes - 마지막 축으로 concat
            scorebox = np.concatenate([scores_npy, bboxes_npy], axis=-1)
            result_name = split_path[-1].replace(".jpg", ".npy")
            np.save(os.path.join(npy_path, result_name), scorebox)

        for pair_ig in origin_img:
            split_ig = torch.split(pair_ig, 3, dim=-1)

            hconcat_image_list = []
            for j, ig in enumerate(split_ig):
                if j == len(split_ig) - 1:  # 마지막 이미지
                    ground_truth = plot_bbox(ig, origin_box[0][:, :4], scores=None, labels=origin_box[0][:, 4:5], thresh=None,
                                             reverse_rgb=True,
                                             class_names=test_dataset.classes, absolute_coordinates=True,
                                             colors=ground_truth_colors)
                    prediction_box = plot_bbox(ground_truth, bboxes, scores=scores[0], labels=ids[0],
                                               thresh=plot_class_thresh,
                                               reverse_rgb=False,
                                               class_names=test_dataset.classes, absolute_coordinates=True)
                    hconcat_image_list.append(prediction_box)
                else:
                    ig = ig.type(torch.uint8)
                    ig = ig.detach().cpu().numpy().copy()
                    ig = cv2.cvtColor(ig, cv2.COLOR_RGB2BGR)
                    hconcat_image_list.append(ig)

            hconcat_images = np.concatenate(hconcat_image_list, axis=1)

        if save_flag:
            image_save_path = os.path.join(test_save_path, split_path[-3], split_path[-2])
            if not os.path.exists(image_save_path):
                os.makedirs(image_save_path)
            cv2.imwrite(os.path.join(image_save_path, split_path[-1]), hconcat_images)
        if show_flag:
            logging.info(f"image name : {os.path.splitext(os.path.basename(name[0]))[0]}")
            cv2.imshow("temp", hconcat_images)
            cv2.waitKey(0)
        if video_flag:
            video_min = 0 if video_min < 0 else video_min
            video_max = test_update_number_per_epoch if video_max > test_update_number_per_epoch else video_max
            if i >= video_min and i <= video_max:
                out.write(hconcat_images)

        xcyc_target, wh_target, objectness, class_target, weights = targetgenerator([output1, output2, output3],
                                                                                    [anchor1, anchor2, anchor3],
                                                                                    gt_boxes,
                                                                                    gt_ids, (netheight, netwidth))

        xcyc_loss, wh_loss, object_loss, class_loss = loss(output1, output2, output3, xcyc_target, wh_target,
                                                           objectness,
                                                           class_target, weights)

        xcyc_loss_sum += xcyc_loss.item()
        wh_loss_sum += wh_loss.item()
        object_loss_sum += object_loss.item()
        class_loss_sum += class_loss.item()

    train_xcyc_loss_mean = np.divide(xcyc_loss_sum, test_update_number_per_epoch)
    train_wh_loss_mean = np.divide(wh_loss_sum, test_update_number_per_epoch)
    train_object_loss_mean = np.divide(object_loss_sum, test_update_number_per_epoch)
    train_class_loss_mean = np.divide(class_loss_sum, test_update_number_per_epoch)
    train_total_loss_mean = train_xcyc_loss_mean + train_wh_loss_mean + train_object_loss_mean + train_class_loss_mean

    logging.info(
        f"train xcyc loss : {train_xcyc_loss_mean} / "
        f"train wh loss : {train_wh_loss_mean} / "
        f"train object loss : {train_object_loss_mean} / "
        f"train class loss : {train_class_loss_mean} / "
        f"train total loss : {train_total_loss_mean}"
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
                                  AP=AP_appender, mAP=mAP_result, folder_name=test_graph_path, epoch=load_period,
                                  auto_open=test_html_auto_open)

if __name__ == "__main__":
    run(input_frame_number=2,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        load_name="608_608_ADAM_PDark_53", load_period=100, GPU_COUNT=0,
        test_weight_path="weights",
        test_dataset_path="Dataset/test",
        test_save_path="result",
        npy_save_path = "npy_result",
        test_graph_path="test_Graph",
        test_html_auto_open=True,
        num_workers=4,
        npy_flag = True,
        show_flag=True,
        save_flag=True,
        video_flag=True,
        video_min = None,
        video_max = None,
        video_fps = 15,
        video_name = "result",
        ignore_threshold=0.5,
        dynamic=False,
        multiperclass=True,
        nms_thresh=0.5,
        nms_topk=500,
        iou_thresh=0.5,
        except_class_thresh=0.05,
        plot_class_thresh=0.5)  #
