import cv2
import logging
import numpy as np
import os
import platform
import torch
from tqdm import tqdm

from core import SoftmaxCrossEntropyLoss
from core import plot_bbox
from core import testdataloader

logfilepath = ""  # 따로 지정하지 않으면 terminal에 뜸
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)


# nograd, model.eval() 하기
def run(input_frame_number=1,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        load_name="480_640_ADAM_PRES18", load_period=10, GPU_COUNT=0,
        test_weight_path="weights",
        test_dataset_path="Dataset/test",
        test_save_path="result",
        num_workers=4,
        show_flag=True,
        save_flag=True,
        video_flag=True,
        video_min=None,
        video_max=None,
        video_fps=15,
        video_name="result"):
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
                                                       input_frame_number= input_frame_number,
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

    SCELoss = SoftmaxCrossEntropyLoss(axis=-1, sparse_label=False, from_logits=False)

    ground_truth_colors = {}
    for i in range(num_classes):
        ground_truth_colors[i] = (0, 1, 0)

    loss_sum = 0
    if video_flag:
        if not os.path.exists(test_save_path):
            os.makedirs(test_save_path)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        dataloader_iter = iter(test_dataloader)
        _, _, _, origin_image = next(dataloader_iter)
        _, height, width, _ = origin_image.shape
        out = cv2.VideoWriter(os.path.join(test_save_path, f'{video_name}_{video_fps}fps.mp4'), fourcc, video_fps, (width*(input_frame_number+1), height))

        if isinstance(video_min, str):
            video_min = 0 if video_min.upper() == "NONE" else video_min
        if isinstance(video_max, str):
            video_max = test_update_number_per_epoch if video_max.upper() == "NONE" else video_max

    numerator = 0
    denominator = 0

    TP, FP = 0, 0 # True Positive, False Positive
    FN, TN = 0, 0 # False Negative, True Negative

    for image, label, name, origin_image in tqdm(test_dataloader):
        _, height, width, _ = origin_image.shape
        logging.info(f"real input size : {(height, width)}")

        image = image.to(device)
        label = label.to(device)

        with torch.no_grad():
            pred = net(image)
            pred_softmax = torch.softmax(pred, dim=-1)

        # accuracy
        pred_argmax = torch.argmax(pred_softmax, dim=1)  # (batch_size , num_outputs)
        pred_argmax = pred_argmax.detach().cpu().numpy().copy()
        label_argmax = torch.argmax(label, dim=1)  # (batch_size , num_outputs)
        label_argmax = label_argmax.detach().cpu().numpy().copy()

        # for confusion matrix
        if pred_argmax.ravel() == 1 and label_argmax.ravel() == 1: # ok pred / real ok
            TP+=1
        elif pred_argmax.ravel() == 1 and label_argmax.ravel() == 0: # ok pred / real ng
            FP+=1
        elif pred_argmax.ravel() == 0 and label_argmax.ravel() == 1: # ng pred / real ok
            FN+=1
        elif pred_argmax.ravel() == 0 and label_argmax.ravel()==0: # ng pred / real ng
            TN+=1

        numerator += sum(pred_argmax == label_argmax)
        denominator += image.shape[0]

        for pair_ig in origin_image:
            split_ig = torch.split(pair_ig, 3, dim=-1)

            hconcat_image_list = []
            for j, ig in enumerate(split_ig):
                if j == len(split_ig) - 1:  # 마지막 이미지
                    # ground truth box 그리기
                    index = torch.argmax(label[0])
                    ground_truth = plot_bbox(ig, score=None, label=name_classes[index],
                                             reverse_rgb=True,
                                             class_names=name_classes,
                                             colors=ground_truth_colors, gt = True)
                    index = torch.argmax(pred_softmax[0])
                    score = torch.max(pred_softmax[0])
                    prediction_box = plot_bbox(ground_truth, score=score, label=name_classes[index],
                                               reverse_rgb=False,
                                               class_names=name_classes)
                    hconcat_image_list.append(prediction_box)
                else:
                    ig = ig.type(torch.uint8)
                    ig = ig.detach().cpu().numpy().copy()
                    ig = cv2.cvtColor(ig, cv2.COLOR_RGB2BGR)
                    hconcat_image_list.append(ig)

            hconcat_images = np.concatenate(hconcat_image_list, axis=1)

        if save_flag:
            if not os.path.exists(test_save_path):
                os.makedirs(test_save_path)
            cv2.imwrite(os.path.join(test_save_path, os.path.basename(name[0])), hconcat_images)
        if show_flag:
            logging.info(f"image name : {os.path.splitext(os.path.basename(name[0]))[0]}")
            cv2.imshow("temp", hconcat_images)
            cv2.waitKey(0)
        if video_flag:
            video_min = 0 if video_min < 0 else video_min
            video_max = test_update_number_per_epoch if video_max > test_update_number_per_epoch else video_max
            if i >= video_min and i <= video_max:
                out.write(hconcat_images)

        loss = SCELoss(pred, label)
        loss_sum += loss.item()

    # epoch 당 평균 loss
    test_loss_mean = np.divide(loss_sum, test_update_number_per_epoch)

    # confusion matrix
    matrix = f"\n" \
             f"           Pos(Real) Neg(Real)  \n" \
             f"Pos(Pred) | TP({TP})  |  FP({FP}) |\n" \
             f"Neg(Pred) | FN({FN})  |  TN({TN}) |\n"
    accuracy = round((numerator / denominator) * 100, 2)
    precision = round((TP / (TP+FP))*100, 2)
    recall = round((TP / (TP+FN))*100, 2)
    f1score = round(((2*precision*recall) / (precision+recall)), 2)
    specificity = round((TN / (FP+TN))*100, 2)

    logging.info(
        f"\nmatrix : {matrix}"
        f"accuracy : {accuracy}%\n"
        f"precision : {precision}%\n"
        f"recall : {recall}%\n"
        f"f1score : {f1score}\n"
        f"specificity : {specificity}%\n"
        f"test loss : {test_loss_mean}")

    with open("evaluate.txt", mode='w+t') as f:
        f.write(f"matrix : {matrix}")
        f.write(f"accuracy : {accuracy}%\n")
        f.write(f"precision : {precision}%\n")
        f.write(f"recall : {recall}%\n")
        f.write(f"f1score : {f1score}\n")
        f.write(f"specificity : {specificity}%\n")

if __name__ == "__main__":
    run(input_frame_number=2,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        load_name="480_640_ADAM_PCENTER_RES18", load_period=10, GPU_COUNT=0,
        test_weight_path="weights",
        test_dataset_path="Dataset/test",
        test_save_path="result",
        num_workers=4,
        show_flag=True,
        video_flag=True,
        save_flag=True,
        video_min = None,
        video_max = None,
        video_fps = 15,
        video_name = "result")
