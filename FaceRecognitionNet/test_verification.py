import logging
import os
import platform

import cv2
import torch
from tqdm import tqdm

from core import testdataloader

logfilepath = ""  # 따로 지정하지 않으면 terminal에 뜸
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)


# nograd, model.eval() 하기
def run(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        threshold = 1.0,
        load_name="250_250_ADAM_PRES18", load_period=10, GPU_COUNT=0,
        test_weight_path="weights",
        test_dataset_path="Dataset/test",
        num_workers=4,
        test_save_path="result",
        show_flag=True,
        save_flag=True):

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


    for anchor, positive, negative, anchor_path, positive_path, negative_path in tqdm(test_dataloader):

        _, _, height, width = anchor.shape

        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        with torch.no_grad():

            anchor_pred = net(anchor)
            positive_pred = net(positive)
            negative_pred = net(negative)

            distance_of_ap = 0
            distance_of_an = 0

            # l2 distance
            anchor = cv2.imread(anchor_path, flags=-1)
            anchor = cv2.resize(anchor, dsize=(width, height), interpolation=1)

            positive = cv2.imread(positive_path, flags=-1)
            positive = cv2.resize(positive, dsize=(width, height), interpolation=1)

            negative = cv2.imread(negative_path, flags=-1)
            negative = cv2.resize(negative, dsize=(width, height), interpolation=1)

            if distance_of_ap < threshold:
                ap_color = (0, 255, 0)
            else:
                ap_color = (0, 0, 255)

            if distance_of_an < threshold:
                an_color = (0, 0, 255)
            else:
                an_color = (0, 255, 0)

            ap_catimage = cv2.hconcat([anchor, positive])
            an_catimage = cv2.hconcat([anchor, negative])

            ap_catimage = cv2.rectangle(ap_catimage, (0, 0), (width*2, height), ap_color, thickness=3)
            an_catimage = cv2.rectangle(an_catimage, (0, 0), (width*2, height), an_color, thickness=3)

            if show_flag:
                catimage = cv2.vconcat([ap_catimage, an_catimage])
                cv2.imshow("result", catimage)
                cv2.waitKey(0)

            if save_flag:
                ps_path, ps_name = os.path.split(positive_path)
                ps_folder = os.path.split(ps_path)[-1]

                ng_path, ng_name = os.path.split(negative_path)
                ng_folder = os.path.split(ng_path)[-1]

                if not os.path.exists(test_save_path):
                    os.makedirs(test_save_path)
                cv2.imwrite(os.path.join(test_save_path, ps_folder, ps_name), ap_catimage)
                cv2.imwrite(os.path.join(test_save_path, ng_folder, ng_name), an_catimage)

if __name__ == "__main__":
    run(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        threshold=1.0,
        load_name="250_250_ADAM_PCENTER_RES18", load_period=10, GPU_COUNT=0,
        test_weight_path="weights",
        test_dataset_path="Dataset/test",
        test_save_path="result",
        num_workers=4,
        show_flag=True,
        save_flag=True)
