import logging
import os
import platform

import cv2
import numpy as np
import torch

logfilepath = ""  # 따로 지정하지 않으면 terminal에 뜸
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)


# nograd, model.eval() 하기
def run(load_name="512_512_ADAM_PRES18", load_period=10,
        input_frame_number = 1,
        test_weight_path="weights",
        test_save_path = "result",
        save_flag=True,
        video_fps=30,
        video_name="result"):

    # 운영체제 확인
    if platform.system() == "Linux":
        logging.info(f"{platform.system()} OS")
    elif platform.system() == "Windows":
        logging.info(f"{platform.system()} OS")
    else:
        logging.info(f"{platform.system()} OS")

    if torch.cuda.device_count() > 0:
        device = torch.device("cuda")
        total_memory = torch.cuda.get_device_properties(device).total_memory
        free_memory = total_memory - torch.cuda.max_memory_allocated(device)
        free_memory = round(free_memory / (1024 ** 3), 2)
        total_memory = round(total_memory / (1024 ** 3), 2)
        logging.info(f'{torch.cuda.get_device_name(device)}')
        logging.info(f'Running on {device} / free memory : {free_memory}GB / total memory {total_memory}GB')
    else:
        device = torch.device("cpu")
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

    weight_path = os.path.join(test_weight_path, load_name)
    trace_path = os.path.join(weight_path, f'{load_name}-prepost-{load_period:04d}.jit')
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

    if save_flag and not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(os.path.join(test_save_path, f'{video_name}_{video_fps}fps.mp4'), fourcc, video_fps, (netwidth*2, netheight))
    cap = cv2.VideoCapture(0)  # 0: default webcam

    if input_frame_number == 1:
        while True:
            success, img = cap.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (netwidth, netheight), interpolation=1)

            torch_img = torch.as_tensor(img[None,:,:,:], device=device)
            with torch.no_grad():
                output = net(torch_img)
            output = output.squeeze()
            foreground = output[0] # foreground
            #background = output[1] # background

            foreground = torch.where(foreground >= 0.5, 255, 0).to(torch.uint8)
            foreground = foreground.detach().cpu().numpy().copy()
            foreground = np.repeat(foreground[:, :, None], repeats=3, axis=-1)

            # background = torch.where(background >= 0.5, 255, 0).to(torch.uint8)
            # background = background.detach().cpu().numpy().copy()
            # background = np.repeat(background[:, :, None], repeats=3, axis=-1)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            result = np.concatenate([img, foreground], axis=1)

            if save_flag:
                out.write(result)

            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    elif input_frame_number == 2:

        flag = True
        before = None

        while True:
            success, img = cap.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (netwidth, netheight), interpolation=1)

            if flag:
                before = img
                flag = False
                continue

            concat_img = np.concatenate([before, img], axis=-1)
            torch_img = torch.as_tensor(concat_img[None,:,:,:], device=device)
            with torch.no_grad():
                output = net(torch_img)

            output = output.squeeze()
            foreground = output[0] # foreground
            #background = output[1] # background

            foreground = torch.where(foreground >= 0.5, 255, 0).to(torch.uint8)
            foreground = foreground.detach().cpu().numpy().copy()
            foreground = np.repeat(foreground[:, :, None], repeats=3, axis=-1)

            # background = torch.where(background >= 0.5, 255, 0).to(torch.uint8)
            # background = background.detach().cpu().numpy().copy()
            # background = np.repeat(background[:, :, None], repeats=3, axis=-1)

            before = img
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            result = np.concatenate([img, foreground], axis=1)
            if save_flag:
                out.write(result)

            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run(load_name="512_512_ADAM_PRES18_DataAug_False_2frame", load_period=1,
        input_frame_number=2,
        test_weight_path="weights",
        test_save_path="result",
        save_flag=False,
        video_fps = 30,
        video_name = "result")
