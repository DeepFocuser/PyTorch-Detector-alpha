import logging
import os
import platform

import cv2
import numpy as np
import onnx
import onnxruntime
import torch

from core import plot_bbox
from core import testdataloader

logfilepath = ""
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)


def export(image_path="sample.jpg",
           originpath="weights",
           newpath="onnxweights",
           load_name="512_512_ADAM_PRES18",
           load_period=70):
    # 1. 운영체제 확인
    if platform.system() == "Linux":
        logging.info(f"{platform.system()} OS")
    elif platform.system() == "Windows":
        logging.info(f"{platform.system()} OS")
    else:
        logging.info(f"{platform.system()} OS")

    if torch.cuda.device_count() > 0:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_info = load_name.split("_")
    height = int(train_info[0])
    width = int(train_info[0])

    _, test_dataset = testdataloader()
    name_classes = test_dataset.classes

    # 2. 이미지 로드 및 전처리
    image = cv2.imread(image_path, flags=-1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # https://deep-learning-study.tistory.com/185
    image = cv2.resize(image, (width, height), interpolation=1)  # 보간법 중요
    # image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4) # 보간법 중요

    torch_tensor = torch.as_tensor(image)
    torch_tensor = torch_tensor.unsqueeze(0)

    # 2. 모델 로드
    origin_weight_path = os.path.join(originpath, load_name)
    jit_path = os.path.join(origin_weight_path, f'{load_name}-prepost-{load_period:04d}.jit')
    new_weight_path = os.path.join(newpath, load_name)

    if not os.path.exists(new_weight_path):
        os.makedirs(new_weight_path)

    if os.path.exists(jit_path):
        logging.info(f"loading {os.path.basename(jit_path)}")
        # control flow 대비를 위해서 jit파일은 torch.jit.script로 저장이 된 상태여야한다.
        net = torch.jit.load(jit_path, map_location=device)
        net.eval()
    else:
        raise FileExistsError

    try:
        # onnx 경량화 필요시 : // https://onnxruntime.ai/docs/performance/quantization.html - 경량화 필요시
        onnx_path = os.path.join(new_weight_path, f"{load_name}-prepost-{load_period:04d}.onnx")
        # Export the model
        torch.onnx.export(net,  # model being run
                          torch_tensor,  # model input (or a tuple for multiple inputs)
                          onnx_path,  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=15,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                        'output': {0: 'batch_size'}})
    except Exception as E:
        logging.error(f"onnx export 예외 발생 : {E}")
    else:
        logging.info(f"onnx export to {onnx_path}")
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logging.info("onnx check 성공")

    try:
        ort_session = onnxruntime.InferenceSession(onnx_path)
        ort_inputs = {ort_session.get_inputs()[0].name: image[None, :, :, :]}
        pred = ort_session.run(None, ort_inputs)

    except Exception as E:
        logging.error(f"onnx Inference 예외 발생 : {E}")
    else:
        logging.info("onnx Inference 성공")

        pred = np.argmax(pred, axis=0).astype(np.uint8)*255
        pred = np.repeat(pred[:, :, None], repeats=3, axis=-1)
        plot_bbox(pred, image_show=True, image_save = True)

        if __name__ == "__main__":
            export(image_path="sample.jpg",
                   originpath = "weights",
                   newpath = "onnxweights",
                   load_name = "512_512_ADAM_PRES18",
                   load_period = 70)
