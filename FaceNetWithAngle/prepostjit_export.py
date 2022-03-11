import logging
import os

import torch

from core import PrePostNet
from core import testdataloader

logfilepath = ""
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)


def export(originpath="weights",
           newpath="jitweights",
           load_name="250_250_ADAM_RES18",
           load_period=1):

    if torch.cuda.device_count() > 0:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    origin_weight_path = os.path.join(originpath, load_name)
    jit_path = os.path.join(origin_weight_path, f'{load_name}-{load_period:04d}.jit')

    new_weight_path = os.path.join(newpath, load_name)

    if not os.path.exists(new_weight_path):
        os.makedirs(new_weight_path)

    if os.path.exists(jit_path):
        logging.info(f"loading {os.path.basename(jit_path)}")
        net = torch.jit.load(jit_path, map_location=device)
    else:
        raise FileExistsError

    _, test_dataset = testdataloader()

    prepostnet = PrePostNet(net=net)  # 새로운 객체가 생성

    try:
        script = torch.jit.script(prepostnet)
        script.save(os.path.join(new_weight_path, f'{load_name}-prepost-{load_period:04d}.jit'))

    except Exception as E:
        logging.error(f"jit export 예외 발생 : {E}")
    else:
        logging.info("jit export 성공")


if __name__ == "__main__":
    export(originpath="weights",
           newpath="jitweights",
           load_name="608_608_ADAM_PCENTER_RES18",
           load_period=1)
