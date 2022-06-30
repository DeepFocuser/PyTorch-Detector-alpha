import glob
import logging
import os
import platform
import time

import mlflow as ml
import numpy as np
import torch
import torchvision
from torch.nn import DataParallel
from torch.optim import Adam, RMSprop, SGD, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary as modelsummary
from tqdm import tqdm

from core import SegmentationNet
from core import PrePostNet
from core import SoftmaxCrossEntropyLoss
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
        save_period=5,
        load_period=10,
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

    logging.info("training segmentation")
    input_shape = (1, 3 * input_frame_number) + tuple(input_size)

    train_dataloader, train_dataset = traindataloader(augmentation=data_augmentation,
                                                      path=train_dataset_path,
                                                      input_size=input_size,
                                                      input_frame_number=input_frame_number,
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
                                                          input_frame_number=input_frame_number,
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

    model = model+"_DataAug_" + str(data_augmentation) + f"_{input_frame_number}frame"

    # https://discuss.pytorch.org/t/how-to-save-the-optimizer-setting-in-a-log-in-pytorch/17187
    weight_path = os.path.join("weights", f"{model}")
    param_path = os.path.join(weight_path, f'{model}-{load_period:04d}.pt')

    start_epoch = 0
    net = SegmentationNet(base=base, input_frame_number=input_frame_number, pretrained=pretrained_base)

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

        '''
            n 채널 대비
            가중치 채널축으로 복제
        '''
        temp = checkpoint['model_state_dict']['_base_network._resnet.conv1.weight']
        _, channel, _, _ = temp.shape

        if input_frame_number*3 != channel:
            repeated = torch.repeat_interleave(checkpoint['model_state_dict']['_base_network._resnet.conv1.weight'], input_frame_number, dim=1)
            checkpoint['model_state_dict']['_base_network._resnet.conv1.weight'] = repeated

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

    SCELoss = SoftmaxCrossEntropyLoss(axis=1, from_logits=False)

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
        for batch_count, (image, mask, _) in enumerate(
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
            mask = mask.to(context)
            image_split = torch.split(image, chunk, dim=0)
            mask_split = torch.split(mask, chunk, dim=0)

            losses = []
            total_loss = 0.0

            for image_part, mask_part in zip(
                    image_split,
                    mask_split):
                pred = net(image_part)
                '''
                pytorch는 trainer.step()에서 batch_size 인자가 없다.
                Loss 구현시 고려해야 한다.(mean 모드) 
                '''
                loss = torch.div(SCELoss(pred, mask_part), subdivision)
                losses.append(loss.item())

                total_loss = total_loss + loss

            total_loss.backward()
            trainer.step()
            lr_sch.step()

            loss_sum += sum(losses)

            if batch_count % batch_log == 0:
                logging.info(f'[Epoch {i}][Batch {batch_count}/{train_update_number_per_epoch}]'
                             f'[Speed {image.shape[0] / (time.time() - time_stamp):.3f} samples/sec]'
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
            pretnet = PrePostNet(net=module, input_frame_number=input_frame_number)  # 새로운 객체가 생성

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

            except Exception as E:
                logging.error(f"pt, jit export 예외 발생 : {E}")
            else:
                logging.info("pt, jit export 성공")

        if i % eval_period == 0 and valid_list:

            loss_sum = 0
            net.eval()
            # loss 구하기
            for image, mask, _ in valid_dataloader:
                image = image.to(context)
                mask = mask.to(context)

                with torch.no_grad():
                    pred = net(image)
                    output_loss = SCELoss(pred, mask)
                    loss_sum += output_loss.item()

            valid_loss_mean = np.divide(loss_sum, valid_update_number_per_epoch)

            # confusion matrix
            logging.info(
                f"valid loss : {valid_loss_mean}")

            if tensorboard:

                batch_image = []
                dataloader_iter = iter(valid_dataloader)
                image, mask, _ = next(dataloader_iter)
                image = image.to(context)
                mask = mask.to(context)

                with torch.no_grad():

                    pred = net(image)
                    pred = torch.softmax(pred, dim=1) # fore_ground

                for img, pd, mk in zip(image, pred, mask):

                    pd = pd[0] # forground
                    mk = mk[0] # forground
                    split_img = torch.split(img, 3, dim=0)  # numpy split과 다르네...

                    # (H,W,3)-ndarray로 만들기
                    mk = torch.where(mk >= 0.5, 255, 0).to(torch.uint8) # 바꿔 야함
                    mk = mk.detach().cpu().numpy().copy()
                    mk = np.repeat(mk[:, :, None], repeats=3, axis=-1)

                    pd = torch.where(pd >= 0.5, 255, 0).to(torch.uint8)
                    pd = pd.detach().cpu().numpy().copy()
                    pd = np.repeat(pd[:,:,None], repeats=3, axis=-1)

                    hconcat_image_list = []
                    for j, ig in enumerate(split_img):

                        ig = ig.permute((1, 2, 0)) * torch.tensor(std, device=ig.device) + torch.tensor(mean,
                                                                                                        device=ig.device)
                        ig = (ig * 255).clamp(0, 255)
                        ig = ig.to(torch.uint8)
                        ig = ig.detach().cpu().numpy().copy()
                        hconcat_image_list.append(ig)

                    hconcat_image_list.extend([mk, pd])
                    hconcat_images = np.concatenate(hconcat_image_list, axis=1)

                    # Tensorboard에 그리기 위해 (height, width, channel) -> (channel, height, width) 를한다.
                    hconcat_images = np.transpose(hconcat_images, axes=(2, 0, 1))
                    batch_image.append(hconcat_images)  # (batch, channel, height, width)

                img_grid = torchvision.utils.make_grid(torch.as_tensor(np.array(batch_image)), nrow=1)
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
        save_period=5,
        load_period=10,
        learning_rate=0.001, decay_lr=0.999, decay_step=10,
        weight_decay=0.000001,
        GPU_COUNT=0,
        base=18,
        pretrained_base=True,
        valid_size=8,
        eval_period=5,
        tensorboard=True,
        using_mlflow=True)
