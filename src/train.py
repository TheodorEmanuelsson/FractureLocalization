import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import(
#    load_checkpoint,
#    save_checkpoint,
    get_loaders,
#    check_accuracy,
    save_validation_inference_imgs,
    save_predictions_as_csv,
)


# Hyperparameters etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train_img/"
VAL_IMG_DIR = "data/val_img/"
TRAINING_ANNOTATION = "data/training_annotation.json"
VALIDATION_ANNOTATION ="data/validation_annotation.json"

def train_fn(loader, model, optimizer, loss_fn, writer, tensorboard_step, scaler):
    
    loop = tqdm(loader)

    for data in loop:

        img = data['image'].to(DEVICE)
        target_width = data['resized_width'].float().to(DEVICE)
        target_height = data['resized_height'].float().to(DEVICE)
    
        # forward
        #with torch.cuda.amp.autocast():
        predictions = model(img)
        width_hat = predictions['label_width']
        height_hat = predictions['label_height']
        if torch.isnan(width_hat).any():
            print('NaN in width prediction')
        if torch.isnan(height_hat).any():
            print('NaN in height prediction')
        loss_width = loss_fn(width_hat, target_width.squeeze(axis = 1))#.squeeze().type(torch.LongTensor))
        loss_height = loss_fn(height_hat, target_height.squeeze(axis = 1))#.squeeze().type(torch.LongTensor))
        loss = loss_width + loss_height
        # backward
        optimizer.zero_grad() # Zero out gradients
        if MIXED_PRECISION == 0:
            loss.backward() # Compute derivatives # scaler.scale(loss).backward()
            optimizer.step() # Take a step # scaler.step(optimizer)
        elif MIXED_PRECISION == 1:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            raise ValueError('Invalid mix-precision argument')

        # Write losses to tensorboard
        writer.add_scalar('Training overall loss', loss, global_step = tensorboard_step)
        writer.add_scalar('Training width loss', loss_width, global_step = tensorboard_step)
        writer.add_scalar('Training height loss', loss_height, global_step = tensorboard_step)

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def val_fn(loader, model, loss_fn, writer, tensorboard_step, scheduler):

    model.eval()
    val_losses = []


    for idx, data in enumerate(loader):

        img = data['image'].to(DEVICE)
        target_width = data['resized_width'].float().to(DEVICE)
        target_height = data['resized_height'].float().to(DEVICE)

        with torch.no_grad():
            predictions = model(img)
            width_hat = predictions['label_width']
            height_hat = predictions['label_height']
            loss_width = loss_fn(width_hat, target_width.squeeze(axis = 1))#.squeeze().type(torch.LongTensor))
            loss_height = loss_fn(height_hat, target_height.squeeze(axis = 1))#.squeeze().type(torch.LongTensor))
            loss = loss_width + loss_height

        val_losses.append(loss)

        # Write losses to tensorboard
        writer.add_scalar('Validation overall loss', loss, global_step = tensorboard_step)
        writer.add_scalar('Validation width loss', loss_width, global_step = tensorboard_step)
        writer.add_scalar('Validation height loss', loss_height, global_step = tensorboard_step)

    mean_val_loss = sum(val_losses) / len(val_losses)
    scheduler.step(mean_val_loss)

def main():

    if ARCHITECTURE == 'ResNet50':
        from models.ResNetCustom import ResNet50
        model = ResNet50(num_classes = 1, channels=1, out_act = None).to(DEVICE)
    elif ARCHITECTURE == 'ResNet34':
        from models.ResNet34 import ResNet34, ResBlock
        model = ResNet34(in_channels=1, resblock = ResBlock, out_act = None).to(DEVICE)
    elif ARCHITECTURE == 'ResNet18':
        from models.ResNet18 import ResNet18, ResBlock
        model = ResNet18(in_channels=1, resblock = ResBlock, out_act = None).to(DEVICE)
    elif ARCHITECTURE == 'SimpleCNN':
        from models.CNN_model_big import Resize_CNN
        model = Resize_CNN(in_channels=1).to(DEVICE)
    elif ARCHITECTURE == 'ResNet101':
        from models.ResNetCustom import ResNet101
        model = ResNet101(num_classes = 1, channels=1, out_act = None).to(DEVICE)
    elif ARCHITECTURE == 'ResNet152':
        from models.ResNetCustom import ResNet152
        model = ResNet152(num_classes = 1, channels=1, out_act = None).to(DEVICE)
    else:
        raise ValueError('Please specify valid architecture')

    # Initialize loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

    # Define tenorboard writer
    writer = SummaryWriter(MODEL_PATH)

    # Get the data loaders
    train_loader, val_loader = get_loaders(
        train_dir= TRAIN_IMG_DIR,
        training_annotation = TRAINING_ANNOTATION,
        val_dir = VAL_IMG_DIR,
        validation_annotation = VALIDATION_ANNOTATION,
        batch_size = BATCH_SIZE,
        validation_size = VALIDATION_SIZE,
        num_workers = NUM_WORKERS,
        pin_memory = PIN_MEMORY,
        transform=AUGMENT
    )

    scaler = torch.cuda.amp.GradScaler()

    # Do the training iterations
    step = 0
    for epoch in range(NUM_EPOCHS):
        print(f'Epoch:{epoch}\n')
        train_fn(train_loader, model, optimizer, loss_fn, writer, step, scaler) 
        val_fn(val_loader, model, loss_fn, writer, step, scheduler)
        step += 1

    # Save the model to runs folder in base format
    torch.save(model, f'runs/training/{ARCHITECTURE}_{MODEL_NAME}_{NUM_EPOCHS}/{MODEL_NAME}_store.zip')

    # Print the last epoch results to csv    
    if VALIDATION_SIZE >= 1:
        # Save results to csv
        save_predictions_as_csv(val_loader, model, model_folder=MODEL_PATH, device=DEVICE)
        save_validation_inference_imgs(MODEL_PATH, RESIZE_SIZE)
        
if __name__ == '__main__':
    # Empty the cuda cache
    torch.cuda.empty_cache()
    #torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--resize', type=int)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val-size', type=int, default=5, help='Number of validation images to do inference on')
    parser.add_argument('--workers', type = int, default=4)
    parser.add_argument('--name', help='Name of training run')
    parser.add_argument('--architecture', default='ResNet50', help = 'Name of the used architecture. See folder models for more.')
    parser.add_argument('--mix-precision', type = int, default=0, help='If training should use mixed precision floats')
    parser.add_argument('--out-act', default=None, help='The activation on the output node: either linear or sigmoid')
    parser.add_argument('--augment', default=None, help = 'Use data augmentations')

    opt = parser.parse_args()

    # User arguments
    LEARNING_RATE = opt.lr
    BATCH_SIZE = opt.batch_size
    VALIDATION_SIZE = opt.val_size
    NUM_EPOCHS = opt.epochs
    NUM_WORKERS = opt.workers
    MODEL_NAME = str(opt.name)
    RESIZE_SIZE = opt.resize
    ARCHITECTURE = opt.architecture
    MIXED_PRECISION = opt.mix_precision
    ACTIVATION = opt.out_act if opt.out_act is None else str(opt.out_act)
    AUGMENT = opt.augment
    eps = 1e-10

    # Derived arguments 
    MODEL_PATH = f'runs/training/{ARCHITECTURE}_{MODEL_NAME}_{NUM_EPOCHS}'
    # Make run folder
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    main()