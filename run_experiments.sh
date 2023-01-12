#!bin/sh

# RUN COMMAND: nohup sh run_experiments.sh > ./runs/training/experiments0.out &

## Runs on the 8GB GPU

# Run 100 iterations of ResNet18
python3 ./src/train.py --epochs 100 --resize 512 --batch-size 32 --val-size 32 --workers 8 --name plat --lr 0.001 --architecture ResNet18 --augment True

# Run 100 iterations of ResNet18 with sigmoid output activation
python3 ./src/train.py --epochs 100 --resize 512 --batch-size 32 --val-size 32 --workers 8 --name sigmoid_plat --lr 0.001 --architecture ResNet18 --out-act sigmoid --augment True

# Run 100 iterations of ResNet34
python3 ./src/train.py --epochs 100 --resize 512 --batch-size 16 --val-size 16 --workers 8 --name plat --lr 0.001 --architecture ResNet34 --augment True

# Run 100 iterations of ResNet34 with sigmoid output activation
python3 ./src/train.py --epochs 100 --resize 512 --batch-size 16 --val-size 16 --workers 8 --name sigmoid_plat --lr 0.001 --architecture ResNet34 --out-act sigmoid --augment True

# Run 100 iterations of ResNet50
python3 ./src/train.py --epochs 100 --resize 512 --batch-size 8 --val-size 8 --workers 8 --name plat --lr 0.001 --architecture ResNet50 --augment True

# Run 100 iterations of ResNet50 with sigmoid output activation
python3 ./src/train.py --epochs 100 --resize 512 --batch-size 8 --val-size 8 --workers 8 --name sigmoid_plat --lr 0.001 --architecture ResNet50 --out-act sigmoid --augment True

# Run 100 iterations of ResNet50
python3 ./src/train.py --epochs 100 --resize 512 --batch-size 8 --val-size 8 --workers 8 --name plat --lr 0.001 --architecture ResNet50 --augment True

# Run 100 iterations of ResNet50 with sigmoid output activation
python3 ./src/train.py --epochs 100 --resize 512 --batch-size 8 --val-size 8 --workers 8 --name sigmoid_plat --lr 0.001 --architecture ResNet50 --out-act sigmoid --augment True

## Runs on the 24GB GPU

# Run 100 iterations of ResNet18
python3 ./src/train.py --epochs 100 --resize 1024 --batch-size 32 --val-size 32 --workers 8 --name plat --lr 0.001 --architecture ResNet18 --augment True

# Run 100 iterations of ResNet18 with sigmoid output activation
python3 ./src/train.py --epochs 100 --resize 1024 --batch-size 32 --val-size 32 --workers 8 --name sigmoid_plat --lr 0.001 --architecture ResNet18 --out-act sigmoid --augment True

# Run 100 iterations of ResNet34
python3 ./src/train.py --epochs 100 --resize 1024 --batch-size 16 --val-size 16 --workers 8 --name plat --lr 0.001 --architecture ResNet34 --augment True

# Run 100 iterations of ResNet34 with sigmoid output activation
python3 ./src/train.py --epochs 100 --resize 1024 --batch-size 16 --val-size 16 --workers 8 --name sigmoid_plat --lr 0.001 --architecture ResNet34 --out-act sigmoid --augment True

# Run 100 iterations of ResNet50
python3 ./src/train.py --epochs 100 --resize 1024 --batch-size 8 --val-size 8 --workers 8 --name plat --lr 0.001 --architecture ResNet50 --augment True

# Run 100 iterations of ResNet50 with sigmoid output activation
python3 ./src/train.py --epochs 100 --resize 1024 --batch-size 8 --val-size 8 --workers 8 --name sigmoid_plat --lr 0.001 --architecture ResNet50 --out-act sigmoid --augment True

# Run 100 iterations of ResNet101
python3 ./src/train.py --epochs 100 --resize 1024 --batch-size 4 --val-size 8 --workers 8 --name plat --lr 0.001 --architecture ResNet101 --augment True

# Run 100 iterations of ResNet101 with sigmoid output activation
python3 ./src/train.py --epochs 100 --resize 1024 --batch-size 4 --val-size 8 --workers 8 --name sigmoid_plat --lr 0.001 --architecture ResNet101 --out-act sigmoid --augment True


## Best models will be run with mixed-precision for comparion with bigger batch-size

python3 ./src/train.py --epochs 50 --resize 1024 --batch-size 8 --val-size 8 --workers 10 --name 50_epoch_mix --lr 0.001 --architecture ResNet50 --mix-precision 1 

python3 ./src/train.py --epochs 50 --resize 1024 --batch-size 4 --val-size 4 --workers 10 --name 50_epoch_mix --lr 0.001 --architecture ResNet101 --mix-precision 1

python3 ./src/train.py --epochs 50 --resize 1024 --batch-size 8 --val-size 8 --workers 10 --name 50_epoch_mix_sig --lr 0.001 --architecture ResNet50 --mix-precision 1 --out-act sigmoid

python3 ./src/train.py --epochs 50 --resize 1024 --batch-size 4 --val-size 4 --workers 10 --name 50_epoch_mix_sig --lr 0.001 --architecture ResNet101 --mix-precision 1 --out-act sigmoid

python3 ./src/train.py --epochs 150 --resize 1024 --batch-size 8 --val-size 8 --workers 10 --name 150_epoch --lr 0.001 --architecture ResNet50 --mix-precision 1

python3 ./src/train.py --epochs 150 --resize 1024 --batch-size 4 --val-size 4 --workers 10 --name 150_epoch --lr 0.001 --architecture ResNet101 --mix-precision 1



