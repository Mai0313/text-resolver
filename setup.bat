@echo off
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118;
pip install dvc dvc-gdrive hydra-colorlog hydra-core hydra-optuna-sweeper tensorboard tensorboardX rootutils wget;
pip install autoroot autorootcwd ipykernel lightning==2.0.8 matplotlib opencv-python opencv-python-headless Pillow==9.5.0 soundfile pyrootutils
pause
