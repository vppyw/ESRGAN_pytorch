# ESRGAN

This repo is my implementation about [ESRGAN](https://github.com/xinntao/ESRGAN) and other related works with pytorch.

##  file sctructure

    ```python
    .
    ├── ckpt
    ├── data
    ├── hr_dataset.py
    ├── metrics.py
    ├── README.md
    ├── requirements.txt
    ├── sr_models.py
    ├── train_gan.sh
    ├── train_psnr.sh
    └── train.py
    ```

## Train SISR task
    - Train PSNR-based model

        ```bash
        python3 train.py --model_type psnr
        ```

    - Train GAN-based model

        ```bash
        python3 train.py --model_type gan
        ```

    - For more details

        ```bash
        python3 train.py --help
        ```

## Evaluate single Image performance (PSNR/SSIM)

    ```bash
    python3 metrics.py \
        --lr_img LR_IMG \
        --hr_img HR_IMG \
        --load_model \
        --model MODEL
    ```

    To see more details.

    ```
    python3 metrics.py --help
    ```

## package requirements

   ```python
   pip install -r requirements.txt
   ```
