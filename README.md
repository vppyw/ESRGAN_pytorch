# ESRGAN

This repo is my implementation about [ESRGAN](https://github.com/xinntao/ESRGAN) and other related works with pytorch.

## file sctructure

  ```python
  .
  ├── ckpt
  ├── data
  ├── hr_dataset.py
  ├── metrics.py
  ├── README.md
  ├── requirements.txt
  ├── sr_models.py
  ├── kd_utils.py
  ├── run_kd.py
  ├── run_esrgan.py
  ├── run_kd.py
  ├── gen_lr2hr.py
  ├── gen_interpolate.py
  └── interpolate_model.py
  ```

## Train SISR task
  - Train PSNR-based model

    ```bash
    python3 run_esrgan.py --model_type psnr
    ```

  - Train GAN-based model

    ```bash
    python3 run_esrgan.py --model_type gan
    ```

  - For more details

    ```bash
    python3 train.py --help
    ```
  
  - KD SRResnet by Attention Map
    
    ```bash
    python3 run_kd.py --help
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

  ```bash
  python3 metrics.py --help
  ```

## HR generation 

  - Generate HR from LR

    ```bash
    python3 gen_lr2hr.py
    ```

  - Generate model interpolate result 

    ```bash
    python3 gen_interpolate.py
    ```
    ![interpolate_result](./inter_result/inter_img_001.png)
   
## Result

  - SResnet with RRDB
    ![SRResnet with RRDB](./psnr_result/Set14/1.png)

  - ESRGAN
    ![EESRGAN](./gan_result/Set14/1.png)
    


## package requirements

  ```
  pip install -r requirements.txt
  ```
