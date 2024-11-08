# UniM2CT
Unified Face Attack Detection via Hybrid Multi-modal CNN-ViT Network: Enhancing Representation Capability
------
## Description
-------
## Datasets
#### SiW-Mv2
You can get SiW-Mv2 by this [github](https://github.com/CHELSEA234/Multi-domain-learning-FAS).
#### FaceForensics++
You can get FaceForensics++ by this [github](https://github.com/ondyari/FaceForensics).
#### GAN Fake Face
Digital manipulation attacks are generated via publicly available author codes:

[StarGAN](https://github.com/yunjey/stargan)

[StyleGAN](https://github.com/NVlabs/stylegan2)

[PGGAN](https://github.com/tkarras/progressive_growing_of_gans)
#### FFHQ
You can get FFHQ by this [github](https://github.com/NVlabs/ffhq-dataset).

------
## Requirement
Pillow==9.4.0 \
scikit_learn==1.3.2 \
scipy~=1.10.0 \
timm==0.9.12 \
torch==2.1.2+cu118 \
torch_dct==0.1.6 \
scikit-learn~=1.3.2 \
matplotlib==3.6.2 \
numpy==1.23.5
## Train
Training code execution:
```python
python -u train.py
```
------
