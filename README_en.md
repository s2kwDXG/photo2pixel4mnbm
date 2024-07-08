# ![LOGO](images/doc/favicon-original.png) Photo2Pixel

---
[日本語](./README.md) | English | [简体中文](./README_cn.md)

[Online Tool](https://photo2pixel.co) |
[Colab](https://colab.research.google.com/drive/108np4teybhBXHKbPMZZ1fykDuUeF2aw8?usp=sharing) |
[Tutorial](#Tutorial)

photo2pixel is an algorithm converting photo into pixel art. There is an [online converter photo2pixel.co](https://photo2pixel.co)
. you can try different combination of pixel size and edge threshold to get the best result.

<img src="images/doc/mountain_8bit_style_pixel.png" style="max-width: 850px" alt="mountain 8bit style pixel art"/>
<img src="images/doc/holy_temple_8bit_style_pixel.png" style="max-width: 850px" alt="holy temple 8bit style pixel art">

## Prerequisites
- python3
- pytorch (for algorithm implementation)
- pillow (for image file io)
- opencv-python

## Tutorial
---
or you can run it with command as bellow:

```bash
# venv上で作業します
python -m venv .venv
source .venv/bin/activate

# pipを最新にする 
pip install --upgrade pip

# venv上の環境を整える
pip install -r requirements.txt

# main.pyを走らせる
python main.py

# images/

# or use custom param
#python convert.py --kernel_size 12 --pixel_size 12 --edge_thresh 128
```

| Parameter   |                                Description                                |    Range    |               Default               |
|-------------|:-------------------------------------------------------------------------:|:-----------:|:-----------------------------------:|
| input       |                             input image path                              |      /      | ./images/example_input_mountain.jpg |
| output      |                             output image path                             |      /      |            ./result.png             |
| kernel_size |             larger kernel size means smooth color transition              |  unlimited  |                 10                  |
| pixel_size  |                           individual pixel size                           |  unlimited  |                 16                  |
| edge_thresh | the black line in edge region, lower edge threshold means more black line |    0~255    |                 100                 |