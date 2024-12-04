# Photo2PixelWithAlpha

---
日本語 | [English](./README_en.md) | [简体中文](./README_cn.md)

photo2pixelWithAlpha オリジナルのphoto2pixelにアルファ付きの画像を加工可能に改変したものです。
ピクセルアートっぽくしてくれるツールです。オリジナルのphoto2pixel制作者に敬意を表します。

[Online Tool](https://photo2pixel.co) |
[Colab](https://colab.research.google.com/drive/108np4teybhBXHKbPMZZ1fykDuUeF2aw8?usp=sharing) |
[Tutorial](#Tutorial)

## Prerequisites
- python3
- pytorch (for algorithm implementation)
- pillow (for image file io)
- opencv-python

## Tutorial
---

### 加工ターゲットの画像ファイル
以下のディレクトリに加工したい画像ファイルを起きます。
サブディレクトリ等も全部収集して処理します。

```
images/origin_images/
```

### 処理の開始
コマンドラインで動作します。下記のとおりです。


```bash
# venv上で作業しましょう
python -m venv .venv
source .venv/bin/activate

# pipを最新にする 
pip install --upgrade pip

# venv上の環境を整える
pip install -r requirements.txt

# main.pyを走らせる
python main.py

# images/origin_image 以下に配置されたpng画像全部を指定のパラメータに変換する
# この時、アルファ値があるとそれも含めて全部変換してくれる

# or use custom param
python convert.py --kernel_size 12 --pixel_size 12 --edge_thresh 128
```

| Parameter   |                                Description                                |    Range    |               Default               |
|-------------|:-------------------------------------------------------------------------:|:-----------:|:-----------------------------------:|
| input       |                             input image path                              |      /      | ./images/example_input_mountain.jpg |
| output      |                             output image path                             |      /      |            ./result.png             |
| kernel_size |             larger kernel size means smooth color transition              |  unlimited  |                 10                  |
| pixel_size  |                           individual pixel size                           |  unlimited  |                 16                  |
| edge_thresh | the black line in edge region, lower edge threshold means more black line |    0~255    |                 100                 |


### input
素材となるディレクトリを指定する。再帰的にディレクトリ階層を潜っていくため、不要な処理をしたくない場合は除いておく方が無難。

### output
inputで階層が深くなっていようとも、ここで記載したPathのRootにのっぺりと出力する。


### kernel_size
階調。少ない数字ほどパキッとする。


### pixel_size
Unityの2Dゲーム素材として使うなら4の倍数を指定しておくのが無難。


### コピペ用
```bash
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python main.py --input ./images/origin_image/characters/1005400 --output result/1005400 --kernel_size 16 --pixel_size 8 --edge_thresh 255  
```
