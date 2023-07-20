## 数据增强--工具箱自带
- 以configs/yolov7/hyp.scratch.tiny.yaml中的data.train_transforms为例. 
它指定了一组应用于图像或标签的数据增强操作，用以生成作为模型输入或损失函数输入的数据。这些数据增强函数定义在 **mindyolo/data/dataset.py** 中。
```yaml
  train_transforms:
    - {func_name: mosaic, prob: 1.0, mosaic9_prob: 0.2, translate: 0.1, scale: 0.5}
    - {func_name: mixup, prob: 0.05, alpha: 8.0, beta: 8.0, needed_mosaic: True}
    - {func_name: hsv_augment, prob: 1.0, hgain: 0.015, sgain: 0.7, vgain: 0.4}
    - {func_name: pastein, prob: 0.05, num_sample: 30}
    - {func_name: label_norm, xyxy2xywh_: True}
    - {func_name: fliplr, prob: 0.5}
    - {func_name: label_pad, padding_size: 160, padding_value: -1}
    - {func_name: image_norm, scale: 255.}
    - {func_name: image_transpose, bgr2rgb: True, hwc2chw: True}
```
_注意：func_name表示数据增强方法名，prob表示该数据增强方法的执行概率，默认值为1_

上述yaml文件执行的具体操作如下：

- `mosaic`：以1.0的概率对输入的图片进行mosaic操作，即将4张不同的图片拼接成一张图片。mosaic9_prob表示使用9宫格方式进行拼接的概率，translate和scale分别表示随机平移和缩放的程度。

- `mixup`：以0.05的概率对输入的图片进行mixup操作，即将两张不同的图片进行混合。其中alpha和beta表示混合系数，needed_mosaic表示是否需要使用mosaic进行混合。

- `hsv_augment`: HSV增强, 以1.0的概率对输入的图片进行HSV颜色空间的调整，增加数据多样性。其中hgain、sgain和vgain分别表示对H、S、V通道的调整程度。

- `pastein `：以0.05的概率在输入的图片中随机贴入一些样本。其中num_sample表示随机贴入的样本数量。

- `label_norm`：将输入的标签从(x1, y1, x2, y2)的格式转换为(x, y, w, h)的格式。

- `fliplr`：以0.5的概率对输入的图片进行水平翻转，增加数据多样性。

- `label_pad`：对输入的标签进行填充，使得每个图片都有相同数量的标签。padding_size表示填充后标签的数量，padding_value表示填充的值。

- `image_norm`：将输入的图片像素值从[0, 255]范围内缩放到[0, 1]范围内。

- `image_transpose`：将输入的图片从BGR格式转换为RGB格式，并将图片的通道数从HWC格式转换为CHW格式。



对于测试数据增强函数

```yaml
  test_transforms:
    - {func_name: letterbox, scaleup: False}
    - {func_name: label_norm, xyxy2xywh_: True}
    - {func_name: label_pad, padding_size: 160, padding_value: -1}
    - {func_name: image_norm, scale: 255. }
    - {func_name: image_transpose, bgr2rgb: True, hwc2chw: True }
```
- `letterbox`:将图片按照等比例缩放后，用固定的背景色填充到指定大小。其中scaleup表示是否允许将图片放大，以适应指定的大小。

- `label_norm`:标签归一化,将输入的标签从(x1, y1, x2, y2)的格式转换为(x, y, w, h)的格式。

- `label_pad`:标签填充,对输入的标签进行填充，使得每个图片都有相同数量的标签。padding_size表示填充后标签的数量，padding_value表示填充的值。

- `image_norm`:图片归一化,将输入的图片像素值从[0, 255]范围内缩放到[0, 1]范围内。

- `image_transpose`:图片变换,将输入的图片从BGR格式转换为RGB格式，并将图片的通道数从HWC格式转换为CHW格式。



## 数据增强--自定义
编写指南：
- 在mindyolo/data/dataset.py文件COCODataset类中添加自定义数据增强方法
- 数据增强方法的输入通常包含图片、标签和自定义参数。
- 编写函数体内容，自定义输出，如下
```python
#mindyolo/data/dataset.py
    def augmentation_fn(self, image, labels,args):
        ...
        return image, labels
```
使用指南：
- 在模型的yaml文件中，以字典的形式定义此数据增强方法，详情请参见本文上述指导。