# General setup for google colab
* Download train.zip to google drive
* Download resnet50_fconv_model_best.pth.tar to google drive
## Mount google drive to colab
* In colab choose **Runtime->Change runtime type->GPU**
* In a cell insert
```
from google.colab import drive
drive.mount("/content/gdrive")
```
## Unzip train.zip to colab
```
!cd /content/data/
! unzip /content/gdrive/MyDrive/train.zip
```
