影像分類攻擊方法模組圖形化介面 使用說明書
===

## 執行說明
### 直接執行 EXE檔案
- `ART_Tool_CPU.exe` 為影像分類攻擊方法模組程式。執行EXE檔方式支援後接參數設定，詳情請至[執行參數說明](#執行參數說明)查看。

- **本程式只能運行在 **Windows 64位元** 作業系統上。其中「ART_Tool_CPU.exe」不支援GPU運行，只能運行在CPU上。**


## 透過Python執行
- 本程式可以運行在Python相關虛擬環境中，像是Anaconda。

- 請注意，本程式只支援Python 3.8版本，在安裝相關環境前，請先注意Python版本是否正確。

- ```requirements.txt``` 為安裝Python相關環境需求，可透過在終端機中執行 ```pip3 install -r requirements.txt``` 來安裝。

- 安裝環境完成後， 執行```python ART_Tool_RGB.py```就可以執行圖形化介面。

- 執行Python檔方式也支援後接參數設定，詳情請至[執行參數說明](#執行參數說明)查看。

## 圖形化介面說明
<img src="https://i.imgur.com/QdrGur4.png" alt="圖形化介面" width="450"/>

- 「選擇目標模型」、「選擇攻擊模型」、「選擇攻擊方法」為透過下拉選單來選擇
- 「雜訊參數」需自行輸入，否則為預設值「0.5」
- 。如果為輸入eps擾動參數，建議輸入值介於0到1之間的浮點數，如果為輸入conf置信度，則建議輸入值介於0到16之間的整數。
- 按下「選擇資料集」後，會進入選擇資料夾介面，接受的資料夾格式請參見[資料集說明](#資料集說明)。
- 在點選資料夾後，按下「選擇資料夾」鍵，就可選擇該資料夾為資料集。

## 資料集說明
- 選擇一個資料夾路徑，該資料夾底下必須包含名稱為「```train```」及「```test```」的兩個子資料夾，各自代表訓練資料集及測試資料集。
- 其中資料集的架構，同一分類的圖片必須存在同一資料夾，且資料夾名稱為該類別的名稱。圖片檔名不影響程式運作。可接受PNG檔及JPEG檔的圖片檔案類型。

### MNIST手寫辨識資料集 範例
- 以下為 MNIST手寫辨識資料集架構，以作為可接受資料集架構範例
    ```
    mnist/                     <=== 圖形介面選擇的資料集路徑
    ├── train/                 <=== 訓練資料集
    │   ├── 0/                 <=== 類別為「0」的圖片資料夾
    │   │   ├── train_0_1.png  <=== 檔名不影響
    │   │   ├── train_0_2.png
    │   │   └── ...
    │   │
    │   ├── 1/      <=== 類別為「1」的圖片資料夾
    │   ├── ...
    │   └── 9/
    │
    └── test/       <=== 測試資料集
        ├── 0/
        ├── ...
        └── 9/
    ```
    
# 執行參數說明
- 本程式支援參數設定，指令格式為
    ```bash=
    $ {ART_Tool_GPU.exe 或 python ART_Tool_RGB.py} [-h] [--interface INTERFACE] 
        [--cuda CUDA] [--dataset-path DATASET_PATH] [--num-workers NUM_WORKERS] 
        [--predict-model PREDICT_MODEL] [--attack-model ATTACK_MODEL] 
        [--white-box] [--attack-func ATTACK_FUNCTION] [--max-iter MAX_ITER] 
        [--eps EPS] [--conf CONFIDENCE] [--epochs EPOCHS] 
        [--batch-size BATCH_SIZE] [--optim OPTIM] [--lr LR] [--momentum MOMENTUM]
    ```
    
## 範例
```bash=
$ ART_Tool_CPU.exe –-cuda 1 --epochs 5 --lr 0.01
$ python ART_Tool_RGB.py --num-workers 4 --norm
```

## 個別參數說明


| 參數            | 後接參數        | 預設值 | 說明                                                                          |
| --------------- | --------------- | ------ | ----------------------------------------------------------------------------- |
| -h, --help      | 無              | 無     | 顯示參數說明                                                                  |
| --interface     | INTERFACE       | 1      | 選擇設定參數方式 (1:GUI介面, 2:標準輸入, 3:執行參數)                          |
| --cuda          | CUDA            | 0      | 設定運行GPU的id (-1:CPU, ≥0:GPU CUDA id)                                      |
| --dataset-path  | DATASET_PATH    | 無     | 設定資料集路徑                                                                |
| --num-workers   | NUM_WORKERS     | 8      | 設定執行緒數量                                                                |
| --predict-model | PREDICT_MODEL   | 1      | 選擇目標模型 (1:LeNet5, 2:CNN, 3:AlexNet, 4:GoogLeNet, 5:VGG19, 6:ResNeXt101) |
| --attack-model  | ATTACK_MODEL    | 1      | 選擇攻擊模型 (1:LeNet5, 2:CNN, 3:AlexNet, 4:GoogLeNet, 5:VGG19, 6:ResNeXt101) |
| --white-box     | 無              | 否     | 設定白盒實驗                                                                  |
| --attack-func   | ATTACK_FUNCTION | 1      | 選擇攻擊方法(1:FGSM, 2:BIM, 3:PGD, 4:C&W L2, 5:C&W Linf)                      |
| --max-iter      | MAX_ITER        | 20     | 設定最大迭代次數                                                              |
| --eps           | EPS             | 0.1    | 設定擾動參數                                                                  |
| --conf          | CONFIDENCE      | 無     | 設定置信度                                                                    |
| --epochs        | EPOCHS          | 20     | 設定訓練回合數                                                                |
| --batch-size    | BATCH_SIZE      | 32     | 設定batch值                                                                   |
| --optim         | OPTIM           | SGD    | 設定優化器                                                                    |
| --lr            | LR              | 0.001  | 設定學習速率                                                                  |
| --momentum      | MOMENTUM        | 0.9    | 設定SGD的 Momentum值                                                          |
   
