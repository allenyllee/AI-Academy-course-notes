# 實戰演練－Python 機器學習程式設計

###### tags: `python` `ML`

:::info
編輯請按左上角的筆<i class="fa fa-pencil"></i>或旁邊的雙欄模式<i class="fa fa-columns"></i>。請以登入模式幫助編輯喔！
:::

[toc]

# API reference

- [API Reference — pandas 0.22.0 documentation](https://pandas.pydata.org/pandas-docs/stable/api.html?highlight=dataframe#dataframe)
- [API Reference — scikit-learn 0.19.1 documentation](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets)
    - [Choosing the right estimator — scikit-learn 0.19.1 documentation](http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
    - [Classifier comparison — scikit-learn 0.19.1 documentation](http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)
    - [Comparing Python Clustering Algorithms — hdbscan 0.8.1 documentation](https://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html)

- [The Matplotlib API — Matplotlib 2.1.1 documentation](https://matplotlib.org/api/index.html)
    - [The Pyplot API — Matplotlib 2.1.1 documentation](https://matplotlib.org/api/pyplot_summary.html)
        - [Pyplot tutorial — Matplotlib 2.0.2 documentation](https://matplotlib.org/users/pyplot_tutorial.html)
        - [Tight Layout guide — Matplotlib 2.0.2 documentation](https://matplotlib.org/users/tight_layout_guide.html)
        - [A simple plot with a custom dashed line — Matplotlib 2.1.0 documentation](https://matplotlib.org/2.1.0/gallery/lines_bars_and_markers/line_demo_dash_control.html)
        - [Line-style reference — Matplotlib 2.1.2 documentation](https://matplotlib.org/gallery/lines_bars_and_markers/line_styles_reference.html?highlight=line%20style%20reference)
- [Routines — NumPy v1.12 Manual](https://docs.scipy.org/doc/numpy-1.12.0/reference/routines.html)
    - [Index — NumPy v1.12 Manual](https://docs.scipy.org/doc/numpy-1.12.0/genindex.html)
- [The Python Standard Library — Python 3.3.7 documentation](https://docs.python.org/3.3/library/index.html)
    - [Python Tutorial](https://www.tutorialspoint.com/python/index.htm)

# Day 1 (02/07)

[課程投影片](https://docs.google.com/presentation/d/16i21V0m2qMR7zPf_YlM2QK9LigDWY0FBSGEC9enbUmE/edit#slide=id.g305b13c770_1_10)

## Scikit-learn & ML Introduction

### Introduction

主要用 [Scikit-learn](http://scikit-learn.org/stable/) 這個套件來教大家如何用 Python 來實踐機器學習。

主要會學到

- 機器學習的基本概念
- 簡單的 Regression 和 SVM 的模型

學習基礎模型的用意在於奠定複雜模型和深度學習的基礎。

練習使用機器學習

- 手寫圖案的辨識
- 預測房價
- ...

#### 機器學習基本流程

處理機器學習問題的基本步驟：

1. 定義問題

    :::info
    根據目的，定義問題的方式通常會不一樣
    :::
    
2. 搜集、清理資料

    :::info
    依據對問題的了解，去搜集相關的資料。
    蒐集到資料之後，也會需要對資料做清理 (e.g. 缺失值的補值、離群值的去除)，有時候也會需要對資料做正規化或是 One Hot Encoding (標籤編碼)
    :::
    
    :::warning
    定義問題和搜集資料的過程，通常需要與了解 Domain Know-how 的人合作跟溝通。
    這會是解決問題的成功關鍵。
    :::
    
    > 第 3 步驟和第 4 步驟是接下來的重點
    > [name=助教]

3. 選擇及建立模型

    :::info
    如何選擇適當的模型及參數處理不同的問題
    :::
    
4. 分析結果及修正模型

    :::info
    該如何評估預測的結果
    :::
    
    > 為了得到更好的預測結果，往往我們會需要重複上述的步驟。例如，重新定義問題、搜集更多資料、修正模型
    > [name=助教]
    
5. 結果呈現

    > 為了讓老闆或聽眾(金主)買單，資料視覺化和簡報技巧也是很重要的一環
    > [name=助教]

#### 為何選用 Scikit-learn?

- 攔括了前面4個步驟所會用到的工具
- 將機器學習會用到的工具分為六大類:
    - Classification
    - Regression
    - Clustering
    - Dimensionality reduction
    
    :::info
    前4類與模型相關
    :::
    
    - Model selection
        
        :::info
        評估完模型之後，選擇模型及其參數
        :::
        
    - Preprocessing
    
        :::info
        e.g. 標籤編碼、訓練/測試資料集的分割
        :::

### Scikit Learn 內建的資料集

> 初次踏入機器學習的領域，也許心中已有想要解決的問題，或許也可能沒有，為了方便大家快速上手，Scikit-learn 提供了完整的資料集，可以讓大家練習實作機器學習不同的演算法，而不用去煩惱要解決什麼樣的問題及收集哪些資料。
> [name=助教]
 
資料集：[sklearn.datasets](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets)

- boston: 波士頓房價 (迴歸模型)
- breast_cancer: 威斯康辛州乳癌 (分類模型)
- diabetes: 糖尿病 (迴歸模型)
- digits: 數字辨識 (分類模型)
- iris: 鸢尾花 (分類模型)
- wine: 酒 (分類模型)
- ...

#### 手寫數字資料集

以手寫數字為例，示範如何載入 Scikit-learn 內建的資料集。

```python=
# 載入資料集的方式: datasets.load_資料集名稱()
from sklearn import datasets
digits = datasets.load_digits()
```

手寫數字資料集總共有 1797 張 8x8 的數字圖像，每個像素為 1~16 灰階值。

```python=+
# images 為 8x8 的 numpy array
print(digits.images.shape) 
# 實際在訓練的時候會將 8x8 的像素轉成 64維度，1797 為sample size
print(digits.data.shape) 
# target 為 0~9，是每張數字圖像應該要對應到的數字
print(digits.target.shape)
```

可以用 matplotlib 來畫出數字圖像

```python=+
import matplotlib.pyplot as plt
# 顯示第一筆資料的目標值
print('digit: {}'.format(digits.target[0]))
# 畫出第一筆資料的圖像
plt.imshow(digits.images[0], cmap=plt.cm.binary, interpolation='nearest') 
plt.show()
```

下列範例會將「手寫數字資料集」的前 64 張圖像，顯示成 8x8 的矩陣。

```python=+
# Jupyter 的 magic function
%matplotlib inline
# 設定 Figure 的大小 (單位為英吋)
fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# 畫出數字: 每張圖都是 8x8 像素
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    
    # 在圖像的左下角標上目標值
    ax.text(0, 7, str(digits.target[i]))
```

#### 練習題：鳶尾花 (Iris)

從花的特徵去預測花的種類

```python=
from sklearn import datasets
# 載入鳶尾花資料集
iris = datasets.import_iris()
```

透過資料的描述對資料進行暸解

了解 4 種 features 分別對應什麼名稱

```python=+
print(iris.DESCR)
print(iris.data.shape)
```

:::info
- 150 筆資料 (包含3種鳶尾花，每種各50朵)
- 4 個特徵值 (連續型資料)
    - sepal length = 萼片長度
    - sepal width = 萼片寬度
    - petal length = 花瓣長度
    - petal width = 花瓣寬度
:::

了解三種labels分別對應什麼花名

```python=+
print(iris.target.shape)
print(iris.target_names)
```
    
- 花種學名 (類別型資料)
    - 0 -> setosa
    - 1 -> versicolor
    - 2 -> virginica

### Scikit Learn 創建虛擬資料的方法

除了內建的資料集之外，Scikit-learn 也提供方法讓我們可以自行設計資料來學習。這些方法可以透過 `datasets.make_Xxx()` 來呼叫。

底下會以 `make_blobs()` 這個方法來做示範。

首先載入相關套件。

```python=
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from myfun import plot_decision_regions
%matplotlib inline
```

接著產生 1000 個點的資料集，平均分成 3 類，每個點有兩個特徵值 (預設行為)，每類資料點的特徵值會以高斯分佈散佈在該類中心點的周遭。

> random_state 可用來固定住亂數產生器的種子。讓實驗具有可重複性，避免隨機性造成的差異。

```python=+
# 產生 3 個分類的資料集，可用來練習分類演算法
centers = [[-5, 0], [0, 1.5], [5, -1]]
X, y = make_blobs(n_samples=1000, centers=centers, random_state=40)
```

這份資料集可以用下列方式達到資料視覺化的目的：

- 用平面座標來表示每個點的兩個特徵值
- 用顏色來表示每個點的分類

```python=+
color = "rbg"
# 將每個點的分類 (0,1,2) 對應到 "rgb" 三原色
color = [color[y[i]] for i in range(len(y))]
# 以散點圖進行資料視覺化
plt.scatter(X[:,0], X[:,1], c = color)
```

### Scikit Learn 資料前處理的常見方法

#### Training & Testing Data Split

:::info
`train_test_split()` 可將資料集拆分成訓練資料集和測試資料集，內有參數可調整測試資料佔整體的比例。
:::

> 測試資料集主要是用來評估模型預測的好壞。
> 
> 那可能會有人問，為什麼不直接用 training data 做訓練且用 training data 做預測的評估呢？那麼不是就可以讓模型看過更多的資料，讓模型預測的更好？
> 
> 其中的問題就在於我們是否該讓模型看過的資料來做模型預測好壞的評估呢？
> 
> 以唸書和考試來做比喻，學生如果看過考試卷，就可以用背答案的方式拿到滿分，而無法正確評估出學生的實力。
> 
> 同樣的道理，要用模型沒有看過的資料去做評估，預測好壞的程度評比才會具有意義。
>
> [name=助教]

載入程式庫

```python=
from sklearn.model_selection import train_test_split
import numpy as np
```

產生練習資料集

```python=+
# X 為 0, 1, ..., 49 重新拆成 10x5 陣列
X = np.arange(50).reshape(10,5)
# y 為 0, 1, ..., 9
y = np.arange(10)
```

將資料集拆分成訓練用和測試用的部分

- test_size: 保留資料集的最後多少比例為測試用，剩餘的用來訓練。一般會設定在 0.2~0.25。
- random_state: 將資料打亂的方式 (設定成整數會變成亂數產生器的種子，也可直接指定成亂數產生器，假如 shuffle = False 則不會有作用)
- shuffle: 控制是否要先將資料集先打亂再做拆分。

```python=+
# 可試著調整 shuffle 和 random_state 觀察結果
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42, shuffle = False)
```

#### Data Normalization (資料正規化)

依造資料分佈的不同，採取的資料正規化的方式也不一樣。

不過目前最常用的正規化方式是 Z-score，它是用線性轉換的方式，將資料的平均值平移至 0，且將標準差縮放至 1。

另外一種方式為 minmax_scale，它也是用線性轉換的方式，將資料映射到 `[0, 1]` 的區間上。

**Z-score 的公式**

$$ x_{\mathrm{new}} = \frac{x_{\mathrm{orig}} - \mu}{\sigma} $$

**minmax 的公式**

$$ x_{\mathrm{new}} = \frac{x - \mathrm{min}(x)}{\mathrm{max}(x) - \mathrm{min}(x)} $$

**資料正規化的範例程式**

載入 Scikit-learn 中的資料預處理套件。

```python=
from sklearn import preprocessing
import numpy as np
```

產生一個 3 筆，每筆有 4 個特徵值的資料集。

```python=+
a = np.array([
    [10, 2.7, 3.6, 5],
    [-100, 5    , -2, 10],
    [120, 20, 40, 50]],
    dtype=np.float64)
```

定義一個函數，包裝 scikit-learn 提供的兩種資料正規化的方法。

```python=+
# minmax 正規化所使用的特徵範圍預設值設定成 0~1
def normalize(x, axis, method, minmax_range = (0,1)):
    if method == 'z-score':
        x_scaled = preprocessing.scale(a, axis = axis)
    elif method== 'minmax':    
        x_scaled = preprocessing.minmax_scale(a, axis = axis, feature_range = minmax_range)

    return x_scaled
```

- Ｚ-Score

    axis 參數用來控制正規化的方向
    
    - axis = 0 (縱向 - 同一個 column)
    - axis = 1 (橫向 - 同一個 row)

    ```python=+
    # 改變 axis，觀察結果如何變化
    axis = 1
    
    a_scaled = normalize(a, axis, method = 'z-score')
    print(a_scaled)
    ```
    
    觀察正規化之後，資料的平均值和標準差
    
    ```python=+
    print(a_scaled.mean(axis = axis))
    print(a_scaled.std(axis = axis))
    ```
    
- Minmax Scale

    axis 參數用來控制正規化的方向
    minmax_range 參數用來控制映射到的特徵值範圍
    ```python=+
    # 改變 axis，觀察結果如何變化
    axis = 1
    # 改變 minmax_range 觀察結果如何變化
    minmax_range = (0, 1)
    
    a_scaled = normalize(a, axis, method = 'minmax', minmax_range)
    print(a_scaled)
    ```
    
    觀察正規化之後，資料的平均值和標準差
    
    ```python=+
    print(a_scaled.mean(axis = axis))
    print(a_scaled.std(axis = axis))
    ```
    
> 隨著用途不同，其實我們都會用不同的轉換方式
> 大家可以依據問題，自行定義適合的轉換方式
> [name=助教]

**為何需要將資料正規化？**

- 提升預測準確度
- 提升模型效率
 
> 當不同特徵值之間的尺度差異太大時，等高線畫出來的形狀會變成狹長的橢圓形，機器學習演算法在找全域極大(小)值的時候，爬坡的方向也許會偏離中心點太遠 (路線走得會比較迂迴)，會變得沒有效率而且影響準確率。
> 
> 將資料正規化之後，等高線圖會變成同心圓，在找全域極大(小)值的時候，會較快到達靶心。
> [name=助教]

**程式範例：資料正規化對於預測準確率的影響**

載入相關程式庫

```python=
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_classification 

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 8, 5
```

製作假資料

```python=+
# 製作具有兩個特徵值的分類資料
X, y = make_classification(
    n_samples = 300, 
    n_features = 2,
    n_redundant = 0,
    n_informative = 2, 
    random_state = 22,
    n_clusters_per_class = 1, 
    scale = 100)

# 將資料可視化
plt.scatter(X[:, 0], X[:, 1], c = y)
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()
```
---
不做資料正規化的預測結果

```python=+
# 將資料分成訓練及測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# 載入 SVM 分類器
from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)
prediction = model.predict(X)
accuracy = model.score(X_test, y_test)
print('accuracy before normalization:%.2f'%accuracy)
```

得到的預測準確率為 0.51

---
做了資料正規化的預測結果

```python=+
# 將資料做 Z-score 正規化
X = preprocessing.scale(X)
# 將資料分成訓練及測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

model = SVC()
model.fit(X_train, y_train)
prediction = model.predict(X)
accuracy = model.score(X_test, y_test)
print('accuracy after normalization:%.2f'%accuracy)
```

得到的預測準確率就可提升至 0.93

#### One Hot Encoding (獨熱編碼)

One Hot Encoding 是分析類別資料常會用到的前處理方法，類別標籤要先轉換成數字才能當作模型的輸入。

載入相關程式庫

```python=
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
```

使用標籤編碼器產生文字標籤的編碼

```python=+
# 產生 X 特徵值的編碼器
encX = LabelEncoder()
encX.fit(['看電視', '讀書', '音樂', '游泳'])

# 產生 y 分類的編碼器
ency = LabelEncoder()
ency.fit(['是', '否'])
```

:::info
想要編碼的標籤必須要可排序和可雜湊，應該不保證編碼順序
:::

```python=+
# '游泳' '看電視' '讀書' '音樂' 分別會轉成 0, 1, 2, 3
print(encX.classes_)
# 否、是分別會轉成 0, 1
print(ency.classes_) 
```

產生假資料

```python=+
data_Xy = {
    '興趣': ['看電視','讀書','音樂','看電視'],
    '成功與否': ['是', '否', '否', '是']
}
df = pd.DataFrame(data = data_Xy, index=['小明','小林','小英','小陳'])
# 指定欄位順序
df = df[['興趣', '成功與否']]
```

將文字標籤編碼成數字

```python=+
df_encode = df.copy()
df_encode['興趣'] = encX.transform(df_encode['興趣'])
df_encode['成功與否'] = ency.transform(df_encode['成功與否'])
```

將數字解碼回文字標籤

```python=+
# 手動預測
prediction = np.array([1, 0, 0, 1])
# 將預測完的結果做反轉換
df['prediction'] = ency.inverse_transform(prediction) 
```

> 將無序類別利用 One Hot Encoding 編碼成有順序性的數字可能會在某些模型上造成問題
> 
> 舉例來說，不同的興趣並沒有強度上(順序上)的相關性，但採用邏輯迴歸模型則會將之<span style="color:red">視為有強度上的差異</span>，這會導致模型無法做精準的預測。
> 
> 由於看電視、讀書和音樂等興趣彼此之間並沒有順序上的關聯性，最好將其轉換成三個互相垂直的維度。
> [name=助教]
 
底下的程式碼片段可將<span style="color:red">無序</span>的類別特徵轉換成數字以當作模型的輸入。

```python=+
X = pd.get_dummies(df.iloc[:,0])
```

> 有序類別利用 One Hot Encoding 進行編碼會是有意義的轉換，例如衣服的尺寸。
> 
> 需要注意編碼數字的順序要跟類別的順序一致。

載入相關程式庫

```python=
import pandas as pd
import numpy as np
```

定義類別對應的數字編碼

```python=+
# 用字典定義類別 -> 編碼
size_mapping = { 'S': 0, 'M': 1, 'L': 2, 'XL': 3 } 
label_mapping = { '否': 0, '是': 1 } 
```

產生假資料

```python=+
data_Xy = {
    '衣服size': [ 'XL', 'S', 'M', 'L'],
    '成功與否': [ '是', '否', '否', '是' ]
}
df = pd.DataFrame(data = data_Xy, index = [ '小明', '小林', '小英', '小陳' ])
df = df[['衣服size','成功與否']]
```

對類別進行編碼

```python=+
df_encode = df.copy()
df_encode['衣服size'] = df_encode['衣服size'].map(size_mapping)
df_encode['成功與否'] = df_encode['成功與否'].map(label_mapping)
```

### Scikit Learn 建立機器學習模型的方法

1. 選擇模型

    ![機器學習地圖](http://scikit-learn.org/stable/_static/ml_map.png "機器學習地圖" =636x396)

    ```python=
    from sklearn.svm import SVC
    ```

2. 建立模型

    ```python=+
    model = SVC()
    ```

3. 訓練模型

    ```python=+
    model.fit(X_train, y_train)
    ```

4. 利用模型預測結果

    ```python=+
    prediction = model.predict(X)
    ```

[Choosing the right estimator — scikit-learn 0.19.1 documentation](http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)

### Scikit Learn 評估模型好壞的方法

任何預測都需要評估準確性，Scikit Learn 用來評估模型的方法都在 `metrics` 裡面。

```python=
from sklearn import metrics
```

#### 回歸模型的評估方法

- Mean Absolute Error: 誤差絕對值的總和取平均

    $$ \mathrm{MAE} \triangleq \frac{\sum_{i=1}^{N}|e_i|}{N} $$
    
    ```python=+
    mae = metrics.mean_absolute_error(prediction, y)
    ```

    :::warning
    絕對值是不可微分的函數，會不好做最佳化
    :::

- Mean Squared Error: 誤差平方的總和取平均

    $$ \mathrm{MSE} \triangleq \frac{\sum_{i=1}^{N}e_i^2}{N} $$
    
    ```python=+
    mse = metrics.mean_squared_error(prediction, y)
    ```

- Root Mean Squared Error: MSE 開根號

    $$ \mathrm{RMSE} \triangleq \sqrt{\frac{\sum_{i=1}^{N}e_i^2}{N}} $$

    :::info
    如果將誤差當作雜訊，RMSE 就會是雜訊的平均能量
    :::
    
    :::warning
    Scikit Learn 並沒有提供直接計算 RMSE 的方法，所以要先算完 MSE 之後，手動開根號
    :::

- $R^2$ Score

    [Coefficient of determination - Wikipedia](https://en.wikipedia.org/wiki/Coefficient_of_determination)
    [Coefficient of determination - Wikiwand](https://www.wikiwand.com/en/Coefficient_of_determination)

$$
\begin{aligned}
R^{2} &\triangleq 1 - \frac{SS_{res}}{SS_{tot}}\\
&= 1 - \frac{\sum_{i=1}^{N} e_i^2}{\sum_{i=1}^{N}{(y_i-\bar{y})^2}}\\
&= 1 - \frac{\mathrm{total\ error}}{\mathrm{data\ variance}} \end{aligned}
$$

$R^2$ Score 用來評估預測倒底解釋了多少資料的變異性。

當預測零誤差時，$R^2$ 會是 0；預測越失準，誤差會比資料變異數大很多，會變成負的值。

接下來透過底下的範例，來觀察在資料上加入不同強度的雜訊對預測結果的影響，也藉以了解不同 metrics 的意義。

先載入相關程式庫

```python=
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

from sklearn import metrics
from sklearn import datasets
from sklearn.linear_model import LinearRegression

%matplotlib inline
```

定義一個方法用來畫出不同雜訊強度下的資料分佈，以及標注出不同的評估量測值。

```python=+
def linear_prediction(plot_dict):
    for noise in plot_dict:
        # 製作假資料並選用線性迴歸模型進行預測
        X, y = datasets.make_regression(n_features= 1, random_state = 42, noise = noise)
        model = LinearRegression()
        model.fit(X, y)
        prediction = model.predict(X)
        
        # 計算評估量測值
        mae = metrics.mean_absolute_error(prediction, y)
        mse = metrics.mean_squared_error(prediction, y)
        r2 = metrics.r2_score(prediction, y)
        
        # 畫圖
        plt.subplot(plot_dict[noise])
        plt.xlabel('prediction')
        plt.ylabel('actual')
        plt.tight_layout()
        plt.plot(prediction, y,'.')
        plt.title('Plot for noise: %d'%noise + '\n' 
                  + 'mae:%.2f'%mae + '\n' 
                  + 'mse:%.2f'%mse + '\n'
                  + 'r2:%.2f'%r2)
    plt.show()
```

**觀察不同 scatter plot 和 evaluation metrics 的變化**

[matplotlib.pyplot.subplot — Matplotlib 2.1.1 documentation](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html#matplotlib.pyplot.subplot)

`plot_dict` 內容結構為 noise:axis 的字典, 其中 noise 為字典的 key 值, 而 axis 值作為 subplot 分割的 row, col, index。

倘若 row, col 和 index 皆可用單位數字表示，則可以用單個 3 位數字來指定。

```python=+
# 指定圖表的大小 (寬, 高) 單位為英吋
rcParams['figure.figsize'] = 8, 4
# 1 row, 4 cols, 從左到右
plot_dict = { 1: 141, 9: 142, 18: 143, 1000: 144 }
linear_prediction(plot_dict)

# 指定圖表的大小 (寬, 高) 單位為英吋
rcParams['figure.figsize'] = 8, 10
# 2 rows, 2 cols, 從左至右，由上到下
plot_dict = { 1: 221, 9: 222, 18: 223, 1000: 224}
linear_prediction(plot_dict)
```

#### 二元分類模型的評估方法

[Confusion Matrix - Ｗikipedia](https://en.wikipedia.org/wiki/Confusion_matrix)

##### Accuracy (準確率)

$$ \mathrm{Accuracy} = \frac{\mathrm{TN+TP}}{\mathrm{TN + TP + FN + FP}} $$

以警察抓壞人為例，TP = 抓到壞人，TN = 真是好人，FP = 冤枉好人，FN = 放過壞人。

假如 10 個人中，只有一個壞人，那麼全當作是好人可以得到 TP = 0, TN = 9, FP = 0, FN = 1, 準確率會是

$$ Accuracy = \frac{9 + 0}{9 + 0 + 1 + 0} = \frac{9}{10} = 90\% $$

不過以警察的立場來說，在執法的過程會寧可冤枉一些好人，也不能放任壞人去為非作歹 (寧枉勿縱)。用準確率來評估會不太適合。

```python=
# Minor (linear algebra) - Wikiwand
# https://www.wikiwand.com/en/Minor_(linear_algebra)

# Adjugate matrix - Wikiwand
# https://www.wikiwand.com/en/Adjugate_matrix

# also see
# machine learning - What are the measure for accuracy of multilabel data? - Cross Validated
# https://stats.stackexchange.com/questions/12702/what-are-the-measure-for-accuracy-of-multilabel-data?newreg=c07f051ba18f41d8b8bf8a3a2db32235

# =========
# get_accuracy
# =========
# binary=True ，二元分類時，使用 TP, FP, TN, FN 的定義去做
# binary=False，多類別的時候，直接計算預測正確與全部的比例，不要分成TP, FP, TN, FN
# 
def get_accuracy(confusion_matrix, binary=False):
    if (binary):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

    correct = 0
    incorrect = 0
    total = 0
    
    i = 0
    while i < len(confusion_matrix) :
        for j in range(len(confusion_matrix)):
            if (i == j):
                correct += confusion_matrix[i,j] # 預測正確
                if (binary):
                    TP += confusion_matrix[i,j] # true positive: 真實為i，預測為i (confusion matrix 中的對角線項目)
                    tmp = np.delete(confusion_matrix, i, 0)
                    tmp = np.delete(tmp, j, 1)
                    #print(tmp)
                    TN += np.sum(tmp) # true negative: 真實不為i, 預測不為i (confusion matrix 中, row=col=i 以外的項目總合)
            if (i != j):
                if(confusion_matrix[i,j] != 0):
                    incorrect += confusion_matrix[i,j] # 預測錯誤
                if (binary):
                    if(confusion_matrix[i,j] != 0):
                        FN += confusion_matrix[i,j] # false negative: 真實為i, 預測不為i (confusion matrix中, row i上不為0的總數)
                    if(confusion_matrix[j,i] != 0):
                        FP += confusion_matrix[j,i] # false positive: 真實不為i, 預測為i (confusion matrix中, col i上不為0的總數)

        i += 1

    if (binary):
        correct = TP + TN
        incorrect = FP + FN
    
    total = correct + incorrect
    print("correct=",correct,"incorrect:",incorrect, "total=", total)
    return correct/total

confusion_matrix = metrics.confusion_matrix(actual, predicted)
print(get_accuracy(confusion_matrix))
print(get_accuracy(confusion_matrix,True)) # 多類別用 TP, FP, TN, FN 做出來怪怪的

# 比較 sklearn 的 accuracy_score
# sklearn.metrics.accuracy_score — scikit-learn 0.19.1 documentation
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
print('accuracy:%.3f'%(metrics.accuracy_score(actual, predicted)))
```

##### F-score

[$F_1$d-score - Wikipedia](https://en.wikipedia.org/wiki/F1_score)

$$ F_1 = 2 \cdot \frac{1}{\frac{1}{\mathrm{recall}} + \frac{1}{\mathrm{precision}}} = 2 \cdot \frac{\mathrm{precision} \cdot {\mathrm{recall}}}{\mathrm{precision} + \mathrm{recall}} $$

其中

$$ \mathrm{Recall} = \frac{\mathrm{TP}}{\mathrm{FN + TP}} $$

$$ \mathrm{Precision} = \frac{\mathrm{TP}}{\mathrm{TP + FP}} (\text{抓對幾成}) $$

用同樣的警察抓壞人為範例，召回率指得是該抓到的壞人實際抓到幾成 (破案率)，召回率高則壞人被抓到的比率高。精確率則是抓對人的比率，冤枉好人的比率變低，精確率就會上升。

因為只有一個壞人，只要他被抓到，召回率就是 100%，否則就是 0%。

精確率在還沒開始抓之前沒有意義，還沒抓到壞人之前是 0%，抓到壞人之後，每冤枉一個好人精確率都會下降。

![](https://scontent.ftpe8-1.fna.fbcdn.net/v/t1.0-9/27545494_2158162204194731_1612820888658569502_n.jpg?oh=1aa7ea38ef435099f9aeec0008e0c7cc&oe=5ADB5A73)

```python=
# Minor (linear algebra) - Wikiwand
# https://www.wikiwand.com/en/Minor_(linear_algebra)

# Adjugate matrix - Wikiwand
# https://www.wikiwand.com/en/Adjugate_matrix

# also see
# machine learning - What are the measure for accuracy of multilabel data? - Cross Validated
# https://stats.stackexchange.com/questions/12702/what-are-the-measure-for-accuracy-of-multilabel-data?newreg=c07f051ba18f41d8b8bf8a3a2db32235

# =========
# get_f1_score
# =========
# binary=True ，二元分類時，使用 TP, FP, TN, FN 的定義去做
# binary=False，多類別的時候，直接計算預測正確與全部的比例，不要分成TP, FP, TN, FN
# 
def get_f1_score(confusion_matrix, index):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for j in range(len(confusion_matrix)):
        if (index == j):
            TP += confusion_matrix[index,j] # true positive: 真實為i，預測為i (confusion matrix 中的對角線項目)
            tmp = np.delete(confusion_matrix, index, 0)
            tmp = np.delete(tmp, j, 1)
            #print(tmp)
            TN += np.sum(tmp) # true negative: 真實不為i, 預測不為i (confusion matrix 中, row=col=i 以外的項目總合)
        if (index != j):
            if(confusion_matrix[index,j] != 0):
                FN += confusion_matrix[index,j] # false negative: 真實為i, 預測不為i (confusion matrix中, row i上不為0的總數)
            if(confusion_matrix[j,index] != 0):
                FP += confusion_matrix[j,index] # false positive: 真實不為i, 預測為i (confusion matrix中, col i上不為0的總數)

    recall = TP / (FN + TP)
    precision = TP / (TP + FP)
    f1_score = 2 * 1/(1/recall + 1/precision)
    
    #print("recall=",recall,"precision:",precision, "f1_score=", f1_score)
    return f1_score


confusion_matrix = metrics.confusion_matrix(actual, predicted)

for i in range(10):
    print(i, get_f1_score(confusion_matrix, i))

# 比較 sklearn 的 classification_report
# sklearn.metrics.classification_report — scikit-learn 0.19.1 documentation
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report
print(metrics.classification_report(actual, predicted)) 
```

#### 多元分類的評估方法

在分類問題上，所有的模型評估方法基本上都可以由 confusion matrix 德書來

### 防止資料過度擬合方法 (Overfitting)

> 在做模型評估時，有時候我們會發現 training 得到很好的結果但是 testing 的結果卻比預期的差很多，通常我們把這樣的現象稱作為過度擬合 (Overfitting)。
> 
> 發生這種現象的原因就在於模型過度追求 training 資料的準確性，甚至去 fit 資料的雜訊。
> 
> 雖然我們可以用複雜的曲線去 fit 所有的 training data，但是與平滑的曲線相比卻顯得有些不合理，因為會降低對雜訊的容忍度。
> 
> 除非確定資料沒有任何雜訊，否則要控制模型的複雜度，已達到泛化 (i.e. 去預測沒有看過的資料) 的目的。
> [name=助教]

**抑制模型複雜度的方法：**

- Regression: **weight regularization**
    - e.g. Linear Regression, Logistic Regression, SVM

- Regression, Classifier: **cross validation** 選擇模型參數
    - e.g. SVM, Decision Tree, XGBoost

- 需要逐步更新的模型：**early stop**

    > 一般用在 boosting 系列的模型或神經網路 (e.g. XGBoost, DNN, CNN, ...)

**Early Stop 防止資料過度擬合方法：**

- Validation Set: 

    從 training data 分出 validation set 代替 testing set 來檢視 model 是否過度擬合

- 方法

    Training 時，如果 Training Error 下降但 Validation Error 沒有下降的話即可停止訓練，一般稱為 Early Stop。

- 適用模型種類：

    一般用在 boosting 系列的 model 或 neural network。E.g. XGBoost, DNN, CNN, ...
    
## Regression Models

### Linear Regression

線性迴歸使用的時機：

- `sklearn.LinearRegression`
- Label 為連續值
- 資料量較少 (< 100K)

    > 將所有資料丟進矩陣，計算最小 MSE
    > 資料量一大，就要使用 SGD (Stoichastic Gradient Descent) 方法

- 假設資料的 Features 和 Label 之間有線性關係

    $$ y = a_1 x_1 + a_2 x_2 + a_3 x_3 + \ldots $$
    
可以利用 Features Transformation 提升模型的複雜度

$$ y = a_1 x_1 + a_2 x_2 + a_3 x_3 + a_{11} x_1^2 + a_{12} x_1 x_2 + \ldots $$

### Weight Regularization

### Logistic Regression

### Imbalanced Dataset

- Precision-Recall
    - 適合用在重視 Recall 的偵測問題，例如警察抓壞人，關心的是抓出多少壞人，而不是多少好人。


    | Confusion Matrix | Predict 0           | Predict 1           |
    |------------------|---------------------|---------------------|
    | **Actual 0**         | True Negative (TN)  | <font color="blue" > **False Positive (FP)** <font> |
    | **Actual 1**         | <font style="background-color:#01DF3A;"> False Negative (FN) <font> |  <font color="blue" style="background-color:#01DF3A;"> **True Positive (TP)** <font> |


<span style="color:blue">

$$
Precision = \frac{TP}{TP+FP}
$$

</span>

<span style="color:#01DF3A">

$$
Recall = \frac{TP}{FN+TP}
$$

</span>

- ROC
    - 適合用在分類問題，例如分辨貓狗照片
    - [ROC Curves and Area Under the Curve (AUC) Explained - YouTube](https://www.youtube.com/watch?v=OAl6eAyP-yo&feature=youtu.be)
    - [Precision-Recall AUC vs ROC AUC for class imbalance problems
](https://www.kaggle.com/general/7517)


    | Confusion Matrix | Predict 0           | Predict 1           |
    |------------------|---------------------|---------------------|
    | **Actual 0**         | <font style="background-color:#FF00FF;"> True Negative (TN) <font>  | <font color="blue" style="background-color:#FF00FF;"> **False Positive (FP)** <font> |
    | **Actual 1**         | False Negative (FN) |  <font color="blue"> **True Positive (TP)** <font> |


<span style="color:blue">

$$
TPR(True\ Positive\ Rate)= Precision = \frac{TP}{TP+FP}
$$

</span>

<span style="color:#FF00FF">

$$
FPR(False\ Positive\ Rate) = \frac{FP}{TN+FP}
$$

</span>

## SVM

### SVM Concept

### SVM in Scikit-learn-1

### SVM in Scikit-learn-2

### SVM Short Summary

#### Model Parameters Selection

### Grid Search Cross Validation

---

# Day 2 (02/08)

[課程投影片](https://docs.google.com/presentation/d/1k_nGaBEuxitCcXHTZ_COzwEkHckHoJiu7H1aJCGztN0/edit#slide=id.g2e3e866df6_0_0)

## 決策樹與隨機森林

**決策樹 summary**

- 掃過所有 feature 與對應的值將資料做切分
- 希望資料盡可能分開，透過切分後的資料純度(Gini or Entropy) 來衡量
- 如果不對決策樹進行任何限制 (樹的深度、葉子至少要有多少樣本)，容易造成 Overfitting
- 透過 feature importance 來排序重要性

**決策樹進化! ensemble**

- Bagging (Bootstrap aggregating)
- Boosting

:::info
**Q&A**
* 請問助教如果有兩個 feature A 和 feature B 去切分決策樹 假設 feature A 切分出來的 Gini 是 0.3 和 0.7 然後 feature B 切分出來的 Gini 是 0.4 和 0.6 而不像是投影片 p.13 舉的例子 這樣該取何者 feature 來切分決策樹？

>***Ans:** The features are always randomly permuted at each split. Therefore, the best found split may vary, even with the same training data and max\features=n\features, if the improvement of the criterion is identical for several splits enumerated during the search of the best split. To obtain a deterministic behaviour during fitting, randomstate has to be fixed. 翻譯蒟蒻：每次 split 時， features 會隨機排序，所以如果遇到有 information gain 一樣時，每次重 train 的結果就可能會不同 (因為選的 features 有可能不一樣)*
> [name=助教]

* 如果大家對 entropy, cross-entropy (log-loss) 的概念還不熟悉，推薦花個十分鐘看這個影片。這個概念之後 deep learning 會很常用到 [https://www.youtube.com/watch?v=ErfnhcEV1O8](https://www.youtube.com/watch?v=ErfnhcEV1O8)

* 請問為什麼entropy一定會在0~1之間? 資料越亂時entropy會超過1 ?

>***Ans:可以參考** [https://stats.stackexchange.com/questions/95261/why-am-i-getting-information-entropy-greater-than-1](https://stats.stackexchange.com/questions/95261/why-am-i-getting-information-entropy-greater-than-1)*


* 我把**回歸**決策樹 (regression tree) 的原理列在思考問題，簡報 p. 27  
這問題就是希望大家能夠體會，混亂程度指標用在分類問題很合理，那如果變成回歸問題呢?  
其實答案在 [Scikit-learn Regressor 的網頁](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor)上就能找到:

>The function to measure the quality of a split. Supported criteria are “mse” for the mean squared error, which is equal to variance reduction as feature selection criterion and minimizes the L2 loss using the mean of each terminal node

>翻譯蒟蒻: 用每個節點上的所有資料 (y) 計算平均數，之後每個資料對平均數計算 L2 loss，找到 loss 最小的 feature 用來分支。其實本質上跟分類並沒有太大的區別，都是希望切分之後的資料能夠越像越好，希望可以解答你的問題:)
> [name=助教]
:::

### GBM與XGBoost
**Boosting? Gradient?**
* 前面我們學到的方法稱為 **Bagging** (Bootstrap aggregating)，用抽樣的資料與 features 生成每一棵樹(樹之間是獨立的)，**最後再取平均**

* **Boosting** 則是希望能夠由後面生成的樹，來**修正前面樹學的不好的地方**
 
* 要怎麼修正前面學錯的地方呢？計算 Gradient! (先想像 Gradient 就是一個能教我們修正錯誤的東西）

**Bagging vs. Boosting**

**What’s XGBoost?**

**XGBoost vs. GBM**


[XGBoost 詳解](http://www.52cs.org/?p=429) \- 中文
[XGBoost parameter tuning](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)  \- 英文

:::info
> [name=助教 Jimmy]
Hi 大家！機器學習2 的講義中我有列出了許多思考問題，主要是希望大家能夠去思考背後的原理，這邊提供參考答案給大家，若覺得有什麼問題也歡迎在底下討論喔！

* 決策樹：

1.  **在分類問題中，若沒有任何限制，決策樹有辦法把訓練資料的 loss 完全降成 0 嗎？**  
    ans: 若資料中的 x -> y 不是一對多的情況 (同樣的 x 卻對應到不同的 y)。則決策樹其實是可以生成各種規則，想辦法把一個 x mapping 到一個 y，這種情況下，就可以把 trainling loss 降成 0
2.  **決策樹做回歸問題時，資料不純度如何計算？樹建置好後如何預測？**  
    ans: 這個問題在[討論區中已經有解答過](https://discuss.aiacademy.tw/t/topic/113)

* 隨機森林：

1.  **RadomForest 中的每一棵樹，是希望能夠盡量生長還是限制生長？**  
    ans: RF 會希望每棵樹盡量 overfitting，最後再透過投票的方式緩解 overfitting 的情形
2.  **每棵樹用取後放回的抽樣，請問這棵樹大約使用了多少 % 的 unique 原資料生成？**  
    ans: 當 n 很大時，可以推導抽到不重複資料的比率約為 63.2 %，數學可參考[連結1](https://math.stackexchange.com/questions/41519/expected-number-of-unique-items-when-drawing-with-replacement)

* XGBoost：

1.  **同樣的 dataset 若存在兩個完全一模一樣的 feature 這兩個 feature 的 importance，在 XGBoost 與 RandomForest 的模型結果中會一樣嗎?**  
    ans: 不會。XGB 只會選其中一個，RF 的 feature importance 很有可能被均分
2.  **XGBoost 中，row\_sample 代表對資料筆樹抽樣 (row)，col\_sample 代表對 features 抽樣，若這兩個都設置成 1 (代表不抽樣，全部使用)，每次訓練後的樹會長的一模一樣嗎?**  
    ans: 會。除非有兩個 feature 算出來的 information gain 一模一樣，不然沒有隨機條件，每次生成的樹都會一模一樣
:::

### 非監督式學習

**主成份分析 (Principal Componet Analysis, PCA)**

**階層式分析**

### Kaggle 實戰

**資料分析競賽的流程**  

* 資料清洗與轉換
    
* 探索式資料分析 (EDA)
    
* 特徵工程 (feautre engineering)
    
* 建立模型
    
* 調整參數
    
* 上傳結果


**Machine Learning Algorithms cheat sheet**
![](https://scontent.ftpe8-4.fna.fbcdn.net/v/t1.0-9/27655285_2158923940785224_5037431082928500792_n.jpg?oh=7ed41404c289ca635aee27fa0a021e70&oe=5B18C468)

**[at071054 Kent 分享]**
Data Science London + Scikit-learn 那題極致的解法 :
http://nbviewer.jupyter.org/gist/luanjunyi/6632d4c0f92bc30750f4


**[at071070 / Yvonne Wu 分享]**
我目前還處於依樣畫葫蘆階段，無法從無到有完成Kaggle題目，都是在Kernels中找排前面又比較簡單的來抄。我參考的sample如下： [https://www.kaggle.com/dansbecker/submitting-from-a-kernel](https://www.kaggle.com/dansbecker/submitting-from-a-kernel)   [https://www.kaggle.com/c/data-science-london-scikit-learn/discussion/34115](https://www.kaggle.com/c/data-science-london-scikit-learn/discussion/34115) 

```python=
import numpy as np
#import sklearn as sk
#import matplotlib.pyplot as plt
import pandas as pd

#from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import Perceptron
#from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import VotingClassifier
#from sklearn import svm
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
plt.rcParams['font.family']='SimHei' #顯示中文

%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

#### READING OUR GIVEN DATA INTO PANDAS DATAFRAME ####
x_train = pd.read_csv('train.csv',header=None)
y_train = pd.read_csv('trainLabels.csv',header=None)
x_test = pd.read_csv('test.csv',header=None)
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_train = y_train.ravel()
print('training_x Shape:',x_train.shape,
      ',training_y Shape:',y_train.shape, 
      ',testing_x Shape:',x_test.shape )
print('training_x len:',len(x_train),
      ',training_y len:',len(y_train), 
      ',testing_x len:',len(x_test) )

#Checking the models
x_all = np.r_[x_train,x_test]
print('x_all shape :',x_all.shape)

#### USING THE GAUSSIAN MIXTURE MODEL ####
from sklearn.mixture import GaussianMixture
lowest_bic = np.infty
bic = []
n_components_range = range(1, 7)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
    # Fit a mixture of Gaussians with EM
        gmm = GaussianMixture(n_components=n_components,covariance_type=cv_type)
        gmm.fit(x_all)
        bic.append(gmm.aic(x_all))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

best_gmm.fit(x_all)
x_train = best_gmm.predict_proba(x_train)
x_test = best_gmm.predict_proba(x_test)

#### TAKING ONLY TWO MODELS FOR KEEPING IT SIMPLE ####
knn = KNeighborsClassifier()
rf = RandomForestClassifier()

param_grid = dict( )
#### GRID SEARCH for BEST TUNING PARAMETERS FOR KNN #####
grid_search_knn = GridSearchCV(knn,param_grid=param_grid,cv=10,scoring='accuracy').fit(x_train,y_train)
print('best estimator KNN:',grid_search_knn.best_estimator_,
      'Best Score', grid_search_knn.best_estimator_.score(x_train,y_train))
knn_best = grid_search_knn.best_estimator_

#### GRID SEARCH for BEST TUNING PARAMETERS FOR RandomForest #####
grid_search_rf = GridSearchCV(rf, param_grid=dict( ), verbose=3,scoring='accuracy',cv=10).fit(x_train,y_train)
print('best estimator RandomForest:',grid_search_rf.best_estimator_,
      'Best Score', grid_search_rf.best_estimator_.score(x_train,y_train))
rf_best = grid_search_rf.best_estimator_

knn_best.fit(x_train,y_train)
print("KNN:",knn_best.predict(x_test)[0:10])
rf_best.fit(x_train,y_train)
print("RF:",rf_best.predict(x_test)[0:10])

#### SCORING THE MODELS ####
print('Score for KNN :',cross_val_score(knn_best,x_train,y_train,cv=10,scoring='accuracy').mean())
print('Score for Random Forest :',cross_val_score(rf_best,x_train,y_train,cv=10,scoring='accuracy').max())

### IN CASE WE WERE USING MORE THAN ONE CLASSIFIERS THEN VOTING CLASSIFIER CAN BE USEFUL ###
#clf = VotingClassifier(
#		estimators=[('knn_best',knn_best),('rf_best',rf_best)],
#		#weights=[871856020222,0.907895269918]
#	)
#clf.fit(x_train,y_train)
#print("Votting:",clf.predict(x_test)[0:10])

##### FRAMING OUR SOLUTION #####
knn_best_pred = knn_best.predict(x_test)
rf_best_pred = rf_best.predict(x_test)
#voting_clf_pred = pd.DataFrame(clf.predict(x_test))


row_id = [i for i in range(1,9001)]

# Generate Submission File 
StackingSubmission = pd.DataFrame({ 'Id': row_id, 'Solution': rf_best_pred })[["Id","Solution"]]
StackingSubmission.to_csv("submission_your_file_name.csv", index=False)

```

