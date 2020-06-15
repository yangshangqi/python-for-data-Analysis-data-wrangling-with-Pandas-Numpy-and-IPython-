## 一.pandas与建模代码的结合

pandas与其它分析库通常是靠NumPy的数组联系起来的。将DataFrame转换为NumPy数组，可以使用.values属性：


```python
import pandas as pd
import numpy as np
data = pd.DataFrame({'x0': [1, 2, 3, 4, 5],
                     'x1': [0.01, -0.01, 0.25, -4.1, 0.],
                     'y': [-1.5, 0., 3.6, 1.3, -2.]})
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x0</th>
      <th>x1</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>0.01</td>
      <td>-1.5</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>-0.01</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>0.25</td>
      <td>3.6</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>-4.10</td>
      <td>1.3</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>0.00</td>
      <td>-2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.columns
```




    Index(['x0', 'x1', 'y'], dtype='object')




```python
data.values
```




    array([[ 1.  ,  0.01, -1.5 ],
           [ 2.  , -0.01,  0.  ],
           [ 3.  ,  0.25,  3.6 ],
           [ 4.  , -4.1 ,  1.3 ],
           [ 5.  ,  0.  , -2.  ]])



要转换回DataFrame，可以传递一个二维ndarray，可带有列名：


```python
df2 = pd.DataFrame(data.values, columns=['one', 'two', 'three'])
df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.0</td>
      <td>0.01</td>
      <td>-1.5</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2.0</td>
      <td>-0.01</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3.0</td>
      <td>0.25</td>
      <td>3.6</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4.0</td>
      <td>-4.10</td>
      <td>1.3</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5.0</td>
      <td>0.00</td>
      <td>-2.0</td>
    </tr>
  </tbody>
</table>
</div>



一般当数据是同构化的时候使用.values属性。例如，全是数字类型。如果数据是异构化的，结果会是Python对象的ndarray：


```python
df3 = data.copy()
df3['strings'] = ['a', 'b', 'c', 'd', 'e']
df3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x0</th>
      <th>x1</th>
      <th>y</th>
      <th>strings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>0.01</td>
      <td>-1.5</td>
      <td>a</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>-0.01</td>
      <td>0.0</td>
      <td>b</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>0.25</td>
      <td>3.6</td>
      <td>c</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>-4.10</td>
      <td>1.3</td>
      <td>d</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>0.00</td>
      <td>-2.0</td>
      <td>e</td>
    </tr>
  </tbody>
</table>
</div>




```python
df3.values
```




    array([[1, 0.01, -1.5, 'a'],
           [2, -0.01, 0.0, 'b'],
           [3, 0.25, 3.6, 'c'],
           [4, -4.1, 1.3, 'd'],
           [5, 0.0, -2.0, 'e']], dtype=object)



对于一些模型，你可能只想使用列的子集。我建议你使用loc和values作索引：


```python
model_cols = ['x0', 'x1']
data.loc[:, model_cols].values
```




    array([[ 1.  ,  0.01],
           [ 2.  , -0.01],
           [ 3.  ,  0.25],
           [ 4.  , -4.1 ],
           [ 5.  ,  0.  ]])



一些库对pandas有本地化支持，会自动完成工作：从DataFrame转换到NumPy中并将模型的参数名添加到输出表的列或Series。其它情况，你可以手工进行“元数据管理”。

之前我们学习了pandas的Categorical类型和pandas.get_dummies函数。假设数据集中有一个非数值列：


```python
data['category'] = pd.Categorical(['a', 'b', 'a', 'a', 'b'],
                                  categories=['a', 'b'])
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x0</th>
      <th>x1</th>
      <th>y</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>0.01</td>
      <td>-1.5</td>
      <td>a</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>-0.01</td>
      <td>0.0</td>
      <td>b</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>0.25</td>
      <td>3.6</td>
      <td>a</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>-4.10</td>
      <td>1.3</td>
      <td>a</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>0.00</td>
      <td>-2.0</td>
      <td>b</td>
    </tr>
  </tbody>
</table>
</div>



如果我们想替换category列为虚变量，我们可以创建虚变量，删除category列，然后添加到结果：


```python
dummies = pd.get_dummies(data.category, prefix='category')
data_with_dummies = data.drop('category', axis=1).join(dummies)
data_with_dummies
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x0</th>
      <th>x1</th>
      <th>y</th>
      <th>category_a</th>
      <th>category_b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>0.01</td>
      <td>-1.5</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>-0.01</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>0.25</td>
      <td>3.6</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>-4.10</td>
      <td>1.3</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>0.00</td>
      <td>-2.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



用虚变量拟合某些统计模型会有一些细微差别。当你不只有数字列时，使用Patsy（下一节的主题）可能更简单，更不容易出错。

## 二.使用patsy创建模型描述

Patsy能够很好的支持statsmodels中特定的线性模型，因此我会关注于它的主要特点，让你尽快掌握。Patsy的公式是一个特殊的字符串语法，如下所示：

y ~ x0 + x1

a+b不是将a与b相加的意思，而是为模型创建的设计矩阵。patsy.dmatrices函数接收一个公式字符串和一个数据集（可以是DataFrame或数组的字典），为线性模型创建设计矩阵：


```python
data = pd.DataFrame({'x0': [1, 2, 3, 4, 5],
                     'x1': [0.01, -0.01, 0.25, -4.1, 0.],
                     'y': [-1.5, 0., 3.6, 1.3, -2.]})
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x0</th>
      <th>x1</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>0.01</td>
      <td>-1.5</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>-0.01</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>0.25</td>
      <td>3.6</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>-4.10</td>
      <td>1.3</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>0.00</td>
      <td>-2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
import patsy
y, X = patsy.dmatrices('y ~ x0 + x1', data)
y
```




    DesignMatrix with shape (5, 1)
         y
      -1.5
       0.0
       3.6
       1.3
      -2.0
      Terms:
        'y' (column 0)




```python
X
```




    DesignMatrix with shape (5, 3)
      Intercept  x0     x1
              1   1   0.01
              1   2  -0.01
              1   3   0.25
              1   4  -4.10
              1   5   0.00
      Terms:
        'Intercept' (column 0)
        'x0' (column 1)
        'x1' (column 2)



这些Patsy的DesignMatrix实例是NumPy的ndarray，带有附加元数据：


```python
np.asarray(y)
```




    array([[-1.5],
           [ 0. ],
           [ 3.6],
           [ 1.3],
           [-2. ]])




```python
np.asarray(X)
```




    array([[ 1.  ,  1.  ,  0.01],
           [ 1.  ,  2.  , -0.01],
           [ 1.  ,  3.  ,  0.25],
           [ 1.  ,  4.  , -4.1 ],
           [ 1.  ,  5.  ,  0.  ]])



你可能想Intercept(截距)这个名词列是哪里来的。这是线性模型（比如普通最小二乘回归）的惯例用法。可以通过给模型添加名词列 +0来不显示截距：


```python
patsy.dmatrices('y ~ x0 + x1 + 0', data)[1]
```




    DesignMatrix with shape (5, 2)
      x0     x1
       1   0.01
       2  -0.01
       3   0.25
       4  -4.10
       5   0.00
      Terms:
        'x0' (column 0)
        'x1' (column 1)



Patsy对象可以直接传递一些算法，比如numpy.linalg.lstsq等，这些算法都会执行一个最小二乘回归：


```python
coef, resid, _, _ = np.linalg.lstsq(X, y)
```

    C:\Users\Administrator\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: FutureWarning: `rcond` parameter will change to the default of machine precision times ``max(M, N)`` where M and N are the input matrix dimensions.
    To use the future default and silence this warning we advise to pass `rcond=None`, to keep using the old, explicitly pass `rcond=-1`.
      """Entry point for launching an IPython kernel.
    

模型的元数据保留在design_info属性中，因此你可以重新附加列名到拟合系数，以获得一个Series，例如：


```python
coef
```




    array([[ 0.31290976],
           [-0.07910564],
           [-0.26546384]])




```python
coef = pd.Series(coef.squeeze(), index=X.design_info.column_names)
coef
```




    Intercept    0.312910
    x0          -0.079106
    x1          -0.265464
    dtype: float64



### 2.1Patsy公式中的数据转换

你可以将Python代码与patsy公式结合。在执行公式时，库将尝试查找在封闭作用域内使用的函数：


```python
y, X = patsy.dmatrices('y ~ x0 + np.log(np.abs(x1) + 1)', data)
X
```




    DesignMatrix with shape (5, 3)
      Intercept  x0  np.log(np.abs(x1) + 1)
              1   1                 0.00995
              1   2                 0.00995
              1   3                 0.22314
              1   4                 1.62924
              1   5                 0.00000
      Terms:
        'Intercept' (column 0)
        'x0' (column 1)
        'np.log(np.abs(x1) + 1)' (column 2)



常见的变量转换包括标准化（平均值为0，方差为1）和中心化（减去平均值）。Patsy有内置的函数进行这样的工作：


```python
y, X = patsy.dmatrices('y ~ standardize(x0) + center(x1)', data)
X
```




    DesignMatrix with shape (5, 3)
      Intercept  standardize(x0)  center(x1)
              1         -1.41421        0.78
              1         -0.70711        0.76
              1          0.00000        1.02
              1          0.70711       -3.33
              1          1.41421        0.77
      Terms:
        'Intercept' (column 0)
        'standardize(x0)' (column 1)
        'center(x1)' (column 2)



patsy.build_design_matrices函数可以使用原始样本数据集的保存信息，来转换新数据：


```python
new_data = pd.DataFrame({'x0': [6, 7, 8, 9],
                         'x1': [3.1, -0.5, 0, 2.3],
                         'y': [1, 2, 3, 4]})
new_X = patsy.build_design_matrices([X.design_info], new_data)
new_X
```




    [DesignMatrix with shape (4, 3)
       Intercept  standardize(x0)  center(x1)
               1          2.12132        3.87
               1          2.82843        0.27
               1          3.53553        0.77
               1          4.24264        3.07
       Terms:
         'Intercept' (column 0)
         'standardize(x0)' (column 1)
         'center(x1)' (column 2)]



因为Patsy中的加号不是加法的意义，当你按照名称将数据集的列相加时，你必须用特殊I函数将它们封装起来：


```python
y, X = patsy.dmatrices('y ~ I(x0 + x1)', data)
X
```




    DesignMatrix with shape (5, 2)
      Intercept  I(x0 + x1)
              1        1.01
              1        1.99
              1        3.25
              1       -0.10
              1        5.00
      Terms:
        'Intercept' (column 0)
        'I(x0 + x1)' (column 1)



### 2.2分类数据与Patsy

非数值数据可以用多种方式转换为模型设计矩阵。完整的讲解超出了本书范围，最好和统计课一起学习。

当你在Patsy公式中使用非数值数据，它们会默认转换为虚变量。如果有截距，会去掉一个，避免共线性：


```python
data = pd.DataFrame({
                 'key1': ['a', 'a', 'b', 'b', 'a', 'b', 'a', 'b'],
                 'key2': [0, 1, 0, 1, 0, 1, 0, 0],
                 'v1': [1, 2, 3, 4, 5, 6, 7, 8],
                 'v2': [-1, 0, 2.5, -0.5, 4.0, -1.2, 0.2, -1.7]})
y, X = patsy.dmatrices('v2 ~ key1', data)
X
```




    DesignMatrix with shape (8, 2)
      Intercept  key1[T.b]
              1          0
              1          0
              1          1
              1          1
              1          0
              1          1
              1          0
              1          1
      Terms:
        'Intercept' (column 0)
        'key1' (column 1)



如果你从模型中忽略截距，每个分类值的列都会包括在设计矩阵的模型中：


```python
y, X = patsy.dmatrices('v2 ~ key1 + 0', data)
X
```




    DesignMatrix with shape (8, 2)
      key1[a]  key1[b]
            1        0
            1        0
            0        1
            0        1
            1        0
            0        1
            1        0
            0        1
      Terms:
        'key1' (columns 0:2)



数字类型列可以使用C函数解释为分类函数：


```python
y, X = patsy.dmatrices('v2 ~ C(key2)', data)
X
```




    DesignMatrix with shape (8, 2)
      Intercept  C(key2)[T.1]
              1             0
              1             1
              1             0
              1             1
              1             0
              1             1
              1             0
              1             0
      Terms:
        'Intercept' (column 0)
        'C(key2)' (column 1)



当你在模型中使用多个分类名，事情就会变复杂，因为会包括key1:key2形式的相交部分，它可以用在方差（ANOVA）模型分析中：


```python
data['key2'] = data['key2'].map({0: 'zero', 1: 'one'})
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key1</th>
      <th>key2</th>
      <th>v1</th>
      <th>v2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>a</td>
      <td>zero</td>
      <td>1</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>a</td>
      <td>one</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>b</td>
      <td>zero</td>
      <td>3</td>
      <td>2.5</td>
    </tr>
    <tr>
      <td>3</td>
      <td>b</td>
      <td>one</td>
      <td>4</td>
      <td>-0.5</td>
    </tr>
    <tr>
      <td>4</td>
      <td>a</td>
      <td>zero</td>
      <td>5</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>b</td>
      <td>one</td>
      <td>6</td>
      <td>-1.2</td>
    </tr>
    <tr>
      <td>6</td>
      <td>a</td>
      <td>zero</td>
      <td>7</td>
      <td>0.2</td>
    </tr>
    <tr>
      <td>7</td>
      <td>b</td>
      <td>zero</td>
      <td>8</td>
      <td>-1.7</td>
    </tr>
  </tbody>
</table>
</div>




```python
y, X = patsy.dmatrices('v2 ~ key1 + key2', data)
X
```




    DesignMatrix with shape (8, 3)
      Intercept  key1[T.b]  key2[T.zero]
              1          0             1
              1          0             0
              1          1             1
              1          1             0
              1          0             1
              1          1             0
              1          0             1
              1          1             1
      Terms:
        'Intercept' (column 0)
        'key1' (column 1)
        'key2' (column 2)




```python
y, X = patsy.dmatrices('v2 ~ key1 + key2 + key1:key2', data)
X
```




    DesignMatrix with shape (8, 4)
      Intercept  key1[T.b]  key2[T.zero]  key1[T.b]:key2[T.zero]
              1          0             1                       0
              1          0             0                       0
              1          1             1                       1
              1          1             0                       0
              1          0             1                       0
              1          1             0                       0
              1          0             1                       0
              1          1             1                       1
      Terms:
        'Intercept' (column 0)
        'key1' (column 1)
        'key2' (column 2)
        'key1:key2' (column 3)



## 三.statsmodels介绍

statsmodels是Python进行拟合多种统计模型、进行统计试验和数据探索可视化的库。Statsmodels包含许多经典的统计方法，但没有贝叶斯方法和机器学习模型。

### 3.1评估线性模型

statsmodels有多种线性回归模型，包括从基本（比如普通最小二乘）到复杂（比如迭代加权最小二乘法）的。

statsmodels的线性模型有两种不同的接口：基于数组和基于公式。它们可以通过API模块引入：


```python
import statsmodels.api as sm
import statsmodels.formula.api as smf
```

为了展示它们的使用方法，我们从一些随机数据生成一个线性模型：


```python
def dnorm(mean, variance, size=1):
    if isinstance(size, int):
        size = size,
    return mean + np.sqrt(variance) * np.random.randn(*size)
np.random.seed(12345)
N = 100
X = np.c_[dnorm(0, 0.4, size=N),
          dnorm(0, 0.6, size=N),
          dnorm(0, 0.2, size=N)]
eps = dnorm(0, 0.1, size=N)
beta = [0.1, 0.3, 0.5]

y = np.dot(X, beta) + eps
```

这里，我使用了“真实”模型和可知参数beta。此时，dnorm可用来生成正态分布数据，带有特定均值和方差。现在有：


```python
X[:5]
```




    array([[-0.12946849, -1.21275292,  0.50422488],
           [ 0.30291036, -0.43574176, -0.25417986],
           [-0.32852189, -0.02530153,  0.13835097],
           [-0.35147471, -0.71960511, -0.25821463],
           [ 1.2432688 , -0.37379916, -0.52262905]])



像之前Patsy看到的，线性模型通常要拟合一个截距。sm.add_constant函数可以添加一个截距的列到现存的矩阵：


```python
X_model = sm.add_constant(X)
X_model[:5]
```




    array([[ 1.        , -0.12946849, -1.21275292,  0.50422488],
           [ 1.        ,  0.30291036, -0.43574176, -0.25417986],
           [ 1.        , -0.32852189, -0.02530153,  0.13835097],
           [ 1.        , -0.35147471, -0.71960511, -0.25821463],
           [ 1.        ,  1.2432688 , -0.37379916, -0.52262905]])



sm.OLS类可以拟合一个普通最小二乘回归：


```python
model = sm.OLS(y, X)
```

这个模型的fit方法返回了一个回归结果对象，它包含估计的模型参数和其它内容：


```python
results = model.fit()
results.params
```




    array([0.17826108, 0.22303962, 0.50095093])



对结果使用summary方法可以打印模型的详细诊断结果：


```python
print(results.summary())
```

                                     OLS Regression Results                                
    =======================================================================================
    Dep. Variable:                      y   R-squared (uncentered):                   0.430
    Model:                            OLS   Adj. R-squared (uncentered):              0.413
    Method:                 Least Squares   F-statistic:                              24.42
    Date:                Sun, 14 Jun 2020   Prob (F-statistic):                    7.44e-12
    Time:                        10:04:35   Log-Likelihood:                         -34.305
    No. Observations:                 100   AIC:                                      74.61
    Df Residuals:                      97   BIC:                                      82.42
    Df Model:                           3                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    x1             0.1783      0.053      3.364      0.001       0.073       0.283
    x2             0.2230      0.046      4.818      0.000       0.131       0.315
    x3             0.5010      0.080      6.237      0.000       0.342       0.660
    ==============================================================================
    Omnibus:                        4.662   Durbin-Watson:                   2.201
    Prob(Omnibus):                  0.097   Jarque-Bera (JB):                4.098
    Skew:                           0.481   Prob(JB):                        0.129
    Kurtosis:                       3.243   Cond. No.                         1.74
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    

这里的参数名为通用名x1, x2等等。假设所有的模型参数都在一个DataFrame中：


```python
data = pd.DataFrame(X, columns=['col0', 'col1', 'col2'])
data['y'] = y
data[:5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col0</th>
      <th>col1</th>
      <th>col2</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>-0.129468</td>
      <td>-1.212753</td>
      <td>0.504225</td>
      <td>0.427863</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.302910</td>
      <td>-0.435742</td>
      <td>-0.254180</td>
      <td>-0.673480</td>
    </tr>
    <tr>
      <td>2</td>
      <td>-0.328522</td>
      <td>-0.025302</td>
      <td>0.138351</td>
      <td>-0.090878</td>
    </tr>
    <tr>
      <td>3</td>
      <td>-0.351475</td>
      <td>-0.719605</td>
      <td>-0.258215</td>
      <td>-0.489494</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1.243269</td>
      <td>-0.373799</td>
      <td>-0.522629</td>
      <td>-0.128941</td>
    </tr>
  </tbody>
</table>
</div>



现在，我们使用statsmodels的公式API和Patsy的公式字符串：


```python
results = smf.ols('y ~ col0 + col1 + col2', data=data).fit()
results.params
```




    Intercept    0.033559
    col0         0.176149
    col1         0.224826
    col2         0.514808
    dtype: float64




```python
results.tvalues
```




    Intercept    0.952188
    col0         3.319754
    col1         4.850730
    col2         6.303971
    dtype: float64



观察下statsmodels是如何返回Series结果的，附带有DataFrame的列名。当使用公式和pandas对象时，我们不需要使用add_constant。

给出一个样本外数据，你可以根据估计的模型参数计算预测值：


```python
results.predict(data[:5])
```




    0   -0.002327
    1   -0.141904
    2    0.041226
    3   -0.323070
    4   -0.100535
    dtype: float64



### 3.2评估时间序列处理

statsmodels的另一模型类是进行时间序列分析，包括自回归过程、卡尔曼滤波和其它态空间模型，和多元自回归模型。

用自回归结构和噪声来模拟一些时间序列数据：


```python
init_x = 4

import random
values = [init_x, init_x]
N = 1000

b0 = 0.8
b1 = -0.4
noise = dnorm(0, 0.1, N)
for i in range(N):
    new_x = values[-1] * b0 + values[-2] * b1 + noise[i]
    values.append(new_x)
```

这个数据有AR(2)结构（两个延迟），参数是0.8和-0.4。拟合AR模型时，你可能不知道滞后项的个数，因此可以用更大的滞后数来拟合这个模型：


```python
MAXLAGS = 5
model = sm.tsa.AR(values)
results = model.fit(MAXLAGS)
```

结果中的估计参数首先是截距，其次是前两个参数的估计值：


```python
results.params
```




    array([-0.00616093,  0.78446347, -0.40847891, -0.01364148,  0.01496872,
            0.01429462])


