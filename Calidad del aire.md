

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
ts = pd.read_csv("calidad_del_aire_cyl_1997_2013.csv",
                sep = ";",
                parse_dates = [0],
                index_col = 0)
```

    /home/luis/anaconda3/lib/python3.7/site-packages/IPython/core/interactiveshell.py:2785: DtypeWarning: Columns (6,8) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)



```python
ts.head()
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
      <th>CO (mg/m3)</th>
      <th>NO (ug/m3)</th>
      <th>NO2 (ug/m3)</th>
      <th>O3 (ug/m3)</th>
      <th>PM10 (ug/m3)</th>
      <th>SH2 (ug/m3)</th>
      <th>PM25 (ug/m3)</th>
      <th>PST (ug/m3)</th>
      <th>SO2 (ug/m3)</th>
      <th>PROVINCIA</th>
      <th>ESTACIÓN</th>
    </tr>
    <tr>
      <th>DIA</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1997-01-01</th>
      <td>1.2</td>
      <td>12</td>
      <td>33</td>
      <td>63</td>
      <td>56</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>19</td>
      <td>ÁVILA</td>
      <td>Ávila</td>
    </tr>
    <tr>
      <th>1997-02-01</th>
      <td>1.3</td>
      <td>15</td>
      <td>35</td>
      <td>59</td>
      <td>47</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17</td>
      <td>ÁVILA</td>
      <td>Ávila</td>
    </tr>
    <tr>
      <th>1997-03-01</th>
      <td>1.5</td>
      <td>18</td>
      <td>43</td>
      <td>54</td>
      <td>65</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>19</td>
      <td>ÁVILA</td>
      <td>Ávila</td>
    </tr>
    <tr>
      <th>1997-04-01</th>
      <td>1.6</td>
      <td>56</td>
      <td>73</td>
      <td>50</td>
      <td>74</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22</td>
      <td>ÁVILA</td>
      <td>Ávila</td>
    </tr>
    <tr>
      <th>1997-05-01</th>
      <td>1.4</td>
      <td>11</td>
      <td>33</td>
      <td>63</td>
      <td>54</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18</td>
      <td>ÁVILA</td>
      <td>Ávila</td>
    </tr>
  </tbody>
</table>
</div>




```python
ts.shape
```




    (153120, 11)




```python
ts = ts.drop(ts[ts["CO (mg/m3)"] == "No cumple el anexo IV de la Decisión del Consejo 97/101/CE"].index)
ts = ts.drop(ts[ts["NO (ug/m3)"] == "No cumple el anexo IV de la Decisión del Consejo 97/101/CE"].index)
ts = ts.drop(ts[ts["NO2 (ug/m3)"] == "No cumple el anexo IV de la Decisión del Consejo 97/101/CE"].index)
ts = ts.drop(ts[ts["O3 (ug/m3)"] == "No cumple el anexo IV de la Decisión del Consejo 97/101/CE"].index)
ts = ts.drop(ts[ts["PM10 (ug/m3)"] == "No cumple el anexo IV de la Decisión del Consejo 97/101/CE"].index)
ts = ts.drop(ts[ts["SH2 (ug/m3)"] == "No cumple el anexo IV de la Decisión del Consejo 97/101/CE"].index)
ts = ts.drop(ts[ts["PM25 (ug/m3)"] == "No cumple el anexo IV de la Decisión del Consejo 97/101/CE"].index)
ts = ts.drop(ts[ts["PST (ug/m3)"] == "No cumple el anexo IV de la Decisión del Consejo 97/101/CE"].index)
ts = ts.drop(ts[ts["SO2 (ug/m3)"] == "No cumple el anexo IV de la Decisión del Consejo 97/101/CE"].index)
ts = ts.astype({"CO (mg/m3)": float})
ts = ts.astype({"NO (ug/m3)": float})
ts = ts.astype({"NO2 (ug/m3)": float})
ts = ts.astype({"O3 (ug/m3)": float})
ts = ts.astype({"PM10 (ug/m3)": float})
ts = ts.astype({"SH2 (ug/m3)": float})
ts = ts.astype({"PM25 (ug/m3)": float})
ts = ts.astype({"PST (ug/m3)": float})
ts = ts.astype({"SO2 (ug/m3)": float})
ts.shape
```




    (51198, 11)




```python
ts.dtypes
```




    CO (mg/m3)      float64
    NO (ug/m3)      float64
    NO2 (ug/m3)     float64
    O3 (ug/m3)      float64
    PM10 (ug/m3)    float64
    SH2 (ug/m3)     float64
    PM25 (ug/m3)    float64
    PST (ug/m3)     float64
    SO2 (ug/m3)     float64
    PROVINCIA        object
    ESTACIÓN         object
    dtype: object




```python
ts.isnull().values.any()
```




    True




```python
ts = ts.apply(lambda x: x.fillna(0))
ts.isnull().values.any()
```




    False




```python
ts.head()
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
      <th>CO (mg/m3)</th>
      <th>NO (ug/m3)</th>
      <th>NO2 (ug/m3)</th>
      <th>O3 (ug/m3)</th>
      <th>PM10 (ug/m3)</th>
      <th>SH2 (ug/m3)</th>
      <th>PM25 (ug/m3)</th>
      <th>PST (ug/m3)</th>
      <th>SO2 (ug/m3)</th>
      <th>PROVINCIA</th>
      <th>ESTACIÓN</th>
    </tr>
    <tr>
      <th>DIA</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1997-01-01</th>
      <td>1.2</td>
      <td>12.0</td>
      <td>33.0</td>
      <td>63.0</td>
      <td>56.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>ÁVILA</td>
      <td>Ávila</td>
    </tr>
    <tr>
      <th>1997-04-01</th>
      <td>1.6</td>
      <td>56.0</td>
      <td>73.0</td>
      <td>50.0</td>
      <td>74.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>22.0</td>
      <td>ÁVILA</td>
      <td>Ávila</td>
    </tr>
    <tr>
      <th>1997-06-01</th>
      <td>1.6</td>
      <td>28.0</td>
      <td>46.0</td>
      <td>56.0</td>
      <td>60.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>20.0</td>
      <td>ÁVILA</td>
      <td>Ávila</td>
    </tr>
    <tr>
      <th>1997-10-01</th>
      <td>1.3</td>
      <td>35.0</td>
      <td>53.0</td>
      <td>38.0</td>
      <td>61.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>18.0</td>
      <td>ÁVILA</td>
      <td>Ávila</td>
    </tr>
    <tr>
      <th>1997-11-01</th>
      <td>1.3</td>
      <td>27.0</td>
      <td>40.0</td>
      <td>25.0</td>
      <td>59.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>16.0</td>
      <td>ÁVILA</td>
      <td>Ávila</td>
    </tr>
  </tbody>
</table>
</div>




```python
ts.PROVINCIA.unique()
```




    array(['ÁVILA', 'BURGOS', 'LEÓN', 'PALENCIA', 'SALAMANCA', 'SEGOVIA',
           'SORIA', 'VALLADOLID', 'ZAMORA'], dtype=object)




```python
ts.ESTACIÓN.unique()
```




    array(['Ávila', 'Aranda de Duero', 'Aranda de Duero 2', 'Burgos 1',
           'Burgos 2', 'Burgos 3', 'Burgos 4', 'Burgos 5', 'Medina de Pomar',
           'Miranda de Ebro 1', 'Miranda de Ebro 2', 'Miranda de Ebro 3',
           'La Robla', 'Lario', 'Le�n 1', 'Le�n 2', 'Le�n 3', 'Le�n 4',
           'Ponferrada 1', 'Ponferrada 2', 'Ponferrada 3', 'Ponferrada 4',
           'Ponferrada 5', 'Guardo', 'Palencia 1', 'Palencia 2', 'Palencia 3',
           'Velilla del R�o Carri�n', 'Venta de Ba�os', 'El Maillo',
           'Salamanca 1', 'Salamanca 2', 'Salamanca 3', 'Salamanca 4',
           'Salamanca 5', 'Salamanca 6', 'Segovia 1', 'Segovia 2',
           'Muriel de la Fuente', 'Soria', 'Medina del Campo', 'Zamora 1',
           'Zamora 2', 'Avila II', 'Burgos1', 'Burgos4', 'Burgos5',
           'Miranda de Ebro1', 'Miranda de Ebro2', 'Le�n1', 'Leon3', 'Leon 4',
           'Ponferrada4', 'Salamanca4', 'Salamanca5', 'Salamanca6'],
          dtype=object)




```python
ts = ts.replace("Burgos1", "Burgos 1")
ts = ts.replace("Burgos4", "Burgos 4")
ts = ts.replace("Burgos5", "Burgos 5")

ts = ts.replace("Salamanca4", "Salamanca 4")
ts = ts.replace("Salamanca5", "Salamanca 5")
ts = ts.replace("Salamanca6", "Salamanca 6")

ts = ts.replace("Le�n1", "Le�n 1")
ts = ts.replace("Le�n3", "Le�n 3")
ts = ts.replace("Leon3", "Le�n 3")
ts = ts.replace("Leon 4", "Le�n 4")

ts = ts.replace("Ponferrada4", "Ponferrada 4")

ts = ts.replace("Miranda de Ebro1", "Miranda de Ebro 1")
ts = ts.replace("Miranda de Ebro2", "Miranda de Ebro 2")
```


```python
ts.ESTACIÓN.unique()
```




    array(['Ávila', 'Aranda de Duero', 'Aranda de Duero 2', 'Burgos 1',
           'Burgos 2', 'Burgos 3', 'Burgos 4', 'Burgos 5', 'Medina de Pomar',
           'Miranda de Ebro 1', 'Miranda de Ebro 2', 'Miranda de Ebro 3',
           'La Robla', 'Lario', 'Le�n 1', 'Le�n 2', 'Le�n 3', 'Le�n 4',
           'Ponferrada 1', 'Ponferrada 2', 'Ponferrada 3', 'Ponferrada 4',
           'Ponferrada 5', 'Guardo', 'Palencia 1', 'Palencia 2', 'Palencia 3',
           'Velilla del R�o Carri�n', 'Venta de Ba�os', 'El Maillo',
           'Salamanca 1', 'Salamanca 2', 'Salamanca 3', 'Salamanca 4',
           'Salamanca 5', 'Salamanca 6', 'Segovia 1', 'Segovia 2',
           'Muriel de la Fuente', 'Soria', 'Medina del Campo', 'Zamora 1',
           'Zamora 2', 'Avila II'], dtype=object)




```python
ts["concat"] = ts["PROVINCIA"] +" - " +  ts["ESTACIÓN"]
ts.head()
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
      <th>CO (mg/m3)</th>
      <th>NO (ug/m3)</th>
      <th>NO2 (ug/m3)</th>
      <th>O3 (ug/m3)</th>
      <th>PM10 (ug/m3)</th>
      <th>SH2 (ug/m3)</th>
      <th>PM25 (ug/m3)</th>
      <th>PST (ug/m3)</th>
      <th>SO2 (ug/m3)</th>
      <th>PROVINCIA</th>
      <th>ESTACIÓN</th>
      <th>concat</th>
    </tr>
    <tr>
      <th>DIA</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1997-01-01</th>
      <td>1.2</td>
      <td>12.0</td>
      <td>33.0</td>
      <td>63.0</td>
      <td>56.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>ÁVILA</td>
      <td>Ávila</td>
      <td>ÁVILA - Ávila</td>
    </tr>
    <tr>
      <th>1997-04-01</th>
      <td>1.6</td>
      <td>56.0</td>
      <td>73.0</td>
      <td>50.0</td>
      <td>74.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>22.0</td>
      <td>ÁVILA</td>
      <td>Ávila</td>
      <td>ÁVILA - Ávila</td>
    </tr>
    <tr>
      <th>1997-06-01</th>
      <td>1.6</td>
      <td>28.0</td>
      <td>46.0</td>
      <td>56.0</td>
      <td>60.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>20.0</td>
      <td>ÁVILA</td>
      <td>Ávila</td>
      <td>ÁVILA - Ávila</td>
    </tr>
    <tr>
      <th>1997-10-01</th>
      <td>1.3</td>
      <td>35.0</td>
      <td>53.0</td>
      <td>38.0</td>
      <td>61.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>18.0</td>
      <td>ÁVILA</td>
      <td>Ávila</td>
      <td>ÁVILA - Ávila</td>
    </tr>
    <tr>
      <th>1997-11-01</th>
      <td>1.3</td>
      <td>27.0</td>
      <td>40.0</td>
      <td>25.0</td>
      <td>59.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>16.0</td>
      <td>ÁVILA</td>
      <td>Ávila</td>
      <td>ÁVILA - Ávila</td>
    </tr>
  </tbody>
</table>
</div>




```python
tsS = {}
for estacion in ts.concat.unique():
    tsS[estacion] = ts.iloc[:,:9][ts.iloc[:,11] == estacion]
```


```python
for estacion in tsS:
    print(estacion)
    tsS[estacion].plot()
```

    ÁVILA - Ávila
    BURGOS - Aranda de Duero
    BURGOS - Aranda de Duero 2
    BURGOS - Burgos 1
    BURGOS - Burgos 2
    BURGOS - Burgos 3
    BURGOS - Burgos 4
    BURGOS - Burgos 5
    BURGOS - Medina de Pomar
    BURGOS - Miranda de Ebro 1
    BURGOS - Miranda de Ebro 2
    BURGOS - Miranda de Ebro 3
    LEÓN - La Robla
    LEÓN - Lario
    LEÓN - Le�n 1
    LEÓN - Le�n 2
    LEÓN - Le�n 3
    LEÓN - Le�n 4
    LEÓN - Ponferrada 1
    LEÓN - Ponferrada 2
    LEÓN - Ponferrada 3
    LEÓN - Ponferrada 4


    /home/luis/anaconda3/lib/python3.7/site-packages/matplotlib/pyplot.py:522: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      max_open_warning, RuntimeWarning)


    LEÓN - Ponferrada 5
    PALENCIA - Guardo
    PALENCIA - Palencia 1
    PALENCIA - Palencia 2
    PALENCIA - Palencia 3
    PALENCIA - Velilla del R�o Carri�n
    PALENCIA - Venta de Ba�os
    SALAMANCA - El Maillo
    SALAMANCA - Salamanca 1
    SALAMANCA - Salamanca 2
    SALAMANCA - Salamanca 3
    SALAMANCA - Salamanca 4
    SALAMANCA - Salamanca 5
    SALAMANCA - Salamanca 6
    SEGOVIA - Segovia 1
    SEGOVIA - Segovia 2
    SORIA - Muriel de la Fuente
    SORIA - Soria
    VALLADOLID - Medina del Campo
    ZAMORA - Zamora 1
    ZAMORA - Zamora 2
    ÁVILA - Avila II



![png](output_15_3.png)



![png](output_15_4.png)



![png](output_15_5.png)



![png](output_15_6.png)



![png](output_15_7.png)



![png](output_15_8.png)



![png](output_15_9.png)



![png](output_15_10.png)



![png](output_15_11.png)



![png](output_15_12.png)



![png](output_15_13.png)



![png](output_15_14.png)



![png](output_15_15.png)



![png](output_15_16.png)



![png](output_15_17.png)



![png](output_15_18.png)



![png](output_15_19.png)



![png](output_15_20.png)



![png](output_15_21.png)



![png](output_15_22.png)



![png](output_15_23.png)



![png](output_15_24.png)



![png](output_15_25.png)



![png](output_15_26.png)



![png](output_15_27.png)



![png](output_15_28.png)



![png](output_15_29.png)



![png](output_15_30.png)



![png](output_15_31.png)



![png](output_15_32.png)



![png](output_15_33.png)



![png](output_15_34.png)



![png](output_15_35.png)



![png](output_15_36.png)



![png](output_15_37.png)



![png](output_15_38.png)



![png](output_15_39.png)



![png](output_15_40.png)



![png](output_15_41.png)



![png](output_15_42.png)



![png](output_15_43.png)



![png](output_15_44.png)



![png](output_15_45.png)



![png](output_15_46.png)



```python
for estacion in tsS:
    tsS[estacion] = tsS[estacion].groupby(pd.Grouper(freq="3M")).agg("mean")
```


```python
for estacion in tsS:
    tsS[estacion]["suma"] = tsS[estacion].pop("CO (mg/m3)")+ tsS[estacion].pop("NO (ug/m3)") + tsS[estacion].pop("NO2 (ug/m3)") + tsS[estacion].pop("O3 (ug/m3)") + tsS[estacion].pop("PM10 (ug/m3)") + tsS[estacion].pop("SH2 (ug/m3)") + tsS[estacion].pop("PM25 (ug/m3)") + tsS[estacion].pop("PST (ug/m3)") + tsS[estacion].pop("SO2 (ug/m3)")
    tsS[estacion].plot(figsize = (20,10))
```

    /home/luis/anaconda3/lib/python3.7/site-packages/matplotlib/pyplot.py:522: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      max_open_warning, RuntimeWarning)



![png](output_17_1.png)



![png](output_17_2.png)



![png](output_17_3.png)



![png](output_17_4.png)



![png](output_17_5.png)



![png](output_17_6.png)



![png](output_17_7.png)



![png](output_17_8.png)



![png](output_17_9.png)



![png](output_17_10.png)



![png](output_17_11.png)



![png](output_17_12.png)



![png](output_17_13.png)



![png](output_17_14.png)



![png](output_17_15.png)



![png](output_17_16.png)



![png](output_17_17.png)



![png](output_17_18.png)



![png](output_17_19.png)



![png](output_17_20.png)



![png](output_17_21.png)



![png](output_17_22.png)



![png](output_17_23.png)



![png](output_17_24.png)



![png](output_17_25.png)



![png](output_17_26.png)



![png](output_17_27.png)



![png](output_17_28.png)



![png](output_17_29.png)



![png](output_17_30.png)



![png](output_17_31.png)



![png](output_17_32.png)



![png](output_17_33.png)



![png](output_17_34.png)



![png](output_17_35.png)



![png](output_17_36.png)



![png](output_17_37.png)



![png](output_17_38.png)



![png](output_17_39.png)



![png](output_17_40.png)



![png](output_17_41.png)



![png](output_17_42.png)



![png](output_17_43.png)



![png](output_17_44.png)



```python
for estacion in tsS:
    tsS[estacion] = tsS[estacion].interpolate(method='linear')
```


```python
for estacion in tsS:
    tsS[estacion].plot(figsize = (20,10))
```

    /home/luis/anaconda3/lib/python3.7/site-packages/matplotlib/pyplot.py:522: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      max_open_warning, RuntimeWarning)



![png](output_19_1.png)



![png](output_19_2.png)



![png](output_19_3.png)



![png](output_19_4.png)



![png](output_19_5.png)



![png](output_19_6.png)



![png](output_19_7.png)



![png](output_19_8.png)



![png](output_19_9.png)



![png](output_19_10.png)



![png](output_19_11.png)



![png](output_19_12.png)



![png](output_19_13.png)



![png](output_19_14.png)



![png](output_19_15.png)



![png](output_19_16.png)



![png](output_19_17.png)



![png](output_19_18.png)



![png](output_19_19.png)



![png](output_19_20.png)



![png](output_19_21.png)



![png](output_19_22.png)



![png](output_19_23.png)



![png](output_19_24.png)



![png](output_19_25.png)



![png](output_19_26.png)



![png](output_19_27.png)



![png](output_19_28.png)



![png](output_19_29.png)



![png](output_19_30.png)



![png](output_19_31.png)



![png](output_19_32.png)



![png](output_19_33.png)



![png](output_19_34.png)



![png](output_19_35.png)



![png](output_19_36.png)



![png](output_19_37.png)



![png](output_19_38.png)



![png](output_19_39.png)



![png](output_19_40.png)



![png](output_19_41.png)



![png](output_19_42.png)



![png](output_19_43.png)



![png](output_19_44.png)



```python
import statsmodels.api as sm
for estacion in tsS:
    res = sm.tsa.ARIMA(tsS[estacion],(1,0,0)).fit()
    prediccion = res.predict(3,len(tsS[estacion]))
    plt.figure(figsize=(20,6))
    plt.plot(tsS[estacion], label='original')
    plt.plot(prediccion, label='prediccion')
    plt.legend(loc=4)
    plt.show()
```

    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)
    /home/luis/anaconda3/lib/python3.7/site-packages/scipy/signal/signaltools.py:1341: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      out_full[ind] += zi
    /home/luis/anaconda3/lib/python3.7/site-packages/scipy/signal/signaltools.py:1344: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      out = out_full[ind]
    /home/luis/anaconda3/lib/python3.7/site-packages/scipy/signal/signaltools.py:1350: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      zf = out_full[ind]



![png](output_20_1.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_3.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_5.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_7.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_9.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_11.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_13.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_15.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_17.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_19.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_21.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_23.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_25.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_27.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_29.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_31.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_33.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_35.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_37.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_39.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_41.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-DEC will be used.
      % freq, ValueWarning)



![png](output_20_43.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-DEC will be used.
      % freq, ValueWarning)



![png](output_20_45.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_47.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_49.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_51.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_53.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_55.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_57.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_59.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_61.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_63.png)



![png](output_20_64.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_66.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_68.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-DEC will be used.
      % freq, ValueWarning)



![png](output_20_70.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_72.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_74.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_76.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_78.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_80.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_82.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_84.png)


    /home/luis/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency Q-OCT will be used.
      % freq, ValueWarning)



![png](output_20_86.png)

