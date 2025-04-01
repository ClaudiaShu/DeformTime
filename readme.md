# DeformTime

This repository maintains the official implementation of methods and experiments presented in our TMLR paper titled "<a href="https://openreview.net/pdf?id=M62P7iOT7d"><strong>DeformTime: Capturing Variable Dependencies with Deformable Attention for Time Series Forecasting</strong></a>".

<!-- [DEFORMTIME: Capturing Variable Dependencies with Deformable Attention for Time Series Forecasting](https://arxiv.org/abs/2030.12345).  -->

<!-- [[project]](https://claudiashu.github.io/publications/2024-arXiv-deformtime/) [[paper]](http) -->

<!-- ## Reference -->

<!-- To cite our paper please use:-->

```
@article{shu2025deformtime,
    author    = {Yuxuan Shu and Vasileios Lampos},
    title     = {{\textsc{DeformTime}: Capturing Variable Dependencies with
                  Deformable Attention for Time Series Forecasting}},
    journal   = {Transactions on Machine Learning Research},
    issn      = {2835-8856},
    year      = {2025},
    url       = {https://openreview.net/forum?id=M62P7iOT7d},
}
```

## Abstract

<link rel="stylesheet" href="style.css">
<!-- <img src=img/figure1.png> -->

In multivariate time series (MTS) forecasting, existing state-of-the-art deep learning approaches tend to focus on autoregressive formulations and overlook the information within exogenous indicators. To address this limitation, we present <span class="small-caps">DeformTime</span>, a neural network architecture that attempts to capture correlated temporal patterns from the input space, and hence, improve forecasting accuracy. It deploys two core operations performed by deformable attention blocks (DABs): learning dependencies across variables from different time steps (variable DAB), and preserving temporal dependencies in data from previous time steps (temporal DAB). Input data transformation is explicitly designed to enhance learning from the deformed series of information while passing through a DAB. We conduct extensive experiments on 6 MTS data sets, using previously established benchmarks as well as challenging infectious disease modelling tasks with more exogenous variables. The results demonstrate that <span class="small-caps">DeformTime</span> improves accuracy against previous competitive methods across the vast majority of MTS forecasting tasks, reducing the mean absolute error by 10\% on average. Notably, performance gains remain consistent across longer forecasting horizons.

## Highlights

We propose <span class="small-caps">DeformTime</span>, a novel MTS forecasting model that better captures inter- and intra-variate dependencies at different temporal granularities. It comprises two Deformable Attention Blocks (DAB) which allow the model to adaptively focus on more informative neighbouring attributes. The below figure shows how different dependencies are established:

<img src=img/dependency.png>

(a) The inter-variable dependency is established across different variables over time. (b) The intra-variable dependency focuses on the important information of the specific variable across time. Both dependencies are adaptively established w.r.t. the input.


## Setting Up the Environment

- Ensure you have [Anaconda](https://www.anaconda.com/products/distribution) installed on your system. You can download and install it from [here](https://www.anaconda.com/products/distribution#download-section).

### 1. Create a Conda Environment

<!-- Maybe have a small paragragh on what environment and device the code and requirements have been tested on -->
Our code was tested with Python 3.9. To create a Conda environment, run the following command in the terminal:

```sh
conda create --name dtime python=3.9
```

### 2. Activate the Conda Environment

Activate the environment using the following command:

```sh
conda activate dtime
```

### 3. Install the Requirements

Install the required packages using:

```setup
pip install -r requirements.txt
```

## Prepare the data sets

### Benchmark data sets

To prepare the benchmark data sets, you need to obtain the ETTh1, ETTh2, and weather data sets. Steps as below:

1. Download the ETTh1, ETTh2, and weather data sets from the [Autoformer repository](https://github.com/thuml/Autoformer).
2. Organize the folders with the following structure:

    ```
    data/
    ├──dataset/
       ├── ETT/
       │   ├── ETTh1.csv
       │   ├── ETTh2.csv
       └── weather/
           └── weather.csv
    ```

### ILI rates

Due to a restricted data sharing policy, we currently cannot provide the full version of the ILI data sets in our experiment. Alternatively, we provide the following instructions for anyone interested in obtaining the data.

#### CDC flu data

The ILI rates of the US HHS regions are obtained by the Centers for Disease Control and Prevention (CDC). You may access the data from their [official website](https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html).

#### RCGP flu data

The ILI data for England in our experiment was obtained from the Royal College of General Practitioners (RCGP) with a fixed list of GP practices throughout England. Although we are unable to provide the original ILI data, you may access the officially reported influenza data [here](https://www.gov.uk/government/collections/weekly-national-flu-reports).

#### Google Health Trends

Queries used in our experiments can be found under the folder `data/dataset/queries/`. The full version of the base queries is provided in `base.csv` and the seed queries used to conduct semantic filtering are in `seed.csv`. We also provide the query lists after semantic filtering. The list for England is available in `UK.csv` (containing $4{,}396$ queries), and the list for US regions is available in `US.csv` (containing $2{,}479$ queries). Apply [here](https://support.google.com/trends/contact/trends_api) to get access to Google Trends API.


## Model training

To train the model(s) in the paper (for the benchmark data sets), run the below commands:

For ETTh1 tasks:
```ETTh1
bash scripts/DeformTime/ETTh1.sh
```

For ETTh2 tasks:
```ETTh2
bash scripts/DeformTime/ETTh2.sh
```

For weather tasks:
```weather
bash scripts/DeformTime/weather.sh
```

The results will be saved in a `result.txt` file.


## Results

Our model achieves the following performance with MAE score. The best results are in **bold** font and the second best are <ins>underlined</ins>. We use iTrans. and Cross. as the abbreviation for iTransformer and Crossformer respectively. See full results in the paper.
<!-- (We use $\epsilon~\\%$ to denote the SMAPE score. See more details in the paper.) -->

<!DOCTYPE html>
<html lang="en">

<body>

<table>
<hr>
  <tr>
    <th>Tasks</th>
    <th>$H$</th>
    <th class="small-caps">DeformTime</th>
    <th>PatchTST</th>
    <th>iTrans.</th>
    <th>TimeMixer</th>
    <th>Cross.</th>
    <th>LightTS</th>
    <th>DLinear</th>

  </tr>

  <tr>
    <td rowspan="4">ETTh1</td>
    <td>96</td>
    <td><b>0.1941</b></td>
    <td><ins>0.2017</ins></td>
    <td>0.2052</td>
    <td>0.2112</td>
    <td>0.2126</td>
    <td>0.2215</td>
    <td>0.2599</td>
  </tr>
  <tr>
    <td>192</td>
    <td><b>0.2116</b></td>
    <td>0.2409</td>
    <td>0.2429</td>
    <td><ins>0.2382</ins></td>
    <td>0.2820</td>
    <td>0.2636</td>
    <td>0.3798</td>

  </tr>
  <tr>
    <td>336</td>
    <td><b>0.2158</b></td>
    <td><ins>0.2559</ins></td>
    <td>0.2593</td>
    <td>0.2625</td>
    <td>0.2947</td>
    <td>0.2807</td>
    <td>0.6328</td>

  </tr>
  <tr>
    <td>720</td>
    <td><b>0.2862</b></td>
    <td>0.3087</td>
    <td><ins>0.2886</ins></td>
    <td>0.3055</td>
    <td>0.3350</td>
    <td>0.5334</td>
    <td>0.7563</td>

  </tr>

  <tr>
    <td rowspan="4">ETTh2</td>
    <td>96</td>
    <td><b>0.3121</b></td>
    <td><ins>0.3145</ins></td>
    <td>0.3420</td>
    <td>0.3454</td>
    <td>0.3486</td>
    <td>0.3507</td>
    <td>0.3349</td>

  </tr>
  <tr>
    <td>192</td>
    <td><b>0.3281</b></td>
    <td><ins>0.3839</ins></td>
    <td>0.4233</td>
    <td>0.4183</td>
    <td>0.4035</td>
    <td>0.4022</td>
    <td>0.4084</td>

  </tr>
  <tr>
    <td>336</td>
    <td><b>0.3450</b></td>
    <td><ins>0.4018</ins></td>
    <td>0.4332</td>
    <td>0.4380</td>
    <td>0.4487</td>
    <td>0.4425</td>
    <td>0.4710</td>

  </tr>
  <tr>
    <td>720</td>
    <td><b>0.3640</b></td>
    <td>0.4960</td>
    <td><ins>0.4565</ins></td>
    <td>0.4729</td>
    <td>0.5832</td>
    <td>0.6252</td>
    <td>0.7981</td>

  </tr>

  <tr>
    <td rowspan="4">Weather</td>
    <td>96</td>
    <td><b>0.0244</b></td>
    <td>0.0258</td>
    <td>0.0277</td>
    <td>0.0322</td>
    <td>0.0271</td>
    <td>0.0293</td>
    <td><ins>0.0251</ins></td>

  </tr>
  <tr>
    <td>192</td>
    <td><b>0.0260</b></td>
    <td>0.0279</td>
    <td>0.0277</td>
    <td>0.0347</td>
    <td>0.0308</td>
    <td>0.0319</td>
    <td><ins>0.0270</ins></td>

  </tr>
  <tr>
    <td>336</td>
    <td><b>0.0291</b></td>
    <td><ins>0.0303</ins></td>
    <td>0.0308</td>
    <td>0.0359</td>
    <td>0.0345</td>
    <td>0.0317</td>
    <td>0.0305</td>

  </tr>
  <tr>
    <td>720</td>
    <td><ins>0.0363</ins></td>
    <td>0.0389</td>
    <td>0.0395</td>
    <td>0.0457</td>
    <td>0.0395</td>
    <td>0.0386</td>
    <td><b>0.0352</b></td>

  </tr>

  <tr>
    <td rowspan="4">ILI-ENG</td>
    <td>7</td>
    <td><b>1.6417</b></td>
    <td>2.3115</td>
    <td>2.3084</td>
    <td>2.1748</td>
    <td><ins>1.8698</ins></td>
    <td>2.2397</td>
    <td>2.8214</td>

  </tr>
  <tr>
    <td>14</td>
    <td><b>2.2308</b></td>
    <td>3.2547</td>
    <td>3.2301</td>
    <td>3.0209</td>
    <td><ins>2.6543</ins></td>
    <td>2.6879</td>
    <td>3.7922</td>

  </tr>
  <tr>
    <td>21</td>
    <td><b>2.6500</b></td>
    <td>4.3192</td>
    <td>4.2347</td>
    <td>3.5501</td>
    <td><ins>3.0014</ins></td>
    <td>3.3616</td>
    <td>4.4739</td>

  </tr>
  <tr>
    <td>28</td>
    <td><b>2.7228</b></td>
    <td>4.9964</td>
    <td>4.8125</td>
    <td>4.1188</td>
    <td><ins>3.1983</ins></td>
    <td>3.4132</td>
    <td>5.0347</td>

  </tr>

  <tr>
    <td rowspan="4">ILI-US2</td>
    <td>7</td>
    <td><b>0.4122</b></td>
    <td>0.7097</td>
    <td>0.6507</td>
    <td>0.5284</td>
    <td><ins>0.4400</ins></td>
    <td>0.4632</td>
    <td>0.7355</td>

  </tr>
  <tr>
    <td>14</td>
    <td><b>0.4752</b></td>
    <td>0.8635</td>
    <td>0.7896</td>
    <td>0.6556</td>
    <td>0.5852</td>
    <td><ins>0.5827</ins></td>
    <td>0.8435</td>

  </tr>
  <tr>
    <td>21</td>
    <td><b>0.5425</b></td>
    <td>1.0286</td>
    <td>0.8042</td>
    <td>0.6794</td>
    <td><ins>0.6245</ins></td>
    <td>0.6683</td>
    <td>0.9124</td>

  </tr>
  <tr>
    <td>28</td>
    <td><b>0.5538</b></td>
    <td>1.1525</td>
    <td>0.9619</td>
    <td>0.8853</td>
    <td><ins>0.6512</ins></td>
    <td>0.7175</td>
    <td>0.9805</td>

  </tr>

  <tr>
    <td rowspan="4">ILI-US9</td>
    <td>7</td>
    <td><b>0.2622</b></td>
    <td>0.4116</td>
    <td>0.4057</td>
    <td>0.3239</td>
    <td><ins>0.3149</ins></td>
    <td>0.3185</td>
    <td>0.4675</td>

  </tr>
  <tr>
    <td>14</td>
    <td><b>0.3084</b></td>
    <td>0.5020</td>
    <td>0.4702</td>
    <td>0.4060</td>
    <td><ins>0.3571</ins></td>
    <td>0.3791</td>
    <td>0.5467</td>

  </tr>
  <tr>
    <td>21</td>
    <td><b>0.3179</b></td>
    <td>0.5935</td>
    <td>0.5106</td>
    <td>0.4576</td>
    <td><ins>0.3418</ins></td>
    <td>0.4754</td>
    <td>0.6001</td>

  </tr>
  <tr>
    <td>28</td>
    <td><b>0.3532</b></td>
    <td>0.6665</td>
    <td>0.6498</td>
    <td>0.5124</td>
    <td><ins>0.3747</ins></td>
    <td>0.4769</td>
    <td>0.6564</td>

  </tr>

</table>
</table>

</body>
</html>

## Acknowledgements

- Our implementation of attention deformation was inspired by [DAT](https://github.com/LeapLabTHU/DAT).
- We also acknowledge [Informer](https://github.com/zhouhaoyi/Informer2020) and [Autoformer](https://github.com/thuml/Autoformer) for their valuable code and data for time series forecasting.

