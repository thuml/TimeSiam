# TimeSiam (ICML 2024)

This is the codebase for the paper: TimeSiam: A Pre-Training Framework for Siamese Time-Series Modeling [[Paper]](https://arxiv.org/abs/2402.02475) [[Slides]](https://cloud.tsinghua.edu.cn/f/99ab4b2aec8b4614b7b4/)

## Introduction

TimeSiam pre-trains Siamese encoders to capture temporal correlations between past and current subseries. It benefits from diverse masking augmented subseries and learns time-dependent representations through past-to-current reconstruction. Lineage embeddings are introduced to further foster the learning of diverse temporal correlations.

* In the spirit of learning temporal correlations, we propose TimeSiam that leverages Siamese networks to **capture correlations among temporally distanced subseries**.
* With Siamese encoders to **reconstruct current masked subseries based on past observation and lineage embeddings to capture subseries disparity**, TimeSiam can learn diverse time-dependent representations.
* TimeSiam **achieves consistent state-of-the-art fine-tuning performance** across thirteen standard benchmarks, excelling in various time series analysis tasks.

<p align="center">
<img src=".\figs\Architecture.png" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overview of TimeSiam.
</p>

#### Pre-training

TimeSiam pre-training involves the following two modules: Siamese subseries sampling and Siamese modeling.

**(1) Siamese Subseries Sampling** 

We construct Siamese subseries pairs by randomly sampling a past sample preceding the current sample in the same time series. Furthermore, we adopt a simple masking augmentation to generate augmented current subseries.

**(2) Siamese Modeling**

Our Siamese sampling strategy natively derives a past-to-current reconstruction task to reconstruct the masked current subseries.


#### Fine-tuning
Under the cooperation of lineage embeddings, TimeSiam can further derive two types of fine-tuning paradigms, covering both fixed and extended input series settings.

**(1) Fixed-Input-Multiple-Lineages**

TimeSiam innovatively pre-trains Siamese encoders with diverse lineage embeddings to capture different distanced temporal correlations, which allows TimeSiam to derive diverse representations with different lineages for the same input series.

**(2) Extended-Input-Multiple-Lineages**

TimeSiam can leverage multiple lineage embeddings trained under different temporal distanced pairs to different segments, which can natively conserve the temporal order of different segments. This advantage is achieved by associating each segment with its
respective lineage embedding.

## Get Started

1. Install Pytorch and necessary dependencies.

```python
pip install -r requirements.txt
```

2. The datasets can be obtained can be obtained from [Google Drive](https://drive.google.com/file/d/1CC4ZrUD4EKncndzgy5PSTzOPSqcuyqqj/view?usp=sharing), [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/a238e34ff81a42878d50/?dl=1), and [TSLD](https://cloud.tsinghua.edu.cn/f/3b271331e8c54fb7872b/?dl=1).


3. Experiment scripts can be found under the folder ```./scripts```.

## Show Cases
The reconstruction effect across various datasets with different data distributions, as detailed below.

<p align="center">
<img src=".\figs\showcase1.png" alt="" align=center />
<br><br>
<b>Figure 2.</b> Showcases of TimeSiam in reconstructing time series from different datasets with 25% masked raito.
</p>

## Multiple Lineages Representation Visualization

We employ Principal Components Analysis (PCA) to elucidate the distribution of temporal representations on the ECL dataset. When time series is fed into a pre-trained Siamese network with different lineage embeddings, the model generates divergent temporal representations that representations derived from
the same lineage embeddings tend to be closely clustered together, while representations from different lineage embeddings exhibit significant dissimilarity. 

<p align="center">
<img src=".\figs\visual_rep_0.1.jpg" alt="" align=center />
<br><br>
<b>Figure 2.</b> Visualizing the effect of temporal shift representations. (a) Test distribution under three types of lineage embeddings. (b) Test distribution under six types of lineage embeddings.
</p>

## Citation
If you find this repo useful, please cite our paper.

```plain
@inproceedings{dong2024timesiam,
  title={TimeSiam: A Pre-Training Framework for Siamese Time-Series Modeling},
  author={Dong, Jiaxiang and Wu, Haixu and Wang, Yuxuan and Qiu, Yunzhong and Zhang, Li and Wang, Jianmin and Long, Mingsheng},
  booktitle={ICML},
  year={2024}
}
```

## Contact

If you have any questions, please contact [djx20@mails.tsinghua.edu.cn](mailto:djx20@mails.tsinghua.edu.cn).


## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

* Code: [TSLib](https://github.com/thuml/Time-Series-Library), [TS2Vec](https://github.com/yuezhihan/ts2vec), [CoST](https://github.com/salesforce/CoST), [LaST](https://github.com/zhycs/LaST), [TF-C](https://github.com/mims-harvard/TFC-pretraining), [COMET](https://github.com/DL4mHealth/COMET), [TST](https://github.com/gzerveas/mvts_transformer), [Ti-MAE](https://github.com/asmodaay/ti-mae), [PatchTST](https://github.com/yuqinie98/PatchTST), [SimMTM](https://github.com/thuml/simmtm).

* Datasets: [AD](https://figshare.com/ndownloader/files/43196127), [PTB](https://figshare.com/ndownloader/files/43196133), [TDBrain](https://brainclinics.com/resources/).

The users need to request permission to download on the TDBrain official website and process the raw data.