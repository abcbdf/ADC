# ADC

Reference implementation (example) of the model proposed in the paper:

**[Adaptive Diffusion in Graph Neural Networks](https://proceedings.neurips.cc/paper/2021/hash/c42af2fa7356818e0389593714f59b52-Abstract.html)**   
by Jialin Zhao, Yuxiao Dong, Ming Ding, Evgeny Kharlamov, Jie Tang
Published at NeurIPS 2021.

To run ADC:

`python ADC.py --config config_Cora_GCN.yaml --t 5`

`python ADC.py --config config_Cora_JK.yaml --t 1`

`python ADC.py --config config_Cora_ARMA.yaml --t 5`

To run GADC:

`python GADC.py --config config_Cora_GCN.yaml`

To run GDC:

`python GDC.py --config config_Cora_GCN.yaml --preprocessing heat`

To run original model (GCN, JKNet, ARMA):

`python GDC.py --config config_Cora_GCN.yaml --preprocessing none`

## Contact
Please contact zjl19970607@163.com in case you have any questions.

## Cite
Please cite our paper if you use the model or this code in your own work:

```
@inproceedings{NEURIPS2021_c42af2fa,
 author = {Zhao, Jialin and Dong, Yuxiao and Ding, Ming and Kharlamov, Evgeny and Tang, Jie},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {M. Ranzato and A. Beygelzimer and Y. Dauphin and P.S. Liang and J. Wortman Vaughan},
 pages = {23321--23333},
 publisher = {Curran Associates, Inc.},
 title = {Adaptive Diffusion in Graph Neural Networks},
 url = {https://proceedings.neurips.cc/paper_files/paper/2021/file/c42af2fa7356818e0389593714f59b52-Paper.pdf},
 volume = {34},
 year = {2021}
}
```
