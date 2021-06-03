# ADC

This is a repository generated from **[Diffusion Improves Graph Learning](https://www.kdd.in.tum.de/gdc)**  

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