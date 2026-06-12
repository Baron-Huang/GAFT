# Graph Adversarial Fusion Transformer for Topology-aware Tumor–Microenvironment Interaction Modeling in SCC
## 🧔: Authors [*Corresponding author]
-  Pan Huang, Yingbo Qu, Peng He, Francesco Mercaldo, Antonella Santone, and Peng Feng*

## :fire: News

- [xxxx/xx/xx] _xxxx_.



## :rocket: Pipeline

Here's an overview of our **Graph Adversarial Fusion Transformer (GAFT)** method:

![Figure 1](./images/xxx.jpg)



## :mag: TODO
<font color="red">**We are currently organizing all the code. Stay tuned!**</font>
- [x] training code
- [x] Evaluation code
- [x] Model code
- [ ] Pretrained weights
- [ ] Datasets





## 🛠️ Getting Started

To get started with **GAFT**, follow the installation instructions below.

1.  Clone the repo

```sh
git clone https://github.com/Baron-Huang/GAFT
```

2. Install dependencies
   
```sh
pip install -r requirements.txt
```

3. Training on Swin Transformer-S Backbone
```sh
sh GAFT.sh
Modify: --abla_type sota --run_mode train --random_seed ${seed}
```

4. Evaluation
```sh
sh GAFT.sh
Modify: --abla_type sota --run_mode test --random_seed ${seed}
```

5. Extract features for plots
```sh
sh GAFT.sh
Modify: --abla_type sota --run_mode test --random_seed ${seed} --feat_extract
```

6. Interpretability plots
```sh
sh GAFT.sh
Modify: --abla_type sota --run_mode test --random_seed ${seed} --bag_weight
```

## :postbox: Contact
If you have any questions, please contact [Dr.Pan Huang](https://scholar.google.com/citations?user=V_7bX4QAAAAJ&hl=zh-CN) (`panhuang@polyu.edu.hk`).
