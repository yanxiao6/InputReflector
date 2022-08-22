
<h1 align="center">InputReflector</h1>


<p align="center">
<b>
A runtime approach that mitigates
DNN mis-predictions caused by the unexpected inputs to
the DNN.</b>
<br/><br/>
<a href=""><img src="images/figure1.png" alt="Logo" width=80%></a>



<p align="center">
Code release and supplementary materials for:</br>
  <b>"Repairing Failure-inducing Inputs with Input Reflection"</b></br>
    The 37th IEEE/ACM International Conference on Automated Software Engineering (<b>ASE 2022</b>)
    <br />
    <a href="https://yanxiao6.github.io/">Yan Xiao</a>
    ·
    <a href="http://linyun.info">Yun Lin</a>
    ·
    <a href="https://www.cs.ubc.ca/~bestchai/">Ivan Beschastnikh</a>
    ·
    <a href="https://sunchangsheng.com">Changsheng Sun</a>
    <br/>
    <a href="https://cs.gmu.edu/~dsr/">David S. Rosenblum</a>
    ·
    <a href="https://www.comp.nus.edu.sg/~dongjs/">Jin Song Dong</a>
    <br/><br/>
    <a href="https://www.comp.nus.edu.sg"><img src="https://www.comp.nus.edu.sg/templates/t3_nus2015/images/assets/logos/logo.png" alt="Logo" height=45px style="padding-right: 40px; padding left: 20px;"> </a>  <br/>
    <a href=""> <img src="https://brand3.sites.olt.ubc.ca/files/2018/09/5NarrowLogo_ex_768.png" height=50px style="padding-right: 20px; padding left: 20px;"> </a>
    <a href=""> <img src="https://cs.gmu.edu/~dsr/images/GMU_PLogo_RGB.jpg" height=60px style="padding-right: 20px; padding left: 20px;"> </a>
    <br/><br/>
  </p>


<p align="center">
    <!-- <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
    <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a><br><br> -->
    <a href='http://linyun.info/publications/ase22.pdf'>
      <img src='https://img.shields.io/badge/Paper-PDF(NotFinalVersion)-green?style=flat&logo=arXiv&logoColor=green' alt='Paper PDF'>
    </a>
    <!-- <a href='https://arxiv.org/abs/2103.02371'>
      <img src='https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg' alt='Paper PDF'>
    </a>  -->
    <a href='https://trustdnn.comp.nus.edu.sg'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    <a href='https://youtu.be/'>
      <img src='https://img.shields.io/badge/Presentation-Video-red?style=flat&logo=youtube&logoColor=red' alt='Youtube Video'>
    </a>
    <!-- <a href='https://colab.research.google.com/drive/' style='padding-left: 0.5rem;'>
      <img src='https://colab.research.google.com/assets/colab-badge.svg' alt='Google Colab'>
    </a>
    <a href='https://discord.gg/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Discord-Room-purple?style=flat&logo=Discord&logoColor=purple' alt='Discord Room'> -->
    </a>
  </p>
</p>


## Repo structure
- `resnet.py`: code for Resnet-20 to train the subject models
- `train_model.py`: code for ConvNet and VGG-16 to train the subject models
- `special_transformation.py`: code for transformations
- `seek_degree.py`: find the cornel degrees for each transformation
- `train.py`: train inputreflector
- `triplet_loss`: loss functions
- `eval.py`: obtain distances between given instances and training data
- `collect_auroc_sia.py`: generate AUROC from the distance
- `search_threshold_quad.py`: search for the best threshold of detecting deviated data on the validation dataset and calculate the model accuracy after calling InputReflector

## Dependencies
pip install -r requirements.txt

## How to run

- To train the subject models and InputReflector: bash log.sh
- To evaluate the performance of InputReflector: bash log_eval.sh


## Supplementary
### Sampling in Quadroplet Loss

In Section 3.2.1, InputReflector uses Quadruplet network for the reflection process. During the construction of the Quadruplet network, the challenge is how to sample the quadruplet from the training data. Algorithm 2 discusses how to mine quadruplet samples during training. The sample mining process of the Siamese network in Section 3.1 also uses this technique. We omit Algorithm 2 due to space limitation.

<img src="images/algo2.png" alt="Sampling in Quadroplet Loss" style="zoom: 50%;" />

The loss of the Quadruplet network consists of two parts (Line 15 in Algorithm 2). The first part, <!-- $loss_{an}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=loss_%7Ban%7D">, is the traditional triplet loss that is the main constraint. The second part, <!-- $loss_{nn}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=loss_%7Bnn%7D">, is auxiliary to the first loss and conforms to the structure of traditional triplet loss but has different triplets. We use two different margins (<!-- $m_{1} > m_{2}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=m_%7B1%7D%20%3E%20m_%7B2%7D">) to balance the two constraints. We now discuss how to mine triplets for each loss.

First, a 2D matrix of distances between all the embeddings is
calculated and stored in <!-- $pairwise\_dist$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=pairwise%5C_dist"> (line 1). Given an anchor, we define the *hardest positive example* as having the same label as the anchor and whose distance from the anchor is the largest (<!-- $dist_{hard_{p}}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=dist_%7Bhard_%7Bp%7D%7D">) among all the positive examples (lines 2-4). Similarly, the *hardest negative example* has a different
label than the anchor and has the smallest distance from the anchor (<!-- $dist_{hard_{n}}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=dist_%7Bhard_%7Bn%7D%7D">) among all the negative examples (lines 5-8). These *hardest positive example* and *hardest negative example* along with the anchor are formed as a triplet to minimizing <!-- $loss_{an}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=loss_%7Ban%7D"> in line 9. After convergence, the maximum intra-class distance is required to be smaller than the minimum inter-class distance with respect to the same anchor. 

To push away negative pairs from positive pairs, one more loss, <!-- $loss_{nn}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=loss_%7Bnn%7D">, is introduced.
Its aim is to make the maximum intra-class distance smaller than the minimum inter-class distance regardless of whether pairs contain the same anchor. This loss constrains the distance between positive pairs (i.e., samples with the same label) to be less than any other negative pairs (i.e., samples with different labels that are also different from the label of the corresponding positive samples). With the help of this constraint, the maximum intra-class distance must be less than the minimum inter-class distance regardless of whether pairs contain the same anchor. To mine such triplets, the valid triplet first needs to be filtered out on line 10 where <!-- $i, j, k$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=i%2C%20j%2C%20k"> are distinct and 

<!-- $(labels[i] \neq labels[j]) \ \& \ (labels[i] \neq labels[k]) \ \& \ (labels[j] \neq labels[k])$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=(labels%5Bi%5D%20%5Cneq%20labels%5Bj%5D)%20%5C%20%5C%26%20%5C%20(labels%5Bi%5D%20%5Cneq%20labels%5Bk%5D)%20%5C%20%5C%26%20%5C%20(labels%5Bj%5D%20%5Cneq%20labels%5Bk%5D)">



Then, the hardest negative pairs are sampled whose distance is the minimum among all negative pairs in each batch during training (line 11-13).  Finally <!-- $loss_{an}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=loss_%7Ban%7D"> is minimized to further enlarge the inter-class variations in Line 14.




## Citation
```bibtex
@inproceedings{xiao2022repairing,
  title={Repairing Failure-inducing Inputs with Input Reflection},
  author={Xiao, Yan and Lin, Yun and Beschastnikh, Ivan and Sun, Changsheng and Rosenblum, David S and Dong, Jin Song},
  booktitle={2021 37th IEEE/ACM International Conference on Automated Software Engineering (ASE)},
  year={2022},
  organization={IEEE}
}
```

## License
This code and model are available for non-commercial scientific research purposes as defined in the [LICENSE](LICENSE) file. By downloading and using the code and model you agree to the terms in the [LICENSE](LICENSE).

## Contact

For more questions, please contact <cssun@u.nus.edu>.