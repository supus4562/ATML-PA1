# Project References and Citations

Here is a compiled list of references for all the architectures, datasets, and methods used in your experiments. You can copy and paste these into your report's bibliography or use the provided BibTeX entries for LaTeX.

## 1. Datasets

**Oxford-IIIT Pet Dataset**
> Parkhi, O. M., Vedaldi, A., Zisserman, A., & Jawahar, C. V. (2012). Cats and dogs. In *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

**PACS Dataset**
> Li, D., Yang, Y., Song, Y. Z., & Hospedales, T. M. (2017). Deeper, broader and artier domain generalization. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*.

**CIFAR-10 & CIFAR-100**
> Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images. *Technical Report, University of Toronto*.

## 2. Architectures

**ResNet (ResNet-18, ResNet-50)**
> He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

**ViT (Vision Transformer)**
> Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. In *International Conference on Learning Representations (ICLR)*.

**CLIP**
> Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. In *International Conference on Machine Learning (ICML)*.

## 3. Methods & Algorithms

**AdaIN (Task 1)**
> Huang, X., & Belongie, S. (2017). Arbitrary style transfer in real-time with adaptive instance normalization. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*.

**DAN - Deep Adaptation Network (Task 2/3)**
> Long, M., Cao, Y., Wang, J., & Jordan, M. (2015). Learning transferable features with deep adaptation networks. In *International Conference on Machine Learning (ICML)*.

**DANN - Domain-Adversarial Neural Networks (Task 2)**
> Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., ... & Lempitsky, V. (2016). Domain-adversarial training of neural networks. *The Journal of Machine Learning Research*, 17(1), 2096-2030.

**CDAN - Conditional Adversarial Domain Adaptation (Task 2)**
> Long, M., Cao, Z., Wang, J., & Jordan, M. I. (2018). Conditional adversarial domain adaptation. In *Advances in Neural Information Processing Systems (NeurIPS)*.

**SAM - Sharpness-Aware Minimization (Task 3)**
> Foret, P., Kleiner, A., Mobahi, H., & Neyshabur, B. (2021). Sharpness-aware minimization for efficiently improving generalization. In *International Conference on Learning Representations (ICLR)*.

**PROSER (Task 4)**
> Zhou, D. W., Ye, H. J., & Zhan, D. C. (2021). Learning placeholders for open-set recognition. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

**RandAugment (used in GCSC, Task 4)**
> Cubuk, E. D., Zoph, B., Shlens, J., & Le, Q. V. (2020). Randaugment: Practical automated data augmentation with a reduced search space. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*.

---

## BibTeX Entries

```bibtex
@inproceedings{parkhi2012cats,
  title={Cats and dogs},
  author={Parkhi, Omkar M and Vedaldi, Andrea and Zisserman, Andrew and Jawahar, CV},
  booktitle={2012 IEEE conference on computer vision and pattern recognition},
  pages={3498--3505},
  year={2012},
  organization={IEEE}
}

@inproceedings{li2017deeper,
  title={Deeper, broader and artier domain generalization},
  author={Li, Da and Yang, Yongxin and Song, Yi-Zhe and Hospedales, Timothy M},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={5542--5550},
  year={2017}
}

@article{krizhevsky2009learning,
  title={Learning multiple layers of features from tiny images},
  author={Krizhevsky, Alex and Hinton, Geoffrey and others},
  year={2009},
  publisher={Toronto, ON, Canada}
}

@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}

@inproceedings{dosovitskiy2020image,
  title={An image is worth 16x16 words: Transformers for image recognition at scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
  booktitle={International Conference on Learning Representations},
  year={2021}
}

@inproceedings{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  booktitle={International conference on machine learning},
  pages={8748--8763},
  year={2021},
  organization={PMLR}
}

@inproceedings{huang2017arbitrary,
  title={Arbitrary style transfer in real-time with adaptive instance normalization},
  author={Huang, Xun and Belongie, Serge},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={1501--1510},
  year={2017}
}

@inproceedings{long2015learning,
  title={Learning transferable features with deep adaptation networks},
  author={Long, Mingsheng and Cao, Yue and Wang, Jianmin and Jordan, Michael},
  booktitle={International conference on machine learning},
  pages={97--105},
  year={2015},
  organization={PMLR}
}

@article{ganin2016domain,
  title={Domain-adversarial training of neural networks},
  author={Ganin, Yaroslav and Ustinova, Evgeniya and Ajakan, Hana and Germain, Pascal and Larochelle, Hugo and Laviolette, Fran{\c{c}}ois and Marchand, Mario and Lempitsky, Victor},
  journal={The journal of machine learning research},
  volume={17},
  number={1},
  pages={2096--2030},
  year={2016},
  publisher={JMLR. org}
}

@inproceedings{long2018conditional,
  title={Conditional adversarial domain adaptation},
  author={Long, Mingsheng and Cao, Zhangjie and Wang, Jianmin and Jordan, Michael I},
  booktitle={Advances in neural information processing systems},
  volume={31},
  year={2018}
}

@inproceedings{foret2020sharpness,
  title={Sharpness-aware minimization for efficiently improving generalization},
  author={Foret, Pierre and Kleiner, Ariel and Mobahi, Hossein and Neyshabur, Behnam},
  booktitle={International Conference on Learning Representations},
  year={2021}
}

@inproceedings{zhou2021learning,
  title={Learning placeholders for open-set recognition},
  author={Zhou, Da-Wei and Ye, Han-Jia and Zhan, De-Chuan},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={4401--4410},
  year={2021}
}

@inproceedings{cubuk2020randaugment,
  title={Randaugment: Practical automated data augmentation with a reduced search space},
  author={Cubuk, Ekin D and Zoph, Barret and Shlens, Jonathon and Le, Quoc V},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops},
  pages={702--703},
  year={2020}
}
```
