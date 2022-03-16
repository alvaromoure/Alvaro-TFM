This is a repository for Chest X-Ray COVID-19 detection based on a COvidNet and different preprocessing steps. The network impolementation is based on the COVIDNet proposed in L. Wang, Z. Q. Lin, and A. Wong, “Covid-net: A tailored deep convolutional neural network design for detection of covid-19 cases from chest radiography images,” Scientific Reports, vol. 10, no. 19549, 2020, following the PyTorch implementation available at [GitHub](https://github.com/iliasprc/COVIDNet)

The respository includes also impelementation of a network based on ResNet50, as proposed in: Li, L., Qin, L., Xu, Z., Yin, Y., Wang, X., Kong, B., Bai, J., Lu, Y., Fang, Z., Song, Q. and Cao, K., 2020. Using artificial intelligence to detect COVID-19 and community-acquired pneumonia based on pulmonary CT: evaluation of the diagnostic accuracy. Radiology, 296(2). [url](https://pubs.rsna.org/doi/10.1148/radiol.2020200905?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200pubmed)

The models available are:

- Original CovidNet
- Modified CovidNet with dropout and Grad_CAM functionalities
- CovidNet with Deep Explainer functionalities
- Covid_Resnet50
- DenseNet

The preprocessing steps evaluated include cropping and segmentation of the lungs based on the 2D Lung segmentation available at [GitHub](https://github.com/imlab-uiip/lung-segmentation-2d). This network is also included in the segmenter directory of this respository. The description of all the experiments and results can be consulted on:

J.D. Arias-Londoño, J.A. Gómez-García, L. Moro-Velázquez, J.I. Godino-Llorente. Artificial Intelligence applied to chest X-Ray images for the automatic detection of COVID-19. A thoughtful evaluation approach. IEEE Access, vol 8. 2020. [Open access](https://ieeexplore.ieee.org/document/9293268)

If you find our work useful, can cite our paper using:

```
@article{arias2020,
  title={Artificial Intelligence applied to chest X-Ray images for the automatic detection of COVID-19. A thoughtful evaluation approach},
  author={Arias-Londo{\~n}o, Julian D and Gomez-Garcia, Jorge A and Moro-Velazquez, Laureano and Godino-Llorente, Juan I},
  journal={IEEE Access},
  volume = {8},
  year={2020}
}
```

To execute the code take into account the following steps:

- Build the train.txt and test.txt files according to the structure proposed in the original implementation, which include the name of every file with its corresponding label ['pneumonia', 'normal', 'COVID-19'].
- The path to those files must be set in dataset.py file
- To reproduce experiment 1 of the paper run main.py with the original images
- To reproduce experiments 2 and 3, you can use LungSegmentation notebook to apply the Segmenter CNN to the dataset and get a segmentation mask for every single image in the dataset.
- In the MatLab_preprocessing directory there is a script that takes the masks obtained in the previous step and apply them to the original images in order to get the crooped and crooped-segmented datasets; this preprocessed images can be used to reproduce expriments 2 and 3 of the paper respectively. For doing that, just change the path to the data directory during the call to main.py, and point it to the directories containing the preprocessed images.
