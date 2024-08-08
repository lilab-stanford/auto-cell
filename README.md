Auto-cell  
===========
## Automated cell annotation and classification on histopathology for spatial biomarker discovery

**Abstract:** Histopathology with hematoxylin and eosin (H&E) staining is routinely employed for clinical diagnoses. Single-cell analysis of histopathology provides a powerful tool for understanding the intricate cellular interactions underlying disease progression and therapeutic response. However, existing efforts are hampered by inefficient and error-prone human annotations. Here, we present an experimental and computational approach for automated cell annotation and classiﬁcation on H&E-stained images. Instead of human annotations, we use multiplexed immunoﬂuorescence (mIF) to deﬁne cell types based on cell lineage protein markers. By co-registering H&E images with mIF of the same tissue section at the single-cell level, we create a dataset of 1,127,252 cells with high-quality annotations. A deep learning model is trained to classify four cell types on H&E images with an overall accuracy of 86%-89%. Further, we show that spatial interactions among speciﬁc immune cells in the tumor microenvironment are linked to patient survival and response to immune checkpoint inhibitors. Our work provides a scalable approach for single-cell analysis of standard histopathology and may enable discovery of novel spatial biomarkers for precision oncology.

## Dependencies:
* NVIDIA GPU (Tested on Nvidia GeForce RTX 2080 Ti x 4) with CUDA 11.8 and cuDNN 8.9 (Tested on Ubuntu 22.04)
* Python (3.7.11). PyTorch (1.9.1) for deep learning. These additional Python libraries were used: lifelines(0.26.3), matplotlib(3.4.2 ), numpy(1.20.3), opencvpython(3.4.2), openslide-python(1.2.0), pandas(1.1.3), pillow(8.3.1), scikit-learn(1.0.2), scipy(1.7.1), seaborn(0.11.2), torchaudio(0.9.1 ), torchvision(0.10.1+cu111), tqdm(4.50.2), umap (0.5.5), scanpy (1.10.1), csbdeep (0.7.4), timm (0.6.13), and scikit-survival (0.17.2).
* Stardist (https://github.com/stardist/stardist)
* DeeperHistReg (https://github.com/MWod/DeeperHistReg)
* Solo-learn (https://github.com/vturrisi/solo-learn)

## Step 1: Processing mIF images for cell type identiﬁcation
* Use the InForm digital image analysis software (Akoya Biosciences) to preprocess mIF images, cell segmentation information for each core should be stored in a .txt file which contains sample name, cell id, cell position, cell area, and the average marker expression value for each cell.
* Use the `./cell_annotation/cell_cluster.py` script to perform cell clustering and annotation and save results in a .csv file.

## Step 2: Co-registering mIF and H&E images, and transferring of cell type labels
* Use the DeeperHistReg to co-register mIF and H&E images and obtain the registered H&E images.
* Run `./cell_annotation/cell_segmet_he.py` to color-normalize H&E images and segment all cell nuclei in them using StarDist algorithm.
* Transfer cell type labels from mIF to H&E images using the `./cell_annotation/cell_transfer.py` script.

## Step 3: Training a deep learning model for cell type classiﬁcation
* Use the BYOL (Bootstrap Your Own Latent) algorithm to pretrain a deep learning model on the H&E images from CPTAC-COAD. The parameter setting is stored in `./cell_train/byol.yaml`.
* Run `./cell_train/cell_cls.py` to fine-tune the pretrained model on the H&E images with cell type labels using domain adaption strategy.

## Step 4: Single-cell spatial feature analysis
* Use the `./spatial_cell_analysis/main.py` script to analyze the spatial distribution of different cells in the tumor microenvironment. The script will output a .csv file containing the spatial features of cells.
* Use the `prognosis_analysis.py` script to perform survival analysis and response prediction to immune checkpoint inhibitors based on the spatial features of cells.

## Acknowledgments
The project was built on many open-source repositories such as Stardist (https://github.com/stardist/stardist), DeeperHistReg (https://github.com/MWod/DeeperHistReg), DAAN (https://github.com/zengjichuan/DANN), and solo-learn (https://github.com/vturrisi/solo-learn). We thank the authors and developers for their contributions.
## License
This code is made available under the GPLv3 License and is available for non-commercial academic purposes.
## Citation
If you find our work useful in your research, please consider citing:


