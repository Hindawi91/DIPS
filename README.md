# Domain-knowledge Inspired Pseudo Supervision (DIPS)

This repository provides the official implementation of our paper soon to be published in the Engineering Applications of Artificial Intelligence Journal titled:<br/>  _**"Domain-knowledge Inspired Pseudo Supervision (DIPS)
for Unsupervised Image-to-Image Translation Models to Support Cross-Domain Classification"**_


![Pseudo Metric Entire Framework](https://github.com/Hindawi91/Pseudo_Supervised_Metrics/assets/38744510/e5ea3bc3-3208-45f7-a85b-b3659fa4f96d)


## Paper
[**Pseudo Supervised Metrics:<br/>Domain-knowledge Inspired Pseudo Supervision (DIPS) for Unsupervised Image-to-Image Translation Models to Support Cross-Domain Classification**](https://arxiv.org/abs/2303.10310)

[Firas Al-Hindawi](https://firashindawi.com)<sup>1</sup>,[Md Mahfuzur Rahman Siddiquee](https://github.com/mahfuzmohammad)<sup>1</sup>, [Teresa Wu]<sup>1</sup>, [Han Hu](https://scholar.google.com/citations?user=5RgSI9EAAAAJ&hl=en)<sup>2</sup>, [Ying Sun]<sup>3</sup><br/>

<sup>1</sup>Arizona State University; <sup>2</sup>University of Arkansas; <sup>3</sup>University of Cincinnati<br/>


## Abstract

The ability to classify images accurately and efficiently is dependent on having access to large labeled datasets and testing on data from the same domain that the model can train on. Classification becomes more challenging when dealing with new data from a different domain, where gathering and especially labeling a larger image dataset for retraining a classification model requires a labor-intensive human effort. Cross-domain classification frameworks were developed to handle this data domain shift problem by utilizing unsupervised image-to-image translation models to translate an input image from the unlabeled domain to the labeled domain. The problem with these unsupervised models lies in their unsupervised nature. For lack of annotations, it is not possible to use the traditional supervised metrics to evaluate these translation models to pick the best-saved checkpoint model. This paper introduces a new method called Pseudo Supervised Metrics that was designed specifically to support cross-domain classification applications contrary to other typically used metrics such as the Fréchet Inception Distance (FID) which were designed to evaluate the model in terms of the quality of the generated image from a human-eye perspective. The introduced metric is shown to outperform state-of-the-art unsupervised metrics such as the FID. Furthermore, it is also shown to be highly correlated with the true supervised metrics, making it both robust and explainable. As a mechanism for the validity of the proposed metric, the problem of the boiling crisis has been approached.

## Usage

### How to replicate our experiments:

1- clone the repository <br />
2- download the ["data"](https://www.dropbox.com/sh/znvoo3t103bd8of/AABrXaEr5_BzBgcgn6r8gwjQa?dl=0) folder and place it inside the repository. The data folder includes the original (non-translated) data and the 30 translated datasets <br />
3- download the ["CNN_Base_Models"](https://www.dropbox.com/sh/3wcnm07h7gtxh6j/AAAsKPFAaObW4kQpmucRFjRfa?dl=0) folder and place it inside the repository <br />
4- run the paper_results.py file to generate the PSM results from the downloaded data and using the saved model. (it will automatically run the DS1 --> DS2 Experiment, change the values of the variables in the beginning of the code file to run the DS2 --> DS1 Experiment)

### How to use the entire workflow using your own dataset:

#### Data Preparation:

<ol type="1">
   <li> Divide Source Dataset into 3 portions as so:
       <ol type="a">
       <li>training
         <ol type="i">
           <li>source_DS_directory/train/class0</li>
           <li>source_DS_directory/train/class1</li>
         </ol>
       </li>
       <li>validation<ol type="i">
           <li>source_DS_directory/val/class0</li>
           <li>source_DS_directory/val/class1</li>
         </ol>
       </li>
       <li>testing<ol type="i">
           <li>source_DS_directory/test/class0</li>
           <li>source_DS_directory/test/class1</li>
         </ol>
       </li>
       </ol>
   </li>
   <li>Divide Target Dataset into 3 portions as so:<ol type="a">
       <li>training
         <ol type="i">
           <li>target_DS_directory/train/class0</li>
           <li>target_DS_directory/train/class1</li>
         </ol>
       </li>
       <li>validation<ol type="i">
           <li>target_DS_directory/val/class0</li>
           <li>target_DS_directory/val/class1</li>
         </ol>
       </li>
       <li>testing<ol type="i">
           <li>target_DS_directory/test/class0</li>
           <li>target_DS_directory/test/class1</li>
         </ol>
       </li>
       </ol>
   </li>
</ol>


#### CNN Training:

<ol type="1">
  <li>Go to the CNN training folder</li>
  <li>In the “DS_CNN_Training.py” file, change the “dataset” variable to the source DS directory, then run the python script.</li>
  <li>Once training is done, the best model would be saved as “CNN - Base Model.hdf5”</li>
  <li>In the “test_DS_on_DS.py” file, change the “dataset” variable to the source DS directory, then run the python script. Then run the python script to test the saved model on the source dataset for sanity check.</li>
</ol>

#### FPGan Training:

<ol type="1">
  <li>Go to the FPGAN training folder</li>
  <li>In the “1_get_base_data.py” file, change the “base_directory” variable to the training portion of your source DS directory (e.g. “source_DS_directory/train/”), then run the python script.</li>
  <li>In the “2_get_target_data.py” file, change the “target_directory” variable to your target DS directory (e.g. “target_DS_directory/”), then run the python script.</li>
  <li>Run the “3-run.sh” to start the FP-GAN training.</li>
  <li>Once the model finishes training, run the “4-val.sh” to translate the validation target DS using every checkpoint model that was saved after 10k iterations.</li>
</ol>

#### Calculate metrics:

<ol type="1">
  <li>To calculate PSM, In the “calculate_PSM.py":
      <ol type="a">
          <li>Change the “pseudo_metric” to either "Balanced Accuracy" or "AUC" depending on your preference.</li>
          <li>Run the python script</li>
          <li>PSM results are saved in “PSM_Results.xlsx”</li>
      </ol>
  </li>


  
  <li>To calculate FID, In the “calculate_FID.py:
      <ol type="a">
          <li>Run the python script</li>
          <li>FID results are saved in “FID_Results.xlsx”</li>
      </ol>
  </li>



  
  <li>To calculate the true metrics (assuming you have the true labels):
      <ol type="a">
          <li>You have to separate class0 from class1 in each translated data folder in “./FPGAN_Training/brats_syn_256_lambda0.1/results_{i}”, where i = 10000, 20000, … , 300,000.</li>
          <li>You could use the separate_val, but you have to change the code based on how you named the data … In our case we separate data using the file name, if the file has “class0” in the name then move to class0 folder, otherwise move to class1 folder.</li>
        <li>Once the data is separated, run the “calculate_true.py” script.</li>
        <li>You could use the “reverse_seperate_val.py” to reverse the separation if needed.</li>
      </ol>
  </li>
</ol>

## Citation

If you use this code for your research, please cite the following papers:

```
@article{al2023pseudo,
  title={Pseudo Supervised Metrics: Evaluating Unsupervised Image to Image Translation Models In Unsupervised Cross-Domain Classification Frameworks},
  author={Al-Hindawi, Firas and Siddiquee, Md Mahfuzur Rahman and Wu, Teresa and Hu, Han and Sun, Ying},
  journal={arXiv preprint arXiv:2303.10310},
  year={2023}
}

@article{al2023framework,
  title={A framework for generalizing critical heat flux detection models using unsupervised image-to-image translation},
  author={Al-Hindawi, Firas and Soori, Tejaswi and Hu, Han and Siddiquee, Md Mahfuzur Rahman and Yoon, Hyunsoo and Wu, Teresa and Sun, Ying},
  journal={Expert Systems with Applications},
  volume={227},
  pages={120265},
  year={2023},
  publisher={Elsevier}
}

```

## Acknowledgments

We thank the authors of [Fixed-Point-GAN](https://github.com/mahfuzmohammad/Fixed-Point-GAN) since we relied heavily on their code, and we also thank [pytorch-fid](https://github.com/mseitzer/pytorch-fid) for FID computation.






