# ISANREG
**ISANREG** is an **I**nterpretable **S**elf-**A**ttention **N**etwork that uses block-attention and attention-attribution to learn **REG**ulatory features.

## Deependencies

* Python version = 3.8.8
* OS = Ubuntu 20.04.4 

### Python Libraries
* tensorflow (2.7.0) (Tensorflow dependencies recommended for the specific version is required for GPU support)

* numpy (1.23.1)

* matplotlib (3.4.2)

* seaborn (0.11.1)

* pandas (1.3.1)

* pyfaidx (0.6.2)

* scipy (1.8.1)


### Other dependencies
* FIMO (Find Individual Motif Occurrences) from MEME Suite (meme-5.4.1)

* Circos for plotting (0.69-9)

### Required Files
* Human reference genome (hg38.fa)

* Encode Exclusion files ("GRCh38_unified_blacklist.bed","dukeExcludeRegions.bed")

* TF Chip-seq processed narrowpeak bed file for training input data generation

* TF binding motif in Meme format from JASPAR

* Meme file of all Homo sapiens specific TF motifs from CisBP ("Homo_sapiens.meme")

## Scripts

**ISANREG** can be run on simulated inputs and in-vitro dataset derived inputs.

### Simulated data
The scripts for running the model on simulated data are:
```
simulated_input.py

--"Homo_sapiens.meme" file needed as input and outputs simulated training, validation and testing files.
```
```
simulated_train.py

--Trains the model on simulated training data.
```
```
simulated_test.py

--Testing the model after training. The high affinity motifs of both the TFs are given as input. The testing input "simulated_test2k.txt" and weights files are provided in the repository which can be directly used for testing the model.
```
```
simulated_distplot.py

--Generates Swarm plot of distance between the embedded motifs. The attention-attribution output file generated after testing on simulated sequences is given as input. The file "simtest_out.txt" is provided in the repository which can be directly used for plotting.
```
### In-vitro analysis
The scripts for running the model on in-vitro datasets are:
```
isanreg_dataprocess.py

--Creates training, validation and testing data for in-vitro datasets. Human reference genome (hg38.fa), Encode Exclusion files, TF Chip-seq processed narrowpeak bed file and TF binding motif in Meme format from JASPAR are given as input
```
```
isanreg_train.py

--Trains the model on TF specific training data.
```
```
isanreg_test.py

--Testing the model after training. Length of the TF motif is given as "--core_len" argument. Calculates the enrichment and significance of motifs from the attention-attribution data which helps in identifying interacting TFs. The input file for testing and weights file of the trained model for ESR1 is provided with the repository which can be used for testing.
```
```
isanreg_circos.py

--Creates input files needed for plotting enriched motifs of individual TFs according to circos requirements. The file "ESR1_out.txt" is provided in the repository which can be directly used for generating inputs for circos
--Circos .conf files have to be manually created for plotting.
```
```
isanreg_allplot.py

--Creates input files needed for plotting top enriched motifs of all the TFs according to circos requirements. 
--Circos .conf files and additional highlight file for validated interacting TFs have to be manually created for plotting.
```
## Usage

### Running the model

```
git clone https://github.com/anilprakash94/isanreg.git isanreg

cd isanreg

```
Then, run the programs according to the requirements and instructions listed in README.md.

For example:

python3 isanreg_dataprocess.py -h

```
usage: isanreg_dataprocess.py [-h] [--data_dir DATA_DIR] [-r REF_FASTA]
                              [--excl_files EXCL_FILES] [--seq_len SEQ_LEN]
                              [-i INPUT_FILE] [--tf_name TF_NAME]
                              [--tf_motif TF_MOTIF]

ISANREG Data processing

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   name of folder having data files
  -r REF_FASTA, --ref_fasta REF_FASTA
                        specify the reference genome fasta file
  --excl_files EXCL_FILES
                        path of bed files having exclusion regions
  --seq_len SEQ_LEN     total flanking sequence length for each input
  -i INPUT_FILE, --input_file INPUT_FILE
                        path of input file having chip_seq peak regions
  --tf_name TF_NAME     specify the name of the TF under study
  --tf_motif TF_MOTIF   specify the TF motif meme file

```

