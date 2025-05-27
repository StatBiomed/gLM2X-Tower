#  🧬Assessing large-scale genomic language models in predicting personal gene expression: promises and limitations

This repo provides the implementation for our study assessing large-scale **genomic language models (gLMs)** and **sequence-to-function (S2F)** models in predicting **personal gene expression**.

We benchmarked:
- **S2F models**: Enformer, Borzoi  
- **gLMs**: Evo2-7B, Nucleotide Transformer v2 (NT), Caduceus-ph using a framework **gLM2X-Tower**.

##  📦 Installation

We recommend using a **separate conda environment** for each model, following the setup instructions from the corresponding original repositories:

- [Enformer](https://github.com/lucidrains/enformer-pytorch)
- [Borzoi](https://github.com/johahi/borzoi-pytorch)
- [Evo2](https://github.com/ArcInstitute/evo2)
- [NT](https://github.com/instadeepai/nucleotide-transformer)
- [Caduceus](https://github.com/kuleshov-group/caduceus)

And please install the following additional package for each environment:

```bash
pip install pyfaidx biopython pandas
```


## 📁 Dataset Preparation

### 🔹 Download Reference Genome

```

cd data
wget https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/phase2_reference_assembly_sequence/hs37d5.fa.gz

gunzip hs37d5.fa.gz

```

### 🔹 Prepare Partition File
```
cd data
gunzip all_partition_data.csv.gz

```

### 🔹 Generate Consensus of Individuals

* Step 1. Download VCF files and make them compressed and be indexed by `bcftools`

    ```
    cd data/vcf_files
    wget -i vcf_urls.txt
    ```

* Step 2. Make consensus fasta files for individuals
    - Install `bcftools` based on [installation](https://samtools.github.io/bcftools/howtos/install.html)
        or try `conda install -c conda-forge bcftools`
    - Prepare sample list file for generating consensus (e.g. `data/train_50_indivs.txt`)
    - Run `make_consensus.py`

        ```
        cd make_data

        python make_consensus.py --sample_list_file {sample name list file, default: ../data/train_50_indivs.txt}  --reference {reference path, default: ../data/hs37d5.fa} --vcf_root {vcf file root, default: ../data/vcf_files} --consensus_root {consensus save root, default: ../data/consensus}

        ```


## 🔧 Fine-tune S2F models

We support three training strategies:
- **r-**: Reference-only training
- **p-**: Individual-only training
- **rp-**: Reference pretraining followed by individual fine-tuning

We predefined the configuration files for each model, if using a custom data path, update it in the config YAML or create a new one.

### 🔹 Reference-Only Training

```
python ft_S2F_models.py --config_file_path configurations/config_{model_name}_ref.yml

# example
# python ft_S2F_models.py --config_file_path configurations/config_Enformer_ref.yml

```
### 🔹 Individual-Only Training

```

python ft_S2F_models.py --config_file_path configurations/config_{model_name}_indiv.yml

# example:
# python ft_S2F_models.py --config_file_path configurations/config_Enformer_indiv.yml

```

### 🔹 Reference-Individual Training

First run reference-only (r-) training, then fine-tune using individual-only configurations but setting `pretrained_model_path` of reference-trained model checkpoints in the config.


## 🧱 Train gLM2X-Tower models

The same three strategies apply to gLM-based models.

⚠️ For `rp-` models, you shall first train using reference-only (r-) and then fine-tune with individual data and setting `pretrained_model_path`.

### 🔹 Reference-only training
```

python train_gLM_based_models.py --config_file_path configurations/config_{model_name}_ref.yml

# example
# python train_gLM_based_models.py --config_file_path configurations/config_caduceus_ref.yml

```
### 🔹 Individual-only training

```

python train_gLM_based_models.py --config_file_path configurations/config_{model_name}_indiv.yml

# example
# python train_gLM_based_models.py --config_file_path configurations/config_caduceus_indiv.yml

```

### ⚠️ Evo2 Embeddings
Due to the high computational cost of Evo2 embedding generation, we recommend precomputing and storing embeddings as `.pt` files.

* Expected folder structure:

```
preCalculatedEmbed_root/
├── sample_name/
│   ├── {gene_ID}_{allele}_1.pt
│   └── {gene_ID}_{allele}_2.pt
      
 ```
Set the `preCalculatedEmbed_root` path in your configuration YAML.

