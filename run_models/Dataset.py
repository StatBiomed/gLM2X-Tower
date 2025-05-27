import os
import torch
from random import shuffle
import pandas as pd
from Bio.Seq import Seq
import torch.nn.functional as F
import pyfaidx
from tqdm import tqdm
import random
from enformer_pytorch import str_to_one_hot
from torch.utils.data import Sampler,DistributedSampler

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GeneExpressionDataset(torch.utils.data.Dataset):

    def __init__(self,split,csv_file,seq_len,ref_path,consensus_root,\
    sample_allele=1,target_name='log_TPM',selected_gene=[],selected_samples=[],loadRaw=True,loadRef=False,nonCan="N"):
        '''
        split: train or test
        csv_file: path of partition CSV file
        seq_len: seq_length around the TSS
        ref_path: reference genome fasta file path
        consensus_root: consensus files of the individual genomes
        
        sample_allele: 1 or 2 represents the first or second allele, 'both' to load both alleles separately, \
        "NA" means load the single fasta but merged allele 1 and allele 2 as single fasta

        target_name: the gene expression labels, default as "log_TPM"
        selected_gene: selected gene ID list
        selected_samples: selected individual ID list
        loadRaw: load raw sequence or not 
        loadRef: load reference sequence or not
        nonCan: fill the non-canical bases as "N" as the default.

        '''

        cur_split_file = pd.read_csv(csv_file)
        #print("cur_split_file",cur_split_file)
        f1 = cur_split_file['split']==split
        f2 = cur_split_file['sample'] == 'ref'
        self.nonCan = nonCan
        
        self.loadRaw = loadRaw
        self.loadRef = loadRef

        self.target_name = target_name
        self.ref_df_split = cur_split_file[f1&f2].reset_index()
        self.ref_df = cur_split_file[f2].reset_index()
        #print(" self.ref_df_split", self.ref_df_split)

        if(len(selected_samples)>0):
            f3 = cur_split_file['sample'].isin(selected_samples)
        else:
            logging.info("use all samples for the current dataset")
            f3 = ~f2
        
        self.sample_df = cur_split_file[f1&f3].reset_index()
        #print("self.sample_df",self.sample_df)
        
        # reset_index 
        self.sample_df ['sample_gene'] = self.sample_df ['sample'] + '_' + self.sample_df ['gene']
        self.sample_df.set_index('sample_gene', inplace=True)

        self.sample_list = list(self.sample_df['sample'].unique())
        
        if(len(selected_gene)!=0):
            
            # filter the partition file by the selected gene list
            self.gene_list = sorted(selected_gene)
            self.ref_df = self.ref_df[self.ref_df['gene'].isin(self.gene_list)].reset_index()
            self.sample_df = self.sample_df[self.sample_df['gene'].isin(self.gene_list)].reset_index()
            self.ref_df_split = self.ref_df_split[self.ref_df_split['gene'].isin(self.gene_list)].reset_index()

        else:
            # use all genes listed in the partition file
            self.gene_list = sorted(list(self.sample_df['gene'].unique()))
            # use all genes
            #print("self.sample_df.shape",self.sample_df.shape)
        
        self.gene_df = self.ref_df.reset_index(drop=True).set_index('gene')
        self.consensus_root = consensus_root
        self.sample_fasta = dict()
        self.sample_allele = sample_allele
        
        # load individual's genomes
        if(len(self.sample_list)>=1):
            logging.info(f"loading sample fasta files for {len(self.sample_list)} individuals")
            self.sample_fasta = self.loadSampleFasta()
        
        # load reference genome
        self.sample_fasta['ref'] = [pyfaidx.Fasta(ref_path)]
        self.seq_len = seq_len
        self.canical_bases = ['A', 'C', 'G', 'T']

        # print out basic information for this dataset
        n_indivs = self.sample_df["sample"].unique().shape[0]
        logging.info(f"Dataset {split} loaded with {self.ref_df_split.shape[0]} genes and {n_indivs} individuals")

    def loadSampleFasta(self):

        '''

        return sample_chr fasta dictionary

        '''


        sample_fasta = dict()
        not_existed_samples = []

        for index, cur_sample in tqdm(enumerate(self.sample_list), total=len(self.sample_list),desc="Loading sample fasta files"):
            cur_index= cur_sample
            if(cur_sample == 'ref'):
                continue

            if(cur_index in sample_fasta):
                continue
            
            if(self.sample_allele == 1 or self.sample_allele ==2):
                cur_fn = [cur_index+'_allele_'+str(self.sample_allele)+'_forward.fa']
            elif(self.sample_allele=='NA'):
                cur_fn = [cur_index+'_all_allele_alt_forward.fa']
            elif(self.sample_allele=='both' ):
                cur_fn = [cur_index+'_allele_1_forward.fa',cur_index+'_allele_2_forward.fa']

            cur_fasta_data_list = []
         
            for temp_fn in cur_fn:
                cur_fasta_path = os.path.join(self.consensus_root,temp_fn)
                if(not os.path.exists(cur_fasta_path)):
                    not_existed_samples.append(cur_index)
                    continue
                curFastaData = pyfaidx.Fasta(cur_fasta_path)
                cur_fasta_data_list.append(curFastaData)
            sample_fasta[cur_index] = cur_fasta_data_list
        
        self.sample_df = self.sample_df[~self.sample_df['sample'].isin(not_existed_samples)]
        logging.info(f"sample file not found {not_existed_samples}")

        return sample_fasta
    


class DistributedGeneSampler(DistributedSampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.indices = list(range(len(data_source)))
        self.gene_to_indices = self._group_by_gene()

    def _group_by_gene(self):
        gene_to_indices = {}
        for idx in self.indices:
            gene = self.data_source.sample_df.iloc[idx]['gene']  # Get the gene
            if gene not in gene_to_indices:
                gene_to_indices[gene] = []
            gene_to_indices[gene].append(idx)
        return gene_to_indices

    def __iter__(self):
        all_indices = []
        all_genes = list(self.gene_to_indices.keys())
        random.shuffle(all_genes)
        
        #for gene, indices in self.gene_to_indices.items():
        for _,gene in enumerate(all_genes):
            indices = self.gene_to_indices[gene]
            random.shuffle(indices)
            all_indices.extend(indices)
        #print("gene sampler all_indices",len(all_indices),len(set(all_indices)))
        return iter(all_indices)

    def __len__(self):
        return len(self.data_source)


class GeneSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.indices = list(range(len(data_source)))
        self.gene_to_indices = self._group_by_gene()

    def _group_by_gene(self):
        gene_to_indices = {}
        for idx in self.indices:
            gene = self.data_source.sample_df.iloc[idx]['gene']  # Get the gene
            if gene not in gene_to_indices:
                gene_to_indices[gene] = []
            gene_to_indices[gene].append(idx)
        return gene_to_indices

    def __iter__(self):
        all_indices = []
        all_genes = list(self.gene_to_indices.keys())
        random.shuffle(all_genes)
        
        for _,gene in enumerate(all_genes):
            indices = self.gene_to_indices[gene]
            random.shuffle(indices)
            all_indices.extend(indices)
        return iter(all_indices)

    def __len__(self):
        return len(self.data_source)


class oneHotDataset(GeneExpressionDataset):
    '''

    return one-hot embeddings as the inputs for the S2F models: Enformer and Borzoi

    ''' 
    def __len__(self):
        return self.sample_df.shape[0]
    
    def encodeSeq(self,sequence,cur_strand):
       
        cur_sequence = ''.join([base if base in self.canical_bases else 'N' for base in sequence])
        if(cur_strand=='-'):
            cur_sequence = Seq(cur_sequence).reverse_complement()
            
        cur_emebdding = str_to_one_hot(str(cur_sequence))
        padding_size = (self.seq_len) - cur_emebdding.size(0)
 
        if (padding_size > 0):
            cur_emebdding = F.pad(cur_emebdding, (0, 0, 0, padding_size))
        return cur_emebdding

    def __getitem__(self,idx):
       
        row = self.sample_df.iloc[idx]
        
        cur_sample = row['sample']
        sample_target = row[self.target_name]
        cur_chr = row['chr']
        cur_TSS = row['TSS']
        cur_strand = row['strand']
        cur_gene = row['gene']

        cur_gene_chr = cur_chr[3:]
        cur_start = max(0,cur_TSS - int(self.seq_len/2))
        cur_end = cur_TSS + int(self.seq_len/2)
        cur_sample_fasta_file = self.sample_fasta[cur_sample]

        if(self.sample_allele!='both'):
              cur_sample_fasta_file = cur_sample_fasta_file[0]
              #print("cur_gene_chr",cur_gene_chr,"cur_start",cur_start,"cur_end",cur_end)
              try:
                original_cur_sample_seq =  str(cur_sample_fasta_file[cur_gene_chr][cur_start:cur_end+1].seq)
              except Exception as e:
                original_cur_sample_seq =  str(cur_sample_fasta_file[f"chr{cur_gene_chr}"][cur_start:cur_end+1].seq)

              cur_sample_emebdding = self.encodeSeq(original_cur_sample_seq,cur_strand)
              return cur_sample_emebdding, sample_target, cur_gene,cur_sample
        
        elif(self.sample_allele=='both'):
            # allele 1 information
            cur_sample_fasta_file_1 = cur_sample_fasta_file[0]
            original_cur_sample_seq_1 =  str(cur_sample_fasta_file_1[cur_gene_chr][cur_start:cur_end].seq)
            cur_sample_emebdding_1 = self.encodeSeq(original_cur_sample_seq_1,cur_strand)
       
            # allele 2 information
            cur_sample_fasta_file_2 = cur_sample_fasta_file[1]
            original_cur_sample_seq_2 =  str(cur_sample_fasta_file_2[cur_gene_chr][cur_start:cur_end].seq)
            cur_sample_emebdding_2 = self.encodeSeq(original_cur_sample_seq_2,cur_strand)
        
            return cur_sample_emebdding_1,cur_sample_emebdding_2,sample_target,cur_gene,cur_sample

class seqOnlyDataset(GeneExpressionDataset):

    def __len__(self):
        return self.sample_df.shape[0]
    
    def __getitem__(self,idx):
       

        row = self.sample_df.iloc[idx]
        cur_sample = row['sample']
        sample_target = row[self.target_name]
        cur_chr = row['chr']
        cur_TSS = row['TSS']
        cur_strand = row['strand']
        cur_gene = row['gene']

        cur_ref_fasta_file = self.sample_fasta['ref']
        cur_gene_chr = cur_chr[3:]
        cur_start = max(0,cur_TSS - int(self.seq_len/2))
        cur_end = cur_TSS + int(self.seq_len/2)

        cur_sample_fasta_file = self.sample_fasta[cur_sample]

        
        if(self.sample_allele=='NA' and self.loadRaw):
              cur_sample_fasta_file = cur_sample_fasta_file[0]
              original_cur_sample_seq =  str(cur_sample_fasta_file[cur_gene_chr][cur_start:cur_end+1].seq)
              if(cur_strand=='-'):
                # get the reverse complement of the sequence
                original_cur_sample_seq = str(Seq(original_cur_sample_seq).reverse_complement())
                
              return original_cur_sample_seq,sample_target,cur_gene,cur_sample
            
        elif(self.sample_allele=='both' and self.loadRaw):
             # if both allele, we generate two one-hot embeddings and get the average values.

            # allele 1 information
            cur_sample_fasta_file_1 = cur_sample_fasta_file[0]
            original_cur_sample_seq_1 =  str(cur_sample_fasta_file_1[cur_gene_chr][cur_start:cur_end].seq)
           

            original_cur_sample_seq_1 = ''.join([base if base in self.canical_bases else self.nonCan for base in original_cur_sample_seq_1])
       
            # allele 2 information
            cur_sample_fasta_file_2 = cur_sample_fasta_file[1]
            original_cur_sample_seq_2 =  str(cur_sample_fasta_file_2[cur_gene_chr][cur_start:cur_end].seq)
           
            original_cur_sample_seq_2 = ''.join([base if base in self.canical_bases else self.nonCan for base in original_cur_sample_seq_2])

        if(self.loadRaw):
            if(cur_strand=='-'):
                # get the reverse complement of the sequence
                original_cur_sample_seq_1 = str(Seq(original_cur_sample_seq_1).reverse_complement())
                original_cur_sample_seq_2 = str(Seq(original_cur_sample_seq_2).reverse_complement())
            
        else:
            original_cur_sample_seq_1 = ""
            original_cur_sample_seq_2 = ""

        return original_cur_sample_seq_1,original_cur_sample_seq_2,sample_target,cur_gene,cur_sample

