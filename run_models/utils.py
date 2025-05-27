
import numpy as np
from scipy.stats import spearmanr
import pandas as pd
from random import shuffle
from torch.utils.data import DataLoader,DistributedSampler
import Dataset

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def calSpearman(results, by_name='samples'):

    index_data = {'coefficient':[],'p-value':[],by_name:[]}
    for groupName, groupData in results.groupby(by=by_name):
        truth_values = groupData['targets'].values
        prediction_values =  groupData['predictions'].values
       
        correlation, p_value = spearmanr(truth_values,prediction_values)
        index_data['coefficient'].append(correlation)
        index_data['p-value'].append(p_value)
        index_data[by_name].append(groupName)
    
    index_data = pd.DataFrame(index_data)
    logging.info(f"\n{index_data}")

    return index_data

def load_data(csv_file,seq_len,target_name,batch_size,\
split,dataset_type, ref_path=None, consensus_root=None, selected_gene=[],selected_sample=[],DDP=False,sample_allele="NA",\
    loadRaw=True,nonCan="N",sampler='random'):

    if(dataset_type=='seqOnly'):
        logging.info("Loading sequence only dataset")
        dataset = Dataset.seqOnlyDataset(split=split, seq_len=seq_len,
                                                  csv_file=csv_file, target_name=target_name,selected_gene=selected_gene, selected_samples=selected_sample,\
                                                    sample_allele=sample_allele,loadRaw=loadRaw,nonCan=nonCan,ref_path=ref_path,consensus_root=consensus_root)
    
    elif(dataset_type=='oneHot'):
        logging.info("Loading one-hot encoded dataset")
        dataset = Dataset.oneHotDataset(split=split, seq_len=seq_len,
                                                  csv_file=csv_file, target_name=target_name,selected_gene=selected_gene, selected_samples=selected_sample,\
                                                    sample_allele=sample_allele,loadRaw=loadRaw,nonCan=nonCan,ref_path=ref_path,consensus_root=consensus_root)

    if(DDP):
        gene_sampler = Dataset.GeneSampler(dataset)
        if(sampler == 'random'):
            return DataLoader(dataset, batch_size=batch_size,sampler=DistributedSampler(dataset))
        elif(sampler == 'gene'):
            return DataLoader(dataset, batch_size=batch_size,sampler=DistributedSampler(gene_sampler))
    
    if(sampler=='random'):
        return DataLoader(dataset, batch_size=batch_size,shuffle=True)
    elif(sampler=='gene'):
        gene_sampler = Dataset.GeneSampler(dataset)
        return DataLoader(dataset, batch_size=batch_size,sampler=gene_sampler)