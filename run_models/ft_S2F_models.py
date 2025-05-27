import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import models
from datetime import datetime
import torch.multiprocessing as mp
import numpy as np
from utils import load_data, calSpearman
from torch.distributed import destroy_process_group,init_process_group
import pandas as pd
from torch.nn.parallel import DistributedDataParallel as DDP
import json
import yaml
from argparse import ArgumentParser


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        gpu_id: int,
        config: dict,
        DDP_flag:bool
    ) -> None:
        self.gpu_id = gpu_id
        self.DDP_flag = DDP_flag
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.model = DDP(model, device_ids=[gpu_id],find_unused_parameters=True) if self.DDP_flag else self.model
        self.config = config
        self.save_root = self.config['model_save_folder']
        self.best_loss = np.inf
        self.gpu_num =  torch.cuda.device_count() 
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(config['learning_rate'])*self.gpu_num)
        self.warmup_epochs = int(0.1 * int(config['num_epochs']))
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=self.warmup_epochs)    
        self.cos_scheduler =  optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=int(config['num_epochs']) - self.warmup_epochs,eta_min=(float(config['learning_rate'])*self.gpu_num)/10)
        self.scheduler =  optim.lr_scheduler.SequentialLR(self.optimizer, [self.warmup_scheduler, self.cos_scheduler], milestones=[self.warmup_epochs])

    
 
    def _run_batch_single(self,ref_inputs,ref_targets):
        self.optimizer.zero_grad()
        outputs,_ = self.model(ref_inputs)
        loss = models.MyContrastiveLoss(outputs, ref_targets,alpha=0)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    def _run_batch_separate_allele(self, sample_inputs_1,sample_inputs_2,sample_targets):
        self.optimizer.zero_grad()
       
        indiv_outputs,_A,_B= self.model(sample_inputs_1,sample_inputs_2,both_allele=True)
        loss = models.MyContrastiveLoss(indiv_outputs, sample_targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch_single(self,epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)} | LR: {self.scheduler.get_last_lr()[0]:.8f}")
        if(self.DDP_flag):
            self.train_data.sampler.set_epoch(epoch)
        epoch_loss = 0.0
        progress_bar = tqdm(self.train_data, desc=f"Epoch {epoch + 1}")
        self.model.train()
        for inputs, targets, cur_sample_names, cur_gene in progress_bar:
            inputs = inputs.to(self.gpu_id).squeeze(1)
            targets = targets.to(self.gpu_id).view(-1,1)
            loss = self._run_batch_single(inputs,targets)
            progress_bar.set_postfix(loss=loss)
            epoch_loss += loss

        epoch_loss = epoch_loss/len(self.train_data)
        return epoch_loss




    def _run_epoch(self, epoch):

        b_sz = len(next(iter(self.train_data))[0])
        
        logging.info(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)} | LR: {self.scheduler.get_last_lr()[0]:.8f}")
        
        if(self.DDP_flag):
            self.train_data.sampler.set_epoch(epoch)
        epoch_loss = 0.0
        progress_bar = tqdm(self.train_data, desc=f"Epoch {epoch + 1}")
        self.model.train()

        for sample_inputs_1,sample_inputs_2, sample_targets, cur_gene,cur_sample_names in progress_bar:

            sample_inputs_1 = sample_inputs_1.to(self.gpu_id).squeeze(1)
            sample_inputs_2 = sample_inputs_2.to(self.gpu_id).squeeze(1)

            sample_targets = sample_targets.to(self.gpu_id).view(-1, 1)

            
            loss = self._run_batch_separate_allele(sample_inputs_1,sample_inputs_2, sample_targets)
            
            progress_bar.set_postfix(loss=loss)
            epoch_loss += loss

        epoch_loss = epoch_loss/len(self.train_data)
        return epoch_loss

    
    def _save_checkpoint(self, epoch,name='last'):
        if(self.DDP_flag):
            ckp = self.model.module.state_dict()
        else:
            ckp = self.model.state_dict()
        PATH = os.path.join(self.save_root,f"{name}.pt")
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

        config_path = os.path.join(self.save_root,'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def eval(self,all_genes=False):

        '''
        
        We do not use validation set, so we just randomly select some samples
        from the training set to evaluate the model.
        
        '''
        return
        
        if(not all_genes):
            selected_gene = ["ENSG00000171806"]
        else:
            selected_gene = []
        
        if(not hasattr(self, "val_gene_data")):
            '''
            self.val_gene_data = load_data(csv_file=config['partition_data_path'],seq_len=config['seq_len'],dataset_type=config["dataset_type"],\
                             target_name=config['target_name'],split='train', selected_gene=selected_gene,\
        selected_sample=config['selected_train_sample_list'],DDP=False,sample_allele=config["sample_allele"], batch_size=config["batch_size"],\
            sampler=config['sampler'],loadRaw=config["loadRaw"],ref_path=config["ref_genome_path"],consensus_root=config["consensus_root"])
            '''
            logging.info("Loading validation dataset")
            #print(config['selected_test_sample_list'])
            self.val_gene_data = load_data(csv_file=config['partition_data_path'],seq_len=config['seq_len'],dataset_type=config["dataset_type"],\
                             target_name=config['target_name'],split='test', selected_gene=selected_gene,\
        selected_sample=["ref"],DDP=False,sample_allele="NA", batch_size=config["batch_size"],\
            sampler=config['sampler'],loadRaw=config["loadRaw"],ref_path=config["ref_genome_path"],consensus_root=config["consensus_root"])

        results = {'targets':[],'predictions':[],'samples':[],'gene':[]}
        self.model.eval()
        with torch.no_grad():
            if(self.config["sample_allele"]=="both"):
                for sample_inputs_1, sample_inputs_2, sample_targets, cur_gene,sample_name in tqdm(self.val_gene_data, desc="Evaluating"):

                    sample_inputs_1 = sample_inputs_1.to(self.gpu_id).squeeze(1)
                    sample_inputs_2 = sample_inputs_2.to(self.gpu_id).squeeze(1)

                    sample_targets = sample_targets.to(self.gpu_id).view(-1, 1)

                    indiv_outputs,_A,_B= self.model(sample_inputs_1,sample_inputs_2,both_allele=True)

                    results['targets'].extend([x.item() for x in sample_targets.flatten()])
                    results['predictions'].extend( [x.item() for x in indiv_outputs.flatten().cpu()])
                    results['samples'].extend(list(sample_name))
                    results['gene'].extend(cur_gene)
            else:
                for sample_inputs, sample_targets, cur_gene,sample_name in tqdm(self.val_gene_data, desc="Evaluating"):

                    sample_inputs = sample_inputs.to(self.gpu_id).squeeze(1)
                    sample_targets = sample_targets.to(self.gpu_id).view(-1, 1)
                    indiv_outputs,_A= self.model(sample_inputs_1,both_allele=False)

                    results['targets'].extend([x.item() for x in sample_targets.flatten()])
                    results['predictions'].extend( [x.item() for x in indiv_outputs.flatten().cpu()])
                    results['samples'].extend(list(sample_name))
                    results['gene'].extend(cur_gene)



            
        test_eval_results = pd.DataFrame(results)
        #across_individual_res = calSpearman(test_eval_results, by_name='gene')
        across_gene_res = calSpearman(test_eval_results, by_name='samples')


    def train(self, max_epochs: int):
        if(self.gpu_id==0 or not self.DDP_flag):
            # evalue the model every epoch
            self.eval(all_genes=True)

        for epoch in range(max_epochs):
            self.model.train()

            if(self.config["sample_allele"]=="both" and self.config["train_data"]=="individuals"):
                epoch_loss = self._run_epoch(epoch)
            elif(self.config["train_data"]=="reference" or (self.config["train_data"]!="reference" and self.config["sample_allele"] != "both")):
                epoch_loss = self._run_epoch_single(epoch)

            if self.gpu_id == 0 or not self.DDP_flag:
                if(epoch == (max_epochs-1)):
                    self._save_checkpoint(epoch)
                elif(epoch_loss < self.best_loss):
                    self._save_checkpoint(epoch,name='best')

            if(self.gpu_id==0 or not self.DDP_flag):
                print(f"epoch {epoch}; loss: {epoch_loss}")
                
                self.eval()
            self.scheduler.step()



def initialize_model(config,device):

    if(config["model_name"]=="Enformer"):
        n_total_bins = 10
        model = models.FTEnformer(target_length=n_total_bins)
       
    elif(config["model_name"]=="Borzoi"):
        
        # we used embeddings from center 10 bins to predict the target gene expression.
        n_total_bins = 10
        model = models.borzoiModel(target_length=n_total_bins)
        
    if(config["pretrained_model_path"]!=""):
        pretrained_model_path = config["pretrained_model_path"]
        logging.info(f"Loading pretrained model from {pretrained_model_path}")
        model.load_state_dict(torch.load(pretrained_model_path,map_location="cpu"))
    
    model.to(device)
    return model
        

def ddp_setup(rank: int, world_size: int,MASTER_PORT="16663"):
   
   """
   Args:
      rank: Unique identifier of each process
      world_size: Total number of GPUs available for training,
        set the available GPUs by `export CUDA_VISIBLE_DEVICES={CUDA ID}`

   """
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = MASTER_PORT
   torch.cuda.set_device(rank)
   init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(rank,world_size,config):
   
   
    model = initialize_model(config=config,device=rank)
   
    if(world_size>1):
        ddp_setup(rank,world_size)
        DDP_flag = True
    else:
        DDP_flag= False
    
    logging.info("loading training dataset")
    train_loader = load_data(csv_file=config['partition_data_path'],seq_len=config['seq_len'],dataset_type=config["dataset_type"],\
                             target_name=config['target_name'],split='train', selected_gene=list(config['selected_gene_list']),\
        selected_sample=config['selected_train_sample_list'],DDP=DDP_flag,sample_allele=config["sample_allele"], batch_size=config["batch_size"],\
            sampler=config['sampler'],loadRaw=config["loadRaw"],ref_path=config["ref_genome_path"],consensus_root=config["consensus_root"])
    
    trainer = Trainer(model=model, train_data=train_loader, gpu_id=rank,config=config,DDP_flag=DDP_flag)
    trainer.train(config["num_epochs"])

    if(DDP_flag):
        destroy_process_group()   
    print("Training complete!")



def setupConfigs(config_file_path=None):

    # load the configuration file
    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)
        config["model_save_path"] = os.path.join(config["model_save_root"], config["model_name"]+"_"+datetime.now().strftime("%H-%M-%S"))

    config["loadRaw"] =False

    # set up the model folder with format of model_name+timestamp
    cur_model_folder = os.path.join(config["model_save_root"], config["model_name"]+"_"+datetime.now().strftime("%H-%M-%S"))
    if(not os.path.exists(cur_model_folder)):
        os.mkdir(cur_model_folder)
    
    config["model_save_folder"] = cur_model_folder
  
    # read selected genes
    config['selected_gene_list']  = []
    
    if(config["selected_gene"]==""):
        # if not set the selected gene subset file, we use all genes in the dataset
        config['selected_gene_list'] = []
    
    else:
        with open(config["selected_gene"], "r") as file:
            config['selected_gene_list'] = file.readlines()  

        config['selected_gene_list'] = [item.strip() for item in config['selected_gene_list']]

    config["dataset_type"] = "oneHot"

    if(config["train_data"]=="reference"):
        selected_train_sample_list = ['ref']

    elif(config["train_sample"]!=""):
        with open(config["train_sample"], 'r') as file:
            selected_train_sample_list = file.read().splitlines()
    
    else:
        selected_train_sample_list = []

    print("selected_train_sample_list",selected_train_sample_list)
    config["selected_train_sample_list"] = selected_train_sample_list

    if(config["test_sample"]!=""):
        with open(config["test_sample"], 'r') as file:
            selected_test_sample_list = file.read().splitlines()
    else:
        selected_test_sample_list = []
    config["selected_test_sample_list"] = selected_test_sample_list

    # save this configuration to the model folder
    config_path = os.path.join(cur_model_folder, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    return config
if __name__ == "__main__":

    parser = ArgumentParser(description="Fine-tune S2F models")
    parser.add_argument('--config_file_path', type=str,
                        help="The path of the configuration file",default="./configurations/config_Enformer_ref.yml")
    args = parser.parse_args()


    world_size = torch.cuda.device_count() 
    logging.info(f"world_size {world_size}")

    config = setupConfigs(args.config_file_path)

    if(world_size>1):
        mp.spawn(main,args=(world_size,config,),nprocs=world_size)
    else:
        main("cuda",1,config)

    
