
from tqdm import tqdm
import torch
import os
import numpy as np
import pandas as pd
import models
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torch.optim as optim
import json
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForMaskedLM, AutoTokenizer
import yaml
from datetime import datetime
from utils import load_data
import psutil
import sys
from argparse import ArgumentParser
from utils import calSpearman

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        gpu_id: int,
        save_root:str,
        config:dict,
        DDP_flag:bool
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.DDP_flag = DDP_flag
        self.config = config
        self.model = DDP(model, device_ids=[self.gpu_id], find_unused_parameters=False) if self.DDP_flag else self.model
        self.save_root = save_root
        self.best_loss = np.inf
        self.gpu_num =  torch.cuda.device_count() 
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(self.config['learning_rate'])*self.gpu_num)
        self.warmup_epochs = int(0.1 * int(self.config['num_epochs']))
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=self.warmup_epochs)    
        self.cos_scheduler =  optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=int(self.config['num_epochs']) - self.warmup_epochs,eta_min=(float(self.config['learning_rate'])*self.gpu_num)/10)
        self.scheduler =  optim.lr_scheduler.SequentialLR(self.optimizer, [self.warmup_scheduler, self.cos_scheduler], milestones=[self.warmup_epochs])
        self.tensor_dict = dict()

    def _run_batch(self, sample_inputs_1, sample_inputs_2, sample_targets):
        self.optimizer.zero_grad()
        averaged_output = self.model(sample_inputs_1,sample_inputs_2)
        loss = models.MyContrastiveLoss(averaged_output, sample_targets)
        loss = loss.float()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    
    def _run_batch_single(self, sample_inputs_1, sample_targets):
        self.optimizer.zero_grad()
    
        averaged_output = self.model(sample_inputs_1)
        loss = models.MyContrastiveLoss(averaged_output, sample_targets,alpha=0)
        loss = loss.float()
        
        loss.backward()
        self.optimizer.step()
        return loss.item()
    

    def _get_caduceus_embed(self,sequences):

        '''
        Get the embeddings from the caduceus model
        '''

        if(not hasattr(self, "caduceus_model")):
            logging.info("Loading caduceus model")
            self.caduceus_name = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"
            self.caduceus_model = AutoModelForMaskedLM.from_pretrained(self.caduceus_name,trust_remote_code=True)
            self.caduceus_tokenizer = AutoTokenizer.from_pretrained(self.caduceus_name,trust_remote_code=True)
            self.caduceus_embedding_layer= -1
            self.caduceus_model.to(self.gpu_id)
            self.caduceus_model.eval()
        
        with torch.no_grad():
            #print("sequences",sequences)
            cur_input_ids = self.caduceus_tokenizer(sequences, return_tensors="pt")["input_ids"]
            outputs = self.caduceus_model(cur_input_ids.to(self.gpu_id), output_hidden_states=True)
            
            outputs = outputs.hidden_states[-1][:,:-1,:]
            #print("outputs",outputs)

        return outputs

    def _get_evo2_embed(self,sequences):
        
        '''
        Get the embeddings from the NT model
        '''
        if(not hasattr(self, "evo2_model")):
            from evo2 import Evo2
            self.evo2_model = Evo2('evo2_7b')
            self.evo2_embedding_layer_name= "blocks.28.mlp.l3"
        
        cur_input_ids = torch.tensor(self.evo2_model.tokenizer.tokenize(sequences),dtype=torch.int,).unsqueeze(0).to(self.gpu_id)
        cur_outputs, cur_embeddings = self.evo2_model(cur_input_ids, return_embeddings=True, layer_names=[self.evo2_embedding_layer_name])
        cur_embeddings = cur_embeddings[self.evo2_embedding_layer_name]
        return cur_embeddings


    def _get_NT_embed(self,sequences):
        '''
        Get the embeddings from the evo2 model
        '''
        import jax.numpy as jnp
        if(not hasattr(self, "NT_parameters")):
            from nucleotide_transformer.pretrained import get_pretrained_model
            import haiku as hk
            import jax
            

            model_version = "500M_multi_species_v2"
            self.random_key = jax.random.PRNGKey(0)

            self.NT_parameters, self.NT_forward_fn, self.tokenizer, self.NT_config = get_pretrained_model(model_name= model_version,embeddings_layers_to_save=(20,),max_positions=2000)
            self.NT_forward_fn = hk.transform(self.NT_forward_fn)
        cur_token_id = [b[1] for b in self.tokenizer.batch_tokenize([sequences])]
        cur_tokens = jnp.asarray(cur_token_id, dtype=jnp.int32)
        embeddings = torch.from_numpy(np.asarray(self.NT_forward_fn.apply(self.NT_parameters, self.random_key, cur_tokens)["embeddings_20"]).copy())
        
        return embeddings



    def _get_gLM_embed(self,seq_batch,sample_names,gene_list,allele):
        
        all_embedding = []
        for index,cur_seq in enumerate(seq_batch):

            cur_gene = gene_list[index]
            cur_sample = sample_names[index]
            cur_name = f"{sample_names[index]}_{cur_gene}_{allele}"
            
            if(cur_name in self.tensor_dict):
                embeddings = self.tensor_dict[cur_name]
                #print("loaded outputs",embeddings.shape)
                embeddings = embeddings.to(self.gpu_id)
            else:
                # load precalculated embeddings or do the inferences
                # if use the precalculated embeddings, we expect the embeddings are stored as the following format:
                '''
                -root folder
                    --sample_name (folder)
                        --{gene_ID}_{allele}_1.pt
                        --{gene_ID}_{allele}_2.pt
                
                '''
                # load from the pre-calculated embeddings
                load_flag = False
                if(self.config['preCalculatedEmbed_root']!=""):
                    # check if the embedding file exists
                    cur_embed_path = os.path.join(self.config["preCalculatedEmbed_root"],cur_sample,f"{cur_gene}_allele_{allele}.pt")
                    if(os.path.exists(cur_embed_path)):
                        embeddings = torch.load(cur_embed_path)
                        load_flag = True
                    else:
                        logging.warning(f"Pre-calculated embeddings not found at {cur_embed_path}, will calculate embeddings on-the-fly.")
                
                elif(not load_flag):
                    if(self.config['model_name'].startswith("caduceus")):
                        embeddings = self._get_caduceus_embed(cur_seq)
                        
                    elif(self.config['model_name'].startswith("NT")):
                        embeddings = self._get_NT_embed(cur_seq)
                        
                    elif(self.config['model_name'].startswith("evo2")):
                        embeddings = self._get_evo2_embed(cur_seq)

                    
                # save tensor to the memory to speed up the training
                size_embed = sys.getsizeof(embeddings)
                mem_size = psutil.virtual_memory().total
                # we will use 60% of the memory to save the tensor
                max_saved_num = (mem_size/(size_embed))*0.6
                # if the total size of the memory is less than 16GB, we will not save the tensor to the memory
                if(mem_size>16*1024*1024*1024):
                    cur_saved_num = len(self.tensor_dict)
                    if(cur_saved_num<max_saved_num):
                        self.tensor_dict[cur_name] = embeddings.cpu()
                    
            all_embedding.append(embeddings)
        
        all_embedding = torch.cat(all_embedding,dim=0).float().to(self.gpu_id)
        return all_embedding

       

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)} | LR: {self.scheduler.get_last_lr()[0]:.8f}")
        if(self.DDP_flag):
            self.train_data.sampler.set_epoch(epoch)
        epoch_loss = 0.0
        progress_bar = tqdm(self.train_data, desc=f"Epoch {epoch + 1}")
        for sample_seq_1,sample_seq_2, sample_targets, cur_gene, cur_sample_names in progress_bar:

            sample_targets = sample_targets.to(self.gpu_id).view(-1, 1)
            sample_inputs_1 = self._get_gLM_embed(sample_seq_1,cur_sample_names,cur_gene,allele=1)
            sample_inputs_2 =  self._get_gLM_embed(sample_seq_2,cur_sample_names,cur_gene,allele=2)
            loss = self._run_batch(sample_inputs_1, sample_inputs_2,sample_targets)
            progress_bar.set_postfix(loss=loss)
            epoch_loss += loss

        epoch_loss = epoch_loss/len(self.train_data)
        return epoch_loss

    def _run_epoch_single(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)} | LR: {self.scheduler.get_last_lr()[0]:.8f}")
        if(self.DDP_flag):
            self.train_data.sampler.set_epoch(epoch)
        epoch_loss = 0.0
        progress_bar = tqdm(self.train_data, desc=f"Epoch {epoch + 1}")

        for ref_seq, ref_targets, cur_gene,cur_sample_names in progress_bar:
            ref_targets = ref_targets.to(self.gpu_id).view(-1, 1)
            ref_inputs =  self._get_gLM_embed(ref_seq,cur_sample_names,cur_gene,allele="")
            loss = self._run_batch_single(ref_inputs,ref_targets)
            progress_bar.set_postfix(loss=loss)
            epoch_loss += loss

        epoch_loss = epoch_loss/len(self.train_data)
        return epoch_loss


    def _save_checkpoint(self, epoch,name='last'):
        #ckp = self.model.module.state_dict()
        if(not self.DDP_flag):
            ckp = self.model.state_dict()
        else:
            ckp = self.model.module.state_dict()

        PATH = os.path.join(self.save_root,f"{name}.pt")
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

        config_path = os.path.join(self.save_root,'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
   
    def eval(self,all_genes=False):


        if(not all_genes):
            selected_gene = ["ENSG00000171806"]
        else:
            selected_gene = []
        
        if(not hasattr(self, "val_gene_data")):
            
            self.val_gene_data = load_data(csv_file=self.config['partition_data_path'],seq_len=self.config['seq_len'],dataset_type=self.config["dataset_type"],\
                             target_name=self.config['target_name'],split='train', selected_gene=selected_gene,\
        selected_sample=self.config['selected_train_sample_list'],DDP=False,sample_allele=self.config["sample_allele"], batch_size=self.config["batch_size"],\
            sampler=self.config['sampler'],loadRaw=self.config["loadRaw"],nonCan = self.config["nonCan"], ref_path=self.config["ref_genome_path"],consensus_root=self.config["consensus_root"])
            
            logging.info("Loading validation dataset")
      
        results = {'targets':[],'predictions':[],'samples':[],'gene':[]}
        self.model.eval()

        with torch.no_grad():
            if(self.config["sample_allele"]=="both"):
                for sample_seq_1,sample_seq_2, sample_targets, cur_gene,cur_sample_names in tqdm(self.val_gene_data, desc="Evaluating"):
                
                    sample_targets = sample_targets.to(self.gpu_id).view(-1, 1)
                    sample_inputs_1 = self._get_gLM_embed(sample_seq_1,cur_sample_names,cur_gene,allele=1)
                    sample_inputs_2 =  self._get_gLM_embed(sample_seq_2,cur_sample_names,cur_gene,allele=2)

                    averaged_output = self.model(sample_inputs_1,sample_inputs_2)

                    results['targets'].extend([x.item() for x in sample_targets.flatten()])
                    results['predictions'].extend( [x.item() for x in averaged_output.flatten().cpu()])
                    results['samples'].extend(list(cur_sample_names))
                    results['gene'].extend(cur_gene)
            else:
                # do not eval when reference training, because it costs much time
                return

            
        test_eval_results = pd.DataFrame(results)
        index_data = calSpearman(test_eval_results, by_name='gene')
  

    def train(self, max_epochs: int):
        if(self.gpu_id==0 or (not self.DDP_flag)):
            
            #self.eval()
            pass
        for epoch in range(max_epochs):
            self.model.train()
            if(self.config["sample_allele"]=="both" and self.config["train_data"]=="individuals"):
                # one sample -> two haplotypes
                logging.info("run two haplotypes")
                epoch_loss = self._run_epoch(epoch)
            
            elif(self.config["train_data"]=="reference" or self.config["sample_allele"]!="both"):
                # one sample -> one haplotype
                epoch_loss = self._run_epoch_single(epoch)

            if self.gpu_id == 0 or (not self.DDP_flag):
                if(epoch == (max_epochs-1)):
                    self._save_checkpoint(epoch)
                elif(epoch_loss < self.best_loss):
                    self._save_checkpoint(epoch,name='best')

            if(self.gpu_id==0 or (not self.DDP_flag)):
                print(f"epoch {epoch}; loss: {epoch_loss}")
                
                self.eval()
            self.scheduler.step()



def setupConfigs(config_file_path):

    # load the configuration file
    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)
        config["model_save_path"] = os.path.join(config["model_save_root"], config["model_name"]+"_"+datetime.now().strftime("%H-%M-%S"))

    if(config["preCalculatedEmbed_root"]!=""):
        # do not need to load raw sequences if embedding is precalculated.
        loadRaw = False
    else:
        loadRaw = True
    
    config["loadRaw"] = loadRaw
    if(config["model_name"].startswith("NT")):
        config["nonCan"] = ""
    else:
        config["nonCan"] = "N"

    
    # set up the model folder with format of model_name+timestamp
    cur_model_folder = os.path.join(config["model_save_root"], config["model_name"]+"_"+datetime.now().strftime("%H-%M-%S"))
    if(not os.path.exists(cur_model_folder)):
        os.mkdir(cur_model_folder)
    config["model_save_folder"] = cur_model_folder
    if(config["selected_gene"]==""):
        # if not set the selected gene subset file, we use all genes in the dataset
        config['selected_gene_list'] = []
    
    else:
        with open(config["selected_gene"], "r") as file:
            config['selected_gene_list'] = file.readlines()  

        config['selected_gene_list'] = [item.strip() for item in config['selected_gene_list']]
    print(config["model_name"])
    if(config["model_name"].startswith("NT")):
        config["input_dim"] = 1024
    elif(config["model_name"].startswith("caduceus")):
        config["input_dim"] = 256
    elif(config["model_name"].startswith("evo2")):
        config["input_dim"] = 4096


    config["dataset_type"] = "seqOnly"

    if(config["train_data"]=="reference"):
        selected_train_sample_list = ['ref']
    elif(config["train_sample"]!=""):
        with open(config["train_sample"], 'r') as file:
            selected_train_sample_list = file.read().splitlines()
        #print("len(selected_train_sample_list)",len(selected_train_sample_list))
    
    else:
        selected_train_sample_list = []
    config["selected_train_sample_list"] = selected_train_sample_list

    if(config["test_sample"]!=""):
        with open(config["test_sample"], 'r') as file:
            selected_test_sample_list = file.read().splitlines()
        #print("len(selected_test_sample_list)",len(selected_test_sample_list))
    else:
        selected_test_sample_list = []
    config["selected_test_sample_list"] = selected_test_sample_list

    # save this configuration to the model folder
    config_path = os.path.join(cur_model_folder, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    return config



def main(rank,world_size,config):
   
    # Set device
    
    cnn_model = models.MultilayerCNN2(input_dim=config["input_dim"])

    #load ref pretrained: TODO: set about the loading the configuration files
    if(config["pretrained_model_path"]!=""):
        cnn_model.load_state_dict(torch.load(config["pretrained_model_path"],map_location="cpu"))

    if(world_size>1):
        ddp_setup(rank,world_size)
        DDP_flag = True
    else:
        DDP_flag = False


    train_loader = load_data(csv_file=config['partition_data_path'],seq_len=config['seq_len'],dataset_type=config["dataset_type"],\
                             target_name=config['target_name'],split='train', selected_gene=list(config['selected_gene_list']),\
        selected_sample=config['selected_train_sample_list'],DDP=DDP_flag,sample_allele=config["sample_allele"], batch_size=config["batch_size"],\
            sampler=config['sampler'],loadRaw=config["loadRaw"],nonCan = config["nonCan"], ref_path=config["ref_genome_path"],consensus_root=config["consensus_root"])
    
    
    trainer = Trainer(model=cnn_model, train_data=train_loader, gpu_id=rank, \
                      save_root=config['model_save_folder'],config=config,DDP_flag=DDP_flag)
    
    trainer.train(config["num_epochs"])


    # disable multiple GPU training
    if(DDP_flag):
        destroy_process_group()   
    print("Training complete!")


def ddp_setup(rank: int, world_size: int,MASTER_PORT="16663"):
   
   """
   Args:
      rank: Unique identifier of each process
      world_size: Total number of GPUs available for training,
        set the available GPUs by `export CUDA_VISIBLE_DEVICES=`

   """
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = MASTER_PORT
   torch.cuda.set_device(rank)
   init_process_group(backend="nccl", rank=rank, world_size=world_size)

if __name__ == "__main__":

    parser = ArgumentParser(description="Fine-tune S2F models")
    parser.add_argument('--config_file_path', type=str,
                        help="The path of the configuration file",default="config_gLM_tower.yml")
    args = parser.parse_args()

    
    world_size = torch.cuda.device_count()
    logging.info(f"world_size {world_size}")

    # read configurations from the yaml file
    config = setupConfigs(args.config_file_path)

    if(world_size>1):
        mp.spawn(main,args=(world_size,config,),nprocs=world_size)
    else:
        main("cuda",1,config)
    

