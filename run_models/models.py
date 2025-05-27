import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from enformer_pytorch import from_pretrained
from borzoi_pytorch import Borzoi
huber_loss = nn.SmoothL1Loss()


def MyContrastiveLoss(outputs, targets,alpha=0.3,margin=0.1,reduction='mean'):
    # Standard MSE loss for individual predictions
    
    mse_loss = huber_loss(outputs, targets)
        
    # Pairwise difference loss
    batch_size = outputs.size(0)
    pairwise_loss = 0.0
    valid_pairs = 0
    
    for i in range(batch_size):
        for j in range(i+1, batch_size):
            # Skip pairs from the same individual
            if(i ==j):
                continue
            # Ground-truth and predicted differences
            true_diff = torch.abs(targets[i] - targets[j])  # |y_i - y_j|
            pred_diff = torch.abs(outputs[i] - outputs[j])  # |ŷ_i - ŷ_j|
                
            # Loss term: Penalize if predicted difference deviates from truth
            pairwise_loss += huber_loss(pred_diff, true_diff)
            valid_pairs += 1
        
    if valid_pairs == 0:
        return mse_loss
        
    pairwise_loss /= valid_pairs  # Normalize
    total_loss = (1 - alpha) * mse_loss + alpha * pairwise_loss
    return total_loss.float()



class AttentionPool2(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.attention = nn.Linear(embedding_dim, 1)

    def forward(self, X):
        """
        X shape: (..., tokens, embedding_dim)
        """
       
        attn_scores = self.attention(X)  # (..., tokens, 1)
        attn_weights = F.softmax(attn_scores, dim=-2)  # (..., tokens, 1)
        output = (attn_weights * X).sum(dim=-2)  # (..., embedding_dim)
        return output


class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size = 2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange('b d (n p) -> b d n p', p = pool_size)

        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias = False)

        nn.init.dirac_(self.to_attn_logits.weight)

        with torch.no_grad():
            self.to_attn_logits.weight.mul_(2)

    def forward(self, x):
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = F.pad(x, (0, remainder), value = 0)
            mask = torch.zeros((b, 1, n), dtype = torch.bool, device = x.device)
            mask = F.pad(mask, (0, remainder), value = True)

        x = self.pool_fn(x)
        logits = self.to_attn_logits(x)

        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)

        attn = logits.softmax(dim = -1)

        return (x * attn).sum(dim = -1)




class borzoiModel(nn.Module):

    def __init__(self, target_length:int=10):
        super().__init__()
        self.base = Borzoi.from_pretrained('johahi/borzoi-replicate-0')
        self.base.crop.target_length = target_length
        self.hidden_dim = 1920
        self.hidden_dim_2 = 1536
        self.attention_pool = AttentionPool2(self.hidden_dim)
        self.prediction_head = nn.Linear(self.hidden_dim, 1)

    
    def forward(self,ref_X,ref_Y=None,both_allele=False):
        
        ref_X = ref_X.permute(0, 2, 1)
        ref_X_preds = self.base(ref_X,is_human=True)
        x_embed = self.base.get_embs_after_crop(ref_X)
        x_seq_embs  = self.base.final_joined_convs(x_embed).permute(0,2,1)

        ref_X_pool = self.attention_pool(x_seq_embs)
        
        x_embed = x_embed.permute(0,2,1)
        
        Y = self.prediction_head(ref_X_pool)


        if(ref_Y==None):
            return Y,x_embed
       
        # return output and embedding for pairs, specially for parallel training
        ref_Y = ref_Y.permute(0, 2, 1)
        ref_Y_preds = self.base(ref_Y,is_human=True)
        y_embed = self.base.get_embs_after_crop(ref_Y)
        y_seq_embs  = self.base.final_joined_convs(y_embed).permute(0,2,1)

        ref_y_pool = self.attention_pool(y_seq_embs)
        
        y_embed = x_embed.permute(0,2,1)
        
        Y_2 = self.prediction_head(ref_y_pool)


        if(both_allele):
            outputs = torch.stack([Y, Y_2], dim=2)  # shape: (batch, targets, 2)
            averaged_output = torch.mean(outputs, dim=2)
            return averaged_output,x_embed,y_embed


    
     
class FTEnformer(nn.Module):

    def __init__(self,target_length:int=10):
        super(FTEnformer, self).__init__()
        self.target_length = target_length
        self.base = from_pretrained(
                "EleutherAI/enformer-official-rough",target_length=target_length,use_tf_gamma=False
            )
        
    
        enformer_hidden_dim = self.base.dim*2


        self.attention_pool = AttentionPool2(enformer_hidden_dim)
        self.prediction_head = nn.Linear(enformer_hidden_dim, 1)

        
    def forward(self,ref_X,indiv_X=None,both_allele=False):
      
        ref_X_embedding = self.base(ref_X,
                return_only_embeddings=True,
                target_length=self.target_length,
            )
        
        ref_X_pool = self.attention_pool(ref_X_embedding)
        Y = self.prediction_head(ref_X_pool)

        if(indiv_X==None):
            return Y,ref_X_embedding
       
        # return output and embedding for pairs, specially for parallel training
        indiv_X_embedding = self.base(indiv_X,
                return_only_embeddings=True,
                target_length=self.target_length,
            )
        indiv_X_pool = self.attention_pool(indiv_X_embedding)
        indiv_Y = self.prediction_head(indiv_X_pool)
        if(both_allele):
            outputs = torch.stack([Y, indiv_Y], dim=2)  # shape: (batch, targets, 2)
            averaged_output = torch.mean(outputs, dim=2)
            return averaged_output,ref_X_embedding,indiv_X_embedding

        return Y,ref_X_embedding, indiv_Y,indiv_X_embedding


class MultilayerCNN2(nn.Module):
    def __init__(self, input_dim=4096, seq_len=64000,hidden_dim=512):
        super().__init__()
        self.blocks = nn.ModuleList([
            self._make_ds_block(input_dim, input_dim, kernel_size=16, stride=8, padding=7, pool_kernel=2),
           
            self._make_ds_block(input_dim, input_dim, kernel_size=8, stride=4, padding=3, pool_kernel=2),
            self._make_ds_block(input_dim, input_dim, kernel_size=4, stride=2, padding=1, pool_kernel=2),
            self._make_ds_block(input_dim, input_dim, kernel_size=3, stride=2, padding=1, pool_kernel=1),
        ])
    
        self.initial_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),  # Pointwise convolution
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        self.pool =  nn.AdaptiveAvgPool1d(1)
    
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 258),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(258, 1)
        )

    def _make_ds_block(self, in_channels, out_channels, kernel_size, stride, padding, pool_kernel):
        """Depthwise separable convolution block."""
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv1d(
                in_channels, in_channels, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding, 
                groups=in_channels  # Depthwise
            ),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            # Pointwise convolution
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            # Optional pooling
            AttentionPool(out_channels,pool_size=pool_kernel) if pool_kernel >1 else nn.Identity()
        )

    def forward(self, x, y=None):
        x = x.permute(0, 2, 1)         
        for block in self.blocks:
            x = block(x)
        x = self.initial_conv(x)
        x = x.squeeze(1)
        x = self.pool(x)  
        x = x.squeeze(-1)
        x = self.classifier(x)

        if(y!=None):
            y = y.permute(0, 2, 1)  
        
            for block in self.blocks:
                y = block(y)
            y = self.initial_conv(y)
           
            y = self.pool(y)  
            y = y.squeeze(-1)     
        
            y = self.classifier(y)
            
            outputs = torch.stack([x, y], dim=2)  # shape: (batch, targets, 2)
            averaged_output = torch.mean(outputs, dim=2)
            return averaged_output
    
        return x
