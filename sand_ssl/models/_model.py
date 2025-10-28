"""
Model class for hugging face models

Author(s): Daniela Wiepert
Last modified: 10/2025
"""

#IMPORT
##built-in
from collections import OrderedDict
import gc
from pathlib import Path
from typing import Union
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

##third-party
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoModel, WhisperModel

##local
from sand_ssl.constants import _MODELS

class Model(nn.Module):
    """
    Model class for hugging face models 

    :param out_dir: Pathlike, output directory to save to
    :param model_type: str, hugging face model type for naming purposes
    :param finetune_method: str, specify finetune method (default=None)
    :param freeze_method: str, freeze method for base pretrained model (default=required-only)
    :param unfreeze_layers: List[str], optionally give list of layers to unfreeze (default = None)
    :param pool_method: str, pooling method for base model output (default=mean)
    :param normalize: bool, specify whether to normalize input
    :param out_features: int, number of output features from classifier (number of classes) (default = 1)
    :param nlayers: int, number of layers in classification head (default = 2)
    :param activation: str, activation function to use in classification head (default = 'relu')
    :param bottleneck: int, optional bottleneck parameter (default=None)
    :param layernorm: bool, true for adding layer norm (default=False)
    :param dropout: float, dropout level (default = 0.0)
    :param binary:bool, specify whether output is making binary decisions (default=True)
    :param clf_type:str, specify layer type ['linear','transformer'] (default='linear')
    :param num_heads:int, number of encoder heads in using transformer build (default = 4)
    :param separate:bool, true if each feature gets a separate classifier head
    :param lora_rank: int, optional value when using LoRA - set rank (default = 8)
    :param lora_alpha: int, optional value when using LoRA - set alpha (default = 16)
    :param lora_dropout: float, optional value when using LoRA - set dropout (default = 0.0)
    :param virtual_tokens: int, optional value when using soft prompting - set num tokens (default = 4)
    :param seed: int, specify random seed for ensuring reproducibility (default = 42)
    :param device: torch device (default = cuda)
    :param from_hub: bool, specify whether to load from hub or from existing pt_ckpt (default = True)
    :param print_memory: bool, true if printing memory information
    :param bucket: gcs bucket (default = None)
    """
    def __init__(self, 
                    model_type:str, out_features:int=5, nlayers:int=2, activation:str='relu', bottleneck:int=None, layernorm:bool=False, dropout:float=0.0, binary:bool=False,
                    seed:int=42, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

        super(Model, self).__init__()

        self.model_type = model_type
        self.out_feats = out_features
        self.nlayers = nlayers
        self.activation = activation
        self.bottleneck = bottleneck
        self.layernorm = layernorm
        self.dropout = dropout
        self.binary = binary
        self.seed = seed
        self.device = device

        torch.manual_seed(self.seed)
        
        self.hf_hub = _MODELS[self.model_type]['hf_hub']
        
        ## INITIALIZE MODEL
        self._load_model_checkpoint(self.hf_hub)

        ## INITIALIZE Weighted sum 
        self.transformer_layers = _MODELS[self.model_type]['nlayers']
        self.weights = nn.Parameter(torch.rand(self.transformer_layers)) 

        ## INITIALIZE CLASSIFIER ARCHITECTURE
        self.in_feats = _MODELS[self.model_type]['in_features']
        if not self.bottleneck:
            self.bottleneck = self.in_feats
        self._initialize_classifier()

        ## SET UP CONFIG
        self.config = {'model_type':self.model_type, 'seed':self.seed, 'in_features':self.in_feats, 'out_features':self.out_feats, 
                        'nlayers':self.nlayers, 'activation':self.activation, 'binary': self.binary,
                        'bottleneck': self.bottleneck, 'layernorm':self.layernorm,'dropout':self.dropout}
        
    
    def get_model_name(self):
        """
        TODO
        """
        model_name = f'{self.model_type}_seed{self.seed}_in{self.in_feats}_out{self.out_feats}_{self.activation}_n{self.nlayers}'
        if self.bottleneck != self.in_feats:
            model_name += f'_bottleneck{self.bottleneck}'
        if self.dropout != 0.0:
            model_name += f'_dropout{self.dropout}'
        if self.layernorm:
            model_name += '_layernorm'
        if self.binary:
            model_name += '_binary'
        return model_name

    ### i/o functions (public and private) ###
    def _load_model_checkpoint(self, checkpoint:Union[str,Path]):
        """
        Load a model checkpoint

        :param checkpoint: pathlike, path to checkpoint (must be a directory)
        """
        print(f'Loading model {self.model_type} from Hugging Face Hub...')
        self.base_model = AutoModel.from_pretrained(checkpoint, output_hidden_states=True, trust_remote_code=True)       
        if not isinstance(self.base_model, _MODELS[self.model_type]['model_instance']):
            raise ValueError('Loaded model is not the expected model type. Please check that your checkpoint points to the correct model type.')
        self.is_whisper_model = isinstance(self.base_model, WhisperModel)
        self._freeze_all()
        
    def _freeze_all(self):
        """
        Freeze the entire model
        Will not need to be overwritten
        """
        for param in self.base_model.parameters():
            param.requires_grad = False 

    def _initialize_classifier(self):
        """
        TODO
        """
        self._params()
        model_dict = OrderedDict()
        for i in range(self.nlayers):
            model_dict[f'linear{i}'] = nn.Linear(self.params['in_feats'][i], self.params['out_feats'][i])
            if i+1 != self.nlayers:
                if self.layernorm:
                    model_dict[f'layernorm{i}'] = nn.LayerNorm(self.params['out_feats'][i])
                
                model_dict[f'{self.activation}{i}'] = self._get_activation_layer(self.activation)
                if self.dropout != 0.0:
                    model_dict[f'dropout{i}'] = nn.Dropout(self.dropout)
        if self.binary:
            model_dict['sigmoid'] = self._get_activation_layer('sigmoid')
        
        self.classifier_head = nn.Sequential(model_dict).apply(self._init_weights)

    def _params(self):
        """
        Get linear classifier parameters based on input parameters
        """
        self.params = {}
        in_feats = [self.in_feats]
        out_feats = [self.out_feats]
        
        for n in range(self.nlayers-1):
            in_feats.append(self.bottleneck)
            out_feats.insert(0, self.bottleneck)

        assert len(in_feats) == self.nlayers and len(out_feats) == self.nlayers, 'Classifier parameters must be same length as nlayers.'
        if len(in_feats) >= 2:
            assert in_feats[0] == self.in_feats and in_feats[-1] == self.bottleneck
            assert out_feats[0] == self.bottleneck and out_feats[-1] == self.out_feats

        self.params['in_feats'] = in_feats
        self.params['out_feats'] = out_feats      
        
    def _get_activation_layer(self, activation) -> nn.Module:
        """
        Create an activation layer based on specified activation function

        :return: nn.Module activation layer
        """
        if activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'relu':
            return nn.ReLU()
        else:
            raise NotImplementedError(f'{activation} not yet implemented.')

    def _init_weights(self, layer:nn.Module):
        """
        Randomize classifier weights

        :param layer: nn.Module, model layer
        """
        if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)

    def _downsample_attention_mask(self, attn_mask:torch.Tensor, target_len:int) -> torch.Tensor:
        """
        Downsample attention mask to target length

        :param attn_mask: torch.Tensor, attention mask
        :param target_len: int, target length to downsample 
        :return: downsampled attention mask
        """
        attn_mask = attn_mask.float().unsqueeze(1) # batch x 1 x time
        attn_mask = F.interpolate(attn_mask, size=target_len, mode="nearest")  # downsample
        return attn_mask.squeeze(1)

    def _pool(self, x:torch.Tensor, attn_mask:torch.Tensor=None) -> torch.Tensor:
        """
        Pooling function
        May need to change dimensions depending on model
        :param x: torch tensor, input
        """
        if attn_mask is not None:
            return x.sum(dim=1) / attn_mask.sum(dim=1).view(-1, 1)
        else:
            return torch.mean(x, 1)

    ### main function(s) ###
    def forward(self, inputs: torch.Tensor, attention_mask:torch.Tensor) -> torch.Tensor:
        """
        Overwritten forward loop. 

        :param sample: batched sample feature input
        :return: classifier output
        """
        if self.is_whisper_model:
            output = self.base_model.encoder(inputs, attention_mask=attention_mask)
        else:
            output = self.base_model(inputs, attention_mask=attention_mask)

        #get all hidden states
        hss = output['hidden_states']
        
        ds_attn_mask = self._downsample_attention_mask(attn_mask=attention_mask.to(torch.float16).detach(), target_len=hss[0].shape[1])
        expand_attn_mask = ds_attn_mask.unsqueeze(-1).repeat(1, 1, hss[0].shape[2])
        pooled_hss = []
        for hs in hss:
            hs[~(expand_attn_mask==1.0)] = 0.0
            pooled = self._pool(hs, ds_attn_mask.to(self.device))
            pooled_hss.append(pooled)
        
        pooled_hss = torch.stack(pooled_hss, dim=2)
        del ds_attn_mask

        normalized_weights = torch.softmax(self.weights, dim=0)
        weighted_sum = torch.matmul(pooled_hss, normalized_weights)

        clf_output = self.classifier_head(weighted_sum)
        del pooled 

        return clf_output