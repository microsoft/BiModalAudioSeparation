# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .laion_clap import CLAP_Module
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.distributed as dist

class CustomCLAP(torch.nn.Module):
    def __init__(self, enable_fusion=False, device=None, amodel= 'HTSAT-tiny', tmodel='roberta', channels=32):        
        super().__init__()

        self.model = CLAP_Module(enable_fusion=enable_fusion, amodel= amodel, device=device)
        self.model.load_ckpt(model_id=4, verbose=False)

        self.tokenize = self.model.tokenize

        self.logit_scale_a, self.logit_scale_t = self.model.model.get_logit_scale()

        for param in self.model.parameters():
            param.requires_grad = False

        self.logit_scale_a, self.logit_scale_t = self.logit_scale_a.detach(), self.logit_scale_t.detach()

        # self.mode = mode
        self.fc = nn.Linear(512, channels)
        self.fc.weight.data.normal_(0.0, 0.0001)
    
    def get_audio_embedding(self, audio):
        return self.model.get_audio_embedding_from_data(audio, use_tensor=True)

    def get_text_embedding(self, x, tokenize=False, return_hidden_states=False):
        if tokenize:
            x = self.model.tokenizer(x)
        
        if return_hidden_states:
            text_embed, hidden_states = self.model.model.get_text_embedding(x, return_hidden_states=return_hidden_states)        
            return text_embed, hidden_states
        
        else:
            text_embed = self.model.model.get_text_embedding(x)
            return text_embed
    
    def get_text_conditioning(self, text, tokenize=True, return_hidden_states=False):
        if return_hidden_states:
            text_embed, hidden_states = self.get_text_embedding(text, tokenize=tokenize, return_hidden_states=True)
            conds = self.fc(text_embed)
            return text_embed, hidden_states, conds
        else:
            text_embed = self.get_text_embedding(text, tokenize=tokenize)    
            conds = self.fc(text_embed)
            return text_embed, conds

    def classification_loss(self, audio_embed, text_embed, labels):
        device = audio_embed.device
        try:
            logits_per_audio = self.logit_scale_a.to(device) * audio_embed @ text_embed.T
        except:
            print("Error")

        return F.cross_entropy(logits_per_audio, labels)

    def contrastive_loss(self, audio_embed, text_embed):
        device = audio_embed.device
        logits_per_audio = self.logit_scale_a * audio_embed @ text_embed.T
        num_logits = logits_per_audio.shape[0]
        labels = torch.arange(num_logits, device=device, dtype=torch.long)

        return F.cross_entropy(logits_per_audio, labels)