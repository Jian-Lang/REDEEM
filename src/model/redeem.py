import torch
from torch import nn
import torch.nn.functional as F
from .modules import CMoEGenerator, SAIMPrompter
from .vilt import ViltModel


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class redeem(torch.nn.Module):
    def __init__(self,
                 vilt: ViltModel,
                 dataset_name: str,
                 max_text_len: int,
                 max_image_len: int,
                 missing_type: str,
                 device: str,
                 prompt_pos: int,
                 prompt_len: int,
                 dropout_rate: float,
                 k: int,
                 hs=768,
                 recon=True,
                 **kargs):
        super(redeem, self).__init__()
        self.device = device
        self.max_text_len = max_text_len
        self.max_image_len = max_image_len
        self.missing_type = missing_type
        self.dataset_name = dataset_name
        self.recon = recon
        self.embedding_layer = vilt.embeddings
        self.encoder_layer = vilt.encoder.layer
        self.layernorm = vilt.layernorm
        self.prompt_len = prompt_len
        self.prompt_pos = prompt_pos
        self.hs = hs
        if dataset_name == "hatememes":
            cls_num = 2
        elif dataset_name == "food101":
            cls_num = 101
        elif dataset_name == "mmimdb":
            cls_num = 23
        # freeze the pretrained multi-modal transformer
        self.freeze()

        # define training component
        self.pooler = vilt.pooler

        # define the filter for completion of missing information
        if missing_type == "text":
            self.generator = CMoEGenerator(k=k)
        elif missing_type == "image":
            self.generator = CMoEGenerator(k=k)
        elif missing_type == "both":
            self.generator_t = CMoEGenerator(k=k)
            self.generator_i = CMoEGenerator(k=k)

        # define the SAIMPrompter prompt
        self.prompter = SAIMPrompter(prompt_len=prompt_len, dim=hs)

        self.label_enhanced = nn.Parameter(torch.randn(cls_num, hs))
        self.classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, hs),
            )
        self.classifier.apply(init_weights)

    def freeze(self):
        for param in self.embedding_layer.parameters():
            param.requires_grad = False
        for param in self.encoder_layer.parameters():
            param.requires_grad = False
        for param in self.layernorm.parameters():
            param.requires_grad = False

    def forward(self,
                input_ids: torch.Tensor,
                pixel_values: torch.Tensor,
                pixel_mask: torch.Tensor,
                token_type_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                r_t_list: torch.Tensor, 
                r_i_list: torch.Tensor,
                r_l_list: torch.Tensor,
                missing_mask: torch.Tensor,
                image_token_type_idx=1):
        embedding, attention_mask = self.embedding_layer(input_ids=input_ids,
                                                         attention_mask=attention_mask,
                                                         token_type_ids=token_type_ids,
                                                         inputs_embeds=None,
                                                         image_embeds=None,
                                                         pixel_values=pixel_values,
                                                         pixel_mask=pixel_mask,
                                                         image_token_type_idx=image_token_type_idx)
        text_emb = embedding[:, :self.max_text_len, :]
        image_emb = embedding[:, self.max_text_len:, :]
        recon_loss = 0
        if self.missing_type == "text":
            padding = self.generator(image_emb, r_t_list)
            missing_mask_t = missing_mask.view(-1, 1, 1).expand(-1,self.max_text_len, self.hs)

            text_emb = text_emb * missing_mask_t + padding * (1-missing_mask_t)
            if self.recon:
                mask = (1-missing_mask).view(-1, 1, 1).expand(-1, self.max_text_len, self.hs)
                recon_loss = F.mse_loss(text_emb, padding, reduction='none')
                recon_loss = recon_loss * mask
                recon_loss = recon_loss.mean()

        elif self.missing_type == "image":
            padding = self.generator(text_emb, r_i_list)
            missing_mask_i = missing_mask.view(-1, 1, 1).expand(-1, self.max_image_len, self.hs)
            image_emb = image_emb * missing_mask_i + padding * (1-missing_mask_i)
            if self.recon:
                # Only compute reconstruction loss for missing positions
                mask = (1-missing_mask).view(-1, 1, 1).expand(-1, self.max_image_len, self.hs)
                recon_loss = F.mse_loss(image_emb, padding, reduction='none')
                recon_loss = recon_loss * mask
                recon_loss = recon_loss.mean()

        elif self.missing_type == "both":
            padding_t = self.generator_t(image_emb, r_t_list)
            padding_i = self.generator_i(text_emb, r_i_list)
            t_missing_mask = [0 if i == 0 else 1 for i in missing_mask]
            i_missing_mask = [0 if i == 1 else 1 for i in missing_mask]
            t_missing_mask = torch.tensor(t_missing_mask).to(self.device)
            i_missing_mask = torch.tensor(i_missing_mask).to(self.device)
            missing_mask_t = t_missing_mask.view(-1, 1, 1).expand(-1,self.max_text_len, self.hs)
            missing_mask_i = i_missing_mask.view(-1, 1, 1).expand(-1, self.max_image_len, self.hs)
            text_emb = text_emb * missing_mask_t + padding_t * (1-missing_mask_t)
            image_emb = image_emb * missing_mask_i + padding_i * (1-missing_mask_i)
            if self.recon:
                # Only compute reconstruction loss for missing positions
                mask = (1-missing_mask).view(-1, 1, 1).expand(-1, self.max_text_len, self.hs)
                recon_loss_t = F.mse_loss(text_emb, padding_t, reduction='none')
                recon_loss_t = recon_loss_t * mask
                recon_loss_t = recon_loss_t.mean()
                recon_loss_i = F.mse_loss(image_emb, padding_i, reduction='none')
                recon_loss_i = recon_loss_i * mask
                recon_loss_i = recon_loss_i.mean()
                recon_loss = recon_loss_t + recon_loss_i

        i_prompt = self.prompter(r_t_list, r_i_list)
        t_prompt = self.prompter(r_i_list, r_t_list)

        if self.dataset_name == "hatememes" or self.dataset_name == "food101":
            label_emb = self.label_enhanced[r_l_list]
            label_cls = self.label_enhanced
            label_emb = torch.mean(label_emb, dim=1)
            label_emb = label_emb.view(-1, 1, self.hs)

        # mmimdb is a multi-label classification task, need some special treatment
        elif self.dataset_name == "mmimdb":
            label_tmp = self.label_enhanced.repeat(r_l_list.shape[0], 1, 1)
            label_cls = self.label_enhanced
            label_emb = torch.matmul(r_l_list.float(), label_tmp)
            label_emb = torch.mean(label_emb, dim=1)
            label_emb = label_emb.view(-1, 1, self.hs)

        output = torch.cat([text_emb, image_emb], dim=1)
        for i, layer_module in enumerate(self.encoder_layer):
            if i == self.prompt_pos:
                output = torch.cat([label_emb,t_prompt,i_prompt,output], dim=1)
                N = embedding.shape[0]
                attention_mask = torch.cat([torch.ones(N,1+self.prompt_len*2).to(self.device), attention_mask], dim=1)
                layer_outputs = layer_module(output,
                                             attention_mask=attention_mask
                                            )
                output = layer_outputs[0]
            else:
                layer_outputs = layer_module(output, 
                                             attention_mask=attention_mask
                                             )
                output = layer_outputs[0]
        output = self.layernorm(output)
        output = self.pooler(output)
        output = torch.cat([output,label_emb.squeeze(1)],dim=1)
        output = self.classifier(output)
        label_cls = label_cls.repeat(N, 1,1)
        label_cls = label_cls.transpose(-1,-2)
        output = output.unsqueeze(1)
        output = torch.matmul(output, label_cls)
        output = output.squeeze(1)
        return {
            "output":output, 
            "recon_loss":recon_loss
            }