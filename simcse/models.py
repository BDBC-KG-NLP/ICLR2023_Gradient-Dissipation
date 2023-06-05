import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable
import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.models.t5.modeling_t5 import T5PreTrainedModel, T5EncoderModel
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
from torch.utils.tensorboard import SummaryWriter

def decouple_contrastive_loss(x, y, cls):
    cos_sim = cls.sim(x.unsqueeze(1), y.unsqueeze(0))
    ones = torch.ones(cos_sim.shape, dtype=torch.float32).to(cls.device)
    diag = torch.eye(cos_sim.shape[0], dtype=torch.float32).to(cls.device)
    mask = ones - diag
    align_sim = torch.mul(cos_sim, diag).sum(dim=-1)
    uniform_sim = (torch.mul(cos_sim, mask).exp().sum(dim=-1) - 1).log()
    loss = (-align_sim + uniform_sim).mean()
    return loss

def decouple_contrastive_positive_loss(x, y, cls):
    cos_sim = cls.sim(x.unsqueeze(1), y.unsqueeze(0))
    ones = torch.ones(cos_sim.shape, dtype=torch.float32).to(cls.device)
    diag = torch.eye(cos_sim.shape[0], dtype=torch.float32).to(cls.device)
    mask = ones - diag
    align_sim = torch.mul(cos_sim, diag).sum(dim=-1)
    uniform_sim = (torch.mul(cos_sim, mask).exp().sum(dim=-1) - 1).log()
    loss = -align_sim + uniform_sim
    loss_mask = loss > 0
    loss = (loss * loss_mask).mean()
    return loss


def margin_loss(x, y, cls, margin=0.15):
    margin = margin * np.pi
    align_sq_vec = (2 - (x - y).norm(dim=1).pow(2)) / 2
    align_angle_vec = torch.arccos(align_sq_vec)
    uniform_sq_vec = torch.mm(x, y.permute(1, 0))
    ones = torch.ones(uniform_sq_vec.shape, dtype=torch.float32).to(cls.device)
    diag = torch.eye(uniform_sq_vec.shape[0], dtype=torch.float32).to(cls.device)
    mask = ones - diag
    uniform_sq_vec = torch.mul(uniform_sq_vec, mask)
    uniform_angle_vec = torch.arccos(uniform_sq_vec)
    uniform_min_vec = torch.min(uniform_angle_vec, dim=0).values
    angle_dist = margin + align_angle_vec - uniform_min_vec
    angle_mask = angle_dist > 0
    loss = (angle_dist * angle_mask).mean()
    return loss

def margin_p_loss(x, y, cls, margin=0.09):
    cos_sim = torch.mm(x, y.permute(1, 0))
    ones = torch.ones(cos_sim.shape, dtype=torch.float32).to(cls.device)
    diag = torch.eye(cos_sim.shape[0], dtype=torch.float32).to(cls.device)
    mask = ones - diag
    align_sim = torch.mul(cos_sim, diag).sum(dim=-1)
    uniform_sim = torch.mul(cos_sim, mask)
    uniform_max_sim = torch.max(uniform_sim, dim=0).values
    cos_dist = margin - align_sim + uniform_max_sim
    cos_mask = cos_dist > 0
    loss = (cos_dist * cos_mask).mean()
    return loss

def margin_e_loss(x, y, cls, margin=0.45):
    align_distance_vec = (x - y).norm(dim=1)
    uniform_sq_vec = (2 - 2 * torch.mm(x, y.permute(1, 0))).sqrt()
    ones = torch.ones(uniform_sq_vec.shape, dtype=torch.float32).to(cls.device)
    diag = torch.eye(uniform_sq_vec.shape[0], dtype=torch.float32).to(cls.device)
    mask = ones - diag
    uniform_dist_vec = torch.mul(uniform_sq_vec, mask) + 2 * diag
    uniform_min_vec = torch.min(uniform_dist_vec, dim=0).values
    angle_dist = margin + align_distance_vec - uniform_min_vec
    angle_mask = angle_dist > 0
    loss = (angle_dist * angle_mask).mean()
    return loss

def lalign(x, y, alpha=2):
    loss = (x - y).norm(dim=1).pow(alpha).mean()
    return loss

def lunif(x, y, t=2):
    x_sq_pdist = torch.pdist(x, p=2).pow(2)
    x_loss = x_sq_pdist.mul(-t).exp().mean().log()
    y_sq_pdist = torch.pdist(y, p=2).pow(2)
    y_loss = y_sq_pdist.mul(-t).exp().mean().log()
    return 2 * t + (x_loss + y_loss) / 2



class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y, change_temp=None):
        # return self.cos(x, y)
        if change_temp is None:
            return self.cos(x, y) / self.temp
        else:
            return self.cos(x, y) / change_temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2",
                                    "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cal_pooler_output(cls,
                      encoder,
                      batch_size,
                      num_sent,
                      token_type_embeddings,
                      position_embeddings,
                      inputs_embeds,
                      input_ids=None,
                      attention_mask=None,
                      token_type_ids=None,
                      position_ids=None,
                      head_mask=None,
                      output_attentions=None,
                      mlm_input_ids=None):
    embedding_output_tuple = encoder.embeddings.norm_and_drop(inputs_embeds, token_type_embeddings, position_embeddings)
    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        embedding_output=embedding_output_tuple,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )
    # MLM auxiliary objective
    mlm_outputs = None

    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1)))  # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    return outputs, pooler_output, mlm_outputs


def cal_similarity(cls, pooler_output, num_sent):
    # Separate representation
    z1, z2 = pooler_output[:, 0], pooler_output[:, 1]
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    # Hard negative
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    # Hard negative
    z1_z3_cos = None
    if num_sent >= 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)
    return z1_z3_cos, cos_sim


def cal_loss(cls, z1, z2, cos_sim, labels):
    if cls.model_args.loss_type == "infonce":
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(cos_sim, labels)
    elif cls.model_args.loss_type == "a&u":
        align_loss = lalign(z1, z2, alpha=cls.model_args.align_alpha)
        uniform_loss = lunif(z1, z2, t=cls.model_args.uniform_t)
        loss = (1 - cls.model_args.loss_lambda) * align_loss + cls.model_args.loss_lambda * uniform_loss
    elif cls.model_args.loss_type == "decouple":
        loss = decouple_contrastive_loss(z1, z2, cls=cls)
    elif cls.model_args.loss_type == "decouple_pos":
        loss = decouple_contrastive_positive_loss(z1, z2, cls=cls)
    elif cls.model_args.loss_type == "mat":
        loss = margin_loss(z1, z2, cls=cls, margin=cls.model_args.margin)
    elif cls.model_args.loss_type == "met":
        loss = margin_e_loss(z1, z2, cls=cls, margin=cls.model_args.margin)
    elif cls.model_args.loss_type == "mpt":
        loss = margin_p_loss(z1, z2, cls=cls, margin=cls.model_args.margin)
    else:
        raise NotImplementedError("You provided the loss type: {}, not implemented!".format(cls.model_args.loss_type))
    return loss


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()


def cl_forward(cls,
               encoder,
               input_ids=None,
               attention_mask=None,
               token_type_ids=None,
               position_ids=None,
               head_mask=None,
               inputs_embeds=None,
               labels=None,
               output_attentions=None,
               output_hidden_states=None,
               return_dict=None,
               mlm_input_ids=None,
               mlm_labels=None,
               ):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    batch_size, num_sent = input_ids.size(0), input_ids.size(1)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))  # (bs * num_sent, len)

    inputs_embeds, token_type_embeddings, position_embeddings = \
        encoder.embeddings.get_divided_embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

    outputs, pooler_output, mlm_outputs = cal_pooler_output(cls, encoder, batch_size, num_sent,
                                                            token_type_embeddings=token_type_embeddings,
                                                            position_embeddings=position_embeddings,
                                                            inputs_embeds=inputs_embeds,
                                                            input_ids=input_ids,
                                                            attention_mask=attention_mask,
                                                            token_type_ids=token_type_ids,
                                                            position_ids=position_ids,
                                                            head_mask=head_mask,
                                                            output_attentions=output_attentions,
                                                            mlm_input_ids=mlm_input_ids)

    z1_z3_cos, cos_sim = cal_similarity(cls=cls,
                                        pooler_output=pooler_output,
                                        num_sent=num_sent)

    z1, z2 = pooler_output[:, 0], pooler_output[:, 1]
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    cls.model_args.loss_type = cls.model_args.loss_type.strip()
    loss = cal_loss(cls=cls, z1=z1, z2=z2, cos_sim=cos_sim, labels=labels)

    cls.writer.add_scalar(tag="loss", scalar_value=loss.item(), global_step=cls.step_num)
    cls.step_num += 1

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def sentemb_forward(
        cls,
        encoder,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))  # (bs * num_sent, len)

    inputs_embeds, token_type_embeddings, position_embeddings = \
        encoder.embeddings.get_divided_embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

    embedding_output_tuple = encoder.embeddings.norm_and_drop(inputs_embeds, token_type_embeddings, position_embeddings)

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        embedding_output=embedding_output_tuple,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)

    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.encoder_name = "bert"
        self.bert = BertModel(config, add_pooling_layer=False)
        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)
        self.writer = SummaryWriter(log_dir=self.model_args.logger_dir, flush_secs=60)
        self.step_num = 1
        cl_init(self, config)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=True,
                output_hidden_states=None,
                return_dict=None,
                sent_emb=False,
                mlm_input_ids=None,
                mlm_labels=None,
                ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                                   input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask,
                                   inputs_embeds=inputs_embeds,
                                   labels=labels,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states,
                                   return_dict=return_dict,
                                   )
        else:
            return cl_forward(self, self.bert,
                              input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids,
                              head_mask=head_mask,
                              inputs_embeds=inputs_embeds,
                              labels=labels,
                              output_attentions=output_attentions,
                              output_hidden_states=output_hidden_states,
                              return_dict=return_dict,
                              mlm_input_ids=mlm_input_ids,
                              mlm_labels=mlm_labels,
                              )

class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.encoder_name = "roberta"
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)
        self.writer = SummaryWriter(log_dir=self.model_args.logger_dir, flush_secs=60)
        self.step_num = 1
        cl_init(self, config)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                sent_emb=False,
                mlm_input_ids=None,
                mlm_labels=None,
                ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                                   input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask,
                                   inputs_embeds=inputs_embeds,
                                   labels=labels,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states,
                                   return_dict=return_dict,
                                   )
        else:
            return cl_forward(self, self.roberta,
                              input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids,
                              head_mask=head_mask,
                              inputs_embeds=inputs_embeds,
                              labels=labels,
                              output_attentions=output_attentions,
                              output_hidden_states=output_hidden_states,
                              return_dict=return_dict,
                              mlm_input_ids=mlm_input_ids,
                              mlm_labels=mlm_labels,
                              )


