from models.med import BertConfig, BertModel, BertLMHeadModel
from models.blip import create_vit, init_tokenizer, load_checkpoint

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np
from medical_knowledge.knowledge import create_knowledge
from medical_knowledge.SKG_knowledge import *
from models.tagencoder import TagEncoder

def seperate_C(v, q, answer_target, q_mask):  # q: b x 12 x 1024  v:  b x 1 x 128 answer_target : 1 x b
    indexs_open = []
    indexs_close = []
    for i in range(len(answer_target)):
        if answer_target[i] == 0:
            indexs_close.append(i)
        else:
            indexs_open.append(i)
    return v[indexs_open, :, :], v[indexs_close, :, :], q[indexs_open, :], q[indexs_close, :], \
           q_mask[indexs_open, :], q_mask[indexs_close, :]


class BLIP_VQA(nn.Module):
    def __init__(self,
                 args=None,
                 med_config='configs/med_config.json',
                 image_size=480,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.args = args
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer,
                                                       drop_path_rate=0.1)
        self.tokenizer = init_tokenizer(args)

        if args.bert == 'base':
            med_config = 'configs/med_config_blip.json'
        elif args.bert == 'sci':
            med_config = 'configs/med_config_sci.json'
        elif args.bert == 'cli':
            med_config = 'configs/med_config_cli.json'

        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)
        self.text_encoder1 = BertModel(config=encoder_config, add_pooling_layer=False)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        text_width = self.text_encoder.config.hidden_size

        decoder_config = BertConfig.from_json_file(med_config)
        self.text_decoder = BertLMHeadModel(config=decoder_config)

        self.vision_proj = nn.Linear(vision_width, 256)
        self.text_proj = nn.Linear(text_width, 256)
        if self.args.have_know:
            self.create_knowledge = create_knowledge(embed_dim=256, queue_size=65536,
                                                     text_encoder=self.text_encoder, text_proj=self.text_proj,
                                                     tokenizer=self.tokenizer, args=args)
        if self.args.SKG_know:
            c = copy.deepcopy
            attn = MultiHeadedAttention(6, 768)
            ff = PositionwiseFeedForward(768, 1024, 0.1)
            self.cross_attn = Decoder(DecoderLayer(768, c(attn), c(ff), 0.1), 2)
            # self.tag_encoder = BertModel(config=BertConfig.from_json_file('configs/tag_config_sci_down.json'), add_pooling_layer=False)
            self.tag_encoder = TagEncoder(0.1, self.args)


    def forward(self, knowledge, t_queue, k_input_queue, k_mask_queue, image, question, answer_target=None, q_mask=None, split=None, answer=None, n=None, weights=None, train=True,
                inference='rank', k_test=128):

        if self.args.test_C:
            text_output = self.text_encoder(question, attention_mask=q_mask,
                                            return_dict=True, mode='text')
            text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)
            image_embeds = self.visual_encoder(image)
            # image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

            ###============== Obtain Knowledge ===================###
            if self.args.have_know:
                image_embeds = self.create_knowledge.get_image_knowledge(image.device, image_feat, image_embeds, k=3)
                if split == 'train':
                    t_queue, k_input_queue, k_mask_queue = self.create_knowledge(image.device, text_feat, knowledge)

            # image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            # image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

            else:
                t_queue = None
                k_input_queue = None
                k_mask_queue = None

            if self.args.SKG_know:
                # tag = self.tokenizer(knowledge, padding='max_length', truncation=True, max_length=25,
                #                      return_tensors="pt").to(image.device)
                # tag_output = self.tag_encoder(tag.input_ids, attention_mask=tag.attention_mask,
                #                               return_dict=True, mode='text')
                tag_output = self.tag_encoder(knowledge, image.device)

                if self.args.SKG_out == 'image':
                    # image_embeds, vis_attn1 = self.cross_attn(image_embeds, tag_output.last_hidden_state)
                    image_embeds, vis_attn1 = self.cross_attn(image_embeds, tag_output)
                else:
                    # image_embeds, vis_attn1 = self.cross_attn(tag_output.last_hidden_state, image_embeds)
                    image_embeds, vis_attn1 = self.cross_attn(tag_output, image_embeds)

                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

                image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

            ###===================================================###

            v_open, v_close, q_open, q_close, q_mask_open, q_mask_close = seperate_C(image_embeds, question,
                                                                                         answer_target, q_mask)
            image_atts_close = torch.ones(v_close.size()[:-1], dtype=torch.long).to(image.device)
            image_atts_open = torch.ones(v_open.size()[:-1], dtype=torch.long).to(image.device)
            q_mask_open = q_mask_open.to(image.device)
            q_mask_close = q_mask_close.to(image.device)

            output_close = self.text_encoder(q_close,
                                             attention_mask=q_mask_close,
                                             encoder_hidden_states=v_close,
                                             encoder_attention_mask=image_atts_close,
                                             return_dict=True)
            output_open = self.text_encoder1(q_open,
                                             attention_mask=q_mask_open,
                                             encoder_hidden_states=v_open,
                                             encoder_attention_mask=image_atts_open,
                                             return_dict=True)
            w_emb_close = None
            w_emb_open = None
            q_emb_close = None
            q_emb_open = None
            if self.args.add_typeatt1 or self.args.add_typeatt2:
                q_emb_close = self.text_encoder(q_close, attention_mask=q_mask_close, return_dict=True,
                                                mode='text').last_hidden_state
                q_emb_open = self.text_encoder1(q_open, attention_mask=q_mask_open, return_dict=True,
                                                mode='text').last_hidden_state
                w_emb_close = self.text_encoder.embeddings(input_ids=q_close)
                w_emb_open = self.text_encoder1.embeddings(input_ids=q_open)
            return output_close.last_hidden_state[:, 0, :], output_open.last_hidden_state[:, 0, :], w_emb_close, \
                   w_emb_open, q_emb_close, q_emb_open, t_queue, k_input_queue, k_mask_queue

    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):

        num_ques = question_states.size(0)
        start_ids = answer_ids[0, 0].repeat(num_ques, 1)  # bos token

        start_output = self.text_decoder(start_ids,
                                         encoder_hidden_states=question_states,
                                         encoder_attention_mask=question_atts,
                                         return_dict=True,
                                         reduction='none')
        logits = start_output.logits[:, 0, :]  # first token's logit

        # topk_probs: top-k probability
        # topk_ids: [num_question, k]
        answer_first_token = answer_ids[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(dim=1, index=answer_first_token)
        topk_probs, topk_ids = prob_first_token.topk(k, dim=1)

        # answer input: [num_question*k, answer_len]
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids, dim=0)
        input_atts = torch.cat(input_atts, dim=0)

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)

        output = self.text_decoder(input_ids,
                                   attention_mask=input_atts,
                                   encoder_hidden_states=question_states,
                                   encoder_attention_mask=question_atts,
                                   labels=targets_ids,
                                   return_dict=True,
                                   reduction='none')

        log_probs_sum = -output.loss
        log_probs_sum = log_probs_sum.view(num_ques, k)

        max_topk_ids = log_probs_sum.argmax(dim=1)
        max_ids = topk_ids[max_topk_ids >= 0, max_topk_ids]

        return max_ids


def blip_vqa(pretrained='', **kwargs):
    model = BLIP_VQA(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
    #         assert(len(msg.missing_keys)==0)
    return model


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))

