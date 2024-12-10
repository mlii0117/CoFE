'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import warnings
warnings.filterwarnings("ignore")

from models.vit_blip import VisionTransformer, interpolate_pos_embed
from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer, AutoTokenizer

import torch
from torch import nn
import torch.nn.functional as F

import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file
from functools import partial
from medical_knowledge.knowledge import create_knowledge
from medical_knowledge.SKG_knowledge import *
from models.tagencoder import TagEncoder

import numpy as np
import time

# torch.set_printoptions(threshold=np.inf)

# --------------------------------------------------------------------#

classnames = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
              'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
              'Pneumonia', 'Pneumothorax', 'Support Devices']

# --------------------------------------------------------------------#


class BLIP_Base(nn.Module):
    def __init__(self,
                 med_config='configs/med_config_blip.json',
                 image_size=224,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 args=None,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()
        self.args = args
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer(args)
        if args.bert == 'base':
            med_config = 'configs/med_config_blip.json'
        elif args.bert == 'sci':
            med_config = 'configs/med_config_sci.json'
        elif args.bert == 'cli':
            med_config = 'configs/med_config_cli.json'
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)
        self.vision_proj = nn.Linear(vision_width, 256)
        self.text_proj = nn.Linear(vision_width, 256)
        self.iu_proj = nn.Linear(vision_width * 2, vision_width)
        if self.args.have_know:
            self.create_knowledge = create_knowledge(embed_dim=256, queue_size=65536,
                                                     text_encoder=self.text_encoder, text_proj=self.text_proj,
                                                     tokenizer=self.tokenizer, args=args)
        if self.args.SKG_know:
            c = copy.deepcopy
            attn = MultiHeadedAttention(6, vision_width)
            ff = PositionwiseFeedForward(vision_width, 1024, 0.1)
            self.cross_attn = Decoder(DecoderLayer(vision_width, c(attn), c(ff), 0.1), 2)

            # self.tag_encoder = BertModel(config=BertConfig.from_json_file('configs/tag_config_sci_down.json'), add_pooling_layer=False)
            self.tag_encoder = TagEncoder(0.1, self.args)

    def forward(self, image, caption, knowledge, mode, split):

        assert mode in ['image', 'text', 'multimodal'], "mode parameter must be image, text, or multimodal"

        if mode == 'image':
            if self.args.dataset_name == 'iu_xray':
                image_embeds0 = self.visual_encoder(image[:, 0])
                image_embeds1 = self.visual_encoder(image[:, 1])
                image_embeds = torch.cat((image_embeds0, image_embeds1), dim=2)
                image_embeds = self.iu_proj(image_embeds)
            else:
                image_embeds = self.visual_encoder(image)

            image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

            ###============== Obtain Knowledge ===================###
            if self.args.SKG_know:
                tag_output = self.tag_encoder(knowledge, image.device)

                if self.args.SKG_out == 'image':
                    image_embeds, vis_attn1 = self.cross_attn(image_embeds, tag_output)
                else:
                    image_embeds, vis_attn1 = self.cross_attn(tag_output, image_embeds)

                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

                image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

            if self.args.have_know:
                image_embeds = self.create_knowledge.get_image_knowledge(image.device, image_feat, image_embeds, k=3)
                if split == 'train':
                    text = self.tokenizer(caption, padding='longest', truncation=True, max_length=90,
                                          return_tensors="pt").to(
                        image.device)
                    text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                                    return_dict=True, mode='text')
                    text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

                    self.create_knowledge(image.device, text_feat, knowledge)

            else:
                text_feat_queue = None
                knowledge_input_ids_queue = None
                knowledge_attention_mask_queue = None

            return image_embeds

        elif mode == 'text':
            # return text features
            text = self.tokenizer(caption, return_tensors="pt").to(image.device)
            text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                            return_dict=True, mode='text')
            return text_output.last_hidden_state

        elif mode == 'multimodal':
            # return multimodel features
            text = self.tokenizer(caption, return_tensors="pt").to(image.device)
            image_embeds = self.visual_encoder(image)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            text.input_ids[:, 0] = self.tokenizer.enc_token_id
            output = self.text_encoder(text.input_ids,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       )
            return output.last_hidden_state


class BLIP_Decoder(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=224,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 prompt='a picture of ',
                 tokenizer=None,
                 args=None
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        # vision_width = 768
        if vit == 'base':
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            msg = self.visual_encoder.load_state_dict(state_dict, strict=False)
        elif vit == 'large':
            from timm.models.helpers import load_custom_pretrained
            from timm.models.vision_transformer import default_cfgs
            load_custom_pretrained(self.visual_encoder, default_cfgs['vit_large_patch16_224_in21k'])

        # self.tokenizer = init_tokenizer()
        self.tokenizer = tokenizer
        self.args = args
        self.prompt = prompt
        if self.args.tokenizer == 'our':
            med_config = BertConfig.from_json_file(med_config)
            med_config.encoder_width = vision_width
            self.prompt_length = len(self.tokenizer(self.prompt)) - 1
        else:
            if args.bert == 'base':
                med_config = 'configs/med_config_blip.json'
            elif args.bert == 'sci':
                med_config = 'configs/med_config_sci.json'
            elif args.bert == 'cli':
                med_config = 'configs/med_config_cli.json'
            encoder_config = BertConfig.from_json_file(med_config)
            encoder_config.encoder_width = vision_width
            self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)

            # med_config.encoder_width = vision_width
            self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        # self.text_decoder = BertLMHeadModel(config=med_config)
        self.batch_size = args.batch_size
        self.embed_dim = 256

        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, self.embed_dim)
        self.text_proj = nn.Linear(text_width, self.embed_dim)

        self.itm_head = nn.Linear(text_width, 2)

        # create momentum encoders
        self.visual_encoder_m, vision_width = create_vit(vit, image_size)
        self.vision_proj_m = nn.Linear(vision_width, self.embed_dim)
        self.text_encoder_m = BertModel(config=encoder_config, add_pooling_layer=False)
        self.text_proj_m = nn.Linear(text_width, self.embed_dim)

        self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                            [self.vision_proj, self.vision_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m]
                            ]

        self.copy_params()
        self.queue_size = 3200
        # create the queue
        self.register_buffer("image_feat_queue", torch.randn(self.embed_dim, self.queue_size))
        self.register_buffer("image_queue", torch.randn(2, 3, image_size * image_size, self.queue_size))
        self.register_buffer("text_feat_queue", torch.randn(self.embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randint(0, 31091, (self.args.max_seq_length, self.queue_size)))
        self.register_buffer("label_queue", torch.randn(len(classnames), self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_feat_queue = nn.functional.normalize(self.image_feat_queue, dim=0)
        self.text_feat_queue = nn.functional.normalize(self.text_feat_queue, dim=0)
        self.image_queue = nn.functional.normalize(self.image_queue, dim=2)

        self.momentum = 0.995
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        # create the decoder
        decoder_config = BertConfig.from_json_file(med_config)
        decoder_config.encoder_width = vision_width
        if args.bert == 'base':
            self.text_decoder = BertLMHeadModel(config=decoder_config)
        elif args.bert == 'sci':
            self.text_decoder = BertLMHeadModel(config=decoder_config)
        elif args.bert == 'cli':
            self.text_decoder = BertLMHeadModel(config=decoder_config)
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))
        tie_encoder_decoder_weights(self.text_encoder, self.text_decoder.bert, '', '/attention')

        if args.dataset_name == 'iu_xray':
            self.iu_proj = nn.Linear(vision_width * 2, vision_width)
        if self.args.have_know:
            self.create_knowledge = create_knowledge(embed_dim=256, queue_size=65536,
                                                     text_encoder=self.text_encoder, text_proj=self.text_proj,
                                                     tokenizer=self.tokenizer, args=args)
        if self.args.SKG_know:
            c = copy.deepcopy
            attn = MultiHeadedAttention(6, vision_width)
            ff = PositionwiseFeedForward(vision_width, 1024, 0.1)
            self.cross_attn = Decoder(DecoderLayer(vision_width, c(attn), c(ff), 0.1), 2)
            # self.tag_encoder = BertModel(config=BertConfig.from_json_file('configs/tag_config_sci_down.json'), add_pooling_layer=False)
            self.tag_encoder = TagEncoder(0.1, self.args)


    def forward(self, image, caption, alpha, label):

        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        # image: b * 2 * 3 * 224 * 224
        if self.args.dataset_name == 'iu_xray':
            image_embeds0 = self.visual_encoder(image[:, 0])
            image_embeds1 = self.visual_encoder(image[:, 1])
            image_embeds = torch.cat((image_embeds0, image_embeds1), dim=2)
            image_embeds = self.iu_proj(image_embeds)
        else:
            image_embeds = self.visual_encoder(image)  # image_embeds

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=self.args.max_seq_length, return_tensors="pt").to(image.device)
        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                        return_dict=True, mode='text')
        # text.input_ids: 2 * 90
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

        # get momentum features
        with torch.no_grad():
            self._momentum_update()

            if self.args.dataset_name == 'iu_xray':
                image_embeds0_m = self.visual_encoder_m(image[:, 0])
                image_embeds1_m = self.visual_encoder_m(image[:, 1])
                image_embeds_m = torch.cat((image_embeds0_m, image_embeds1_m), dim=2)
                image_embeds_m = self.iu_proj(image_embeds_m)
            else:
                image_embeds_m = self.visual_encoder_m(image)  # image_embeds

            # image_embeds_m = self.visual_encoder_m(image)
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)

            text_output_m = self.text_encoder_m(text.input_ids, attention_mask=text.attention_mask,
                                                return_dict=True, mode='text')
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)

            image_feat_all = torch.cat([image_feat_m.t(), self.image_feat_queue.clone().detach()], dim=1)

            text_feat_all = torch.cat([text_feat_m.t(), self.text_feat_queue.clone().detach()], dim=1)

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp

            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

        loss_itc = (loss_i2t + loss_t2i) / 2

        label = label.to(image.device)
        self._dequeue_and_enqueue(image_feat_m, text_feat_m, label, image, text.input_ids)

        ###============== Image-text Matching ===================###
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id

        # forward the positve image-text pair
        output_pos = self.text_encoder(encoder_input_ids,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       )
        # vl_embeddings = output_pos.last_hidden_state[:, 0, :]
        # vl_output = self.itm_head(vl_embeddings)
        # loss_itm = vl_output[:, 1].sum()

        with torch.no_grad():
            weights_t2i = F.softmax(sim_t2i[:, :self.batch_size], dim=1) + 1e-4
            weights_t2i.fill_diagonal_(0)
            weights_i2t = F.softmax(sim_i2t[:, :self.batch_size], dim=1) + 1e-4
            weights_i2t.fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(self.batch_size):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(self.batch_size):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(encoder_input_ids[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        output_neg = self.text_encoder(text_ids_all,
                                       attention_mask=text_atts_all,
                                       encoder_hidden_states=image_embeds_all,
                                       encoder_attention_mask=image_atts_all,
                                       return_dict=True,
                                       )

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]], dim=0)
        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat([torch.ones(self.batch_size, dtype=torch.long), torch.zeros(2 * self.batch_size, dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)
        ##================= LM ========================##

        decoder_input_ids = text.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        decoder_targets = decoder_input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)
        decoder_targets[:, :self.prompt_length] = -100

# <<<===delete==================================================================================

        # decoder_output = self.text_decoder(decoder_input_ids,
        #                                    attention_mask=text.attention_mask,
        #                                    encoder_hidden_states=image_embeds,
        #                                    encoder_attention_mask=image_atts,
        #                                    labels=decoder_targets,
        #                                    return_dict=True,
        #                                    )
        #
        # loss_lm = decoder_output.loss
# ===========================================================================================>>>

        ###============ generate cfimage_feat ============== ###
        image_feat_queue = self.image_feat_queue.clone().detach()
        text_feat_queue = self.text_feat_queue.clone().detach()
        image_queue = self.image_queue.clone().detach()
        text_queue = self.text_queue.clone().detach()
        label_queue = self.label_queue.clone().detach()  # 14 * 3200
        sim_t2nt = text_feat @ text_feat_queue
        [_, sim_i2ni_index] = sim_t2nt.sort(dim=1, descending=True)  # 2 * 3200
        index_matrix = label_queue.T[sim_i2ni_index]  # 2 * 3200 * 14

        nimage = []
        nimage_feat = []
        ntext = []
        ntext_feat = []
        for nl_num, l in zip(range(len(index_matrix)), label):
            for nl in range(len(index_matrix[nl_num])):
                if not l.equal(index_matrix[nl_num][nl]):
                    nimage.append(image_queue[:, :, :, nl])
                    nimage_feat.append(image_feat_queue[:, nl])
                    ntext.append(text_queue[:, nl])
                    ntext_feat.append(text_feat_queue[:, nl])
                    break
        nimage = torch.stack(nimage)
        nimage_feat = torch.stack(nimage_feat)
        ntext = torch.stack(ntext)
        ntext_feat = torch.stack(ntext_feat)

        cfimage_feat = self.gen_cfimage(image.reshape(self.batch_size, 2, 3, -1), nimage, text_feat, ntext_feat)

        pos = torch.cosine_similarity(text_feat, image_feat, dim=1)
        neg = torch.cosine_similarity(text_feat, cfimage_feat, dim=1)

        logit = torch.stack((pos, neg), 1)  # [b, 2]
        softmax_logit = nn.functional.softmax(logit, 1)  # [b, 2]
        loss_infoNCE = - torch.log(softmax_logit[:, 0]).mean()
# <<<===add=====================================================================================
        ###============= generate cftext_feat ================###
        cftext_feat = self.gen_cftext(text.input_ids, ntext, image_feat, nimage_feat)

        decoder_output = self.text_decoder(decoder_input_ids,
                                           attention_mask=text.attention_mask,
                                           encoder_hidden_states=image_embeds,
                                           encoder_attention_mask=image_atts,
                                           labels=decoder_targets,
                                           return_dict=True,
                                           memory=cftext_feat,
                                           )

        loss_lm = decoder_output.loss
# ===========================================================================================>>>
        loss = loss_itc + loss_itm + loss_lm + loss_infoNCE
        return loss#  , u_norm


    def generate(self, image, knowledge, sample=False, num_beams=3, max_length=90, min_length=10, top_p=0.9,
                 repetition_penalty=1.0):
        if self.args.dataset_name == 'iu_xray':
            image_embeds0 = self.visual_encoder(image[:, 0])
            image_embeds1 = self.visual_encoder(image[:, 1])
            image_embeds = torch.cat((image_embeds0, image_embeds1), dim=2)
            image_embeds = self.iu_proj(image_embeds)
        else:
            image_embeds = self.visual_encoder(image)

        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        knowledge_used = ''
        ###============== Obtain Knowledge ===================###
        if self.args.SKG_know:
            tag_output = self.tag_encoder(knowledge, image.device)

            if self.args.SKG_out == 'image':
                image_embeds, vis_attn1 = self.cross_attn(image_embeds, tag_output)
            else:
                image_embeds, vis_attn1 = self.cross_attn(tag_output, image_embeds)

            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

            knowledge_used = ''

        if self.args.have_know:
            image_embeds = self.create_knowledge.get_image_knowledge(image.device, image_feat, image_embeds, k=3)
            knowledge_used = ''

        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
# <<<===delete==================================================================================
        # model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}
# ===========================================================================================>>>

        att_masks = image_embeds.new_ones(image_embeds.shape[:2], dtype=torch.long)
        image_mask = att_masks.unsqueeze(-2)
# <<<===add=====================================================================================
        ### find similar text_feat
        text_queue = self.text_queue.clone().detach()
        text_feat_queue = self.text_feat_queue.clone().detach()
        label_queue = self.label_queue.clone().detach()
        sim_i2t = image_feat @ text_feat_queue
        [_, sim_i2t_index] = sim_i2t.sort(dim=1, descending=True)
        text = text_queue[:, sim_i2t_index[:, 0]].T
        text_feat = text_feat_queue[:, sim_i2t_index[:, 0]].T
        label = label_queue[:, sim_i2t_index[:, 0]].T
        ### get index
        image_feat_queue = self.image_feat_queue.clone().detach()
        sim_t2nt = text_feat @ text_feat_queue
        [_, sim_i2ni_index] = sim_t2nt.sort(dim=1, descending=True)
        index_matrix = label_queue.T[sim_i2ni_index]
        ### generate counterfactual text feature
        nimage_feat = []
        ntext = []
        for nl_num, l in zip(range(len(index_matrix)), label):
            for nl in range(len(index_matrix[nl_num])):
                if not l.equal(index_matrix[nl_num][nl]):
                    nimage_feat.append(image_feat_queue[:, nl])
                    ntext.append(text_queue[:, nl])
                    break
        nimage_feat = torch.stack(nimage_feat)
        ntext = torch.stack(ntext)

        cftext_feat = self.gen_cftext(text, ntext, image_feat, nimage_feat, 1)  # .to(image.device)
# ===========================================================================================>>>

# <<<===add=====================================================================================
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts, 'memory': cftext_feat}
# ===========================================================================================>>>

        prompt = [self.prompt] * image.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]
        if sample:
            # nucleus sampling
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 do_sample=True,
                                                 top_p=top_p,
                                                 num_return_sequences=1,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id,
                                                 repetition_penalty=1.1,
                                                 **model_kwargs)
        else:
            # beam search
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 num_beams=num_beams,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id,
                                                 repetition_penalty=repetition_penalty,
                                                 **model_kwargs)

        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[len(self.prompt):])

        return captions, knowledge_used

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, label, image, text):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        labels = concat_all_gather(label)
        image = concat_all_gather(image)
        text = concat_all_gather(text)
        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_feat_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_feat_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.label_queue[:, ptr:ptr + batch_size] = labels.T
        self.image_queue[:, :, :, ptr:ptr + batch_size] = image.reshape(batch_size, 2, 3, -1).permute(1, 2, 3, 0)
        self.text_queue[:, ptr:ptr + batch_size] = text.T

        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def gen_cfimage(self, image, nimage, text_feat, ntext_feat):
        # image: b * 2 * 3 * 224^2  text: b * 90
        batch_size = image.size(0)
        image_dim = image.size(-1)  # 224^2
        image_length = int(image_dim**0.5)
        patch_size = 16
        patch_num = (image_length // patch_size)**2  # 196

        image = image.reshape(batch_size, 2, 3, 16, -1)  # b * 2 * 3 * 16 * 3136
        nimage = nimage.reshape(batch_size, 2, 3, 16, -1)

        r_image = []
        for ptr in range(0, image.size(-1), 16):
            one = torch.ones(batch_size, 2, 3, 16, image_dim // 16).to(image.device)
            one[:, :, :, :, ptr: ptr+16] = 0
            rimage = (image * one  + nimage * (1 - one)).reshape(batch_size, 2, 3, image_length, -1)

            rimage_embed0 = self.visual_encoder(rimage[:, 0])
            rimage_embed1 = self.visual_encoder(rimage[:, 1])
            rimage_embed = torch.cat((rimage_embed0, rimage_embed1), dim=2)
            rimage_embeds = self.iu_proj(rimage_embed)  # b * 197 * 768

            r_image_feat = F.normalize(self.vision_proj(rimage_embeds[:, 0, :]), dim=-1)  # b * 256
            r_image.append(r_image_feat)
        rimage_feat = torch.stack(r_image).permute(1, 0, 2)  # b * 196 * 256

        one = torch.ones(patch_num, patch_num).to(image.device)
        zero = torch.zeros(batch_size, patch_num, patch_num).to(image.device)
        one2 = torch.ones(batch_size, patch_num, patch_num).to(image.device)

        ex_text = text_feat.unsqueeze(2)
        ex_ntext = ntext_feat.unsqueeze(2)

        sim_InT = (rimage_feat @ ex_ntext).squeeze()  # b * 196
        [_, sim_InT_index] = sim_InT.sort(dim=1, descending=True)
        padding = sim_InT_index[:, 0]

        copy_sim_InT_index = sim_InT_index.unsqueeze(1).repeat(1, patch_num, 1)  # b * 196 * 196
        replace_index = copy_sim_InT_index * torch.tril(one, diagonal=0).unsqueeze(0).repeat(batch_size, 1, 1) \
                + padding.unsqueeze(1).unsqueeze(2) * torch.triu(one, diagonal=1)
        u = zero.scatter(2, replace_index.long(), one2)  # b * 196 * 196
        u = u.unsqueeze(2).repeat(1, 1, 16, 1).permute(0, 1, 3, 2).reshape(batch_size, patch_num, -1).permute(1, 0, 2)\
            .unsqueeze(2).repeat(1, 1, 16, 1).unsqueeze(2).repeat(1, 1, 3, 1, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1, 1)  # 196 * b * 2 * 3 * 16 * 3136

        r_image2 = []
        for num in range(patch_num):
            rimage2 = (image * u[num] + nimage * (1 - u[num])).reshape(batch_size, 2, 3, image_length, -1)

            rimage_embed02 = self.visual_encoder(rimage2[:, 0])
            rimage_embed12 = self.visual_encoder(rimage2[:, 1])
            rimage_embed2 = torch.cat((rimage_embed02, rimage_embed12), dim=2)
            rimage_embeds2 = self.iu_proj(rimage_embed2)  # b * 197 * 768

            rimage_feat2 = F.normalize(self.vision_proj(rimage_embeds2[:, 0, :]), dim=-1)  # b * 256
            r_image2.append(rimage_feat2)
        rimage_feat2 = torch.stack(r_image2).permute(1, 0, 2)

        sim_rInT = rimage_feat2 @ ex_ntext
        sim_rIT = rimage_feat2 @ ex_text
        _, result_index = (sim_rInT > sim_rIT).long().max(dim=1)

        result = []
        for index, feat in zip(result_index, rimage_feat2):
            result.append(feat[index])
        cfimage_feat = F.normalize(torch.stack(result).squeeze())

        return cfimage_feat

    @torch.no_grad()
    def gen_cftext(self, text, ntext, image_feat, nimage_feat, k=0):
        batch_size = text.size(0)
        text_dim = text.size(1)  # 90

        copy_text = text.unsqueeze(1).repeat(1, text_dim, 1)  # b * 90 * 90
        copy_ntext = ntext.unsqueeze(1).repeat(1, text_dim, 1)

        eye = torch.eye(text_dim).unsqueeze(0).repeat(batch_size, 1, 1).to(text.device)
        one = torch.ones(text_dim, text_dim).to(text.device)
        zero = torch.zeros(batch_size, text_dim, text_dim).to(text.device)
        one2 = torch.ones(batch_size, text_dim, text_dim).to(text.device)

        rtext = (copy_text * (1 - eye) + copy_ntext * eye).long()  # b * 90 * 90
        attention_mask = (rtext != 0).long()

        r_text = []
        for i in range(text_dim):
            text_output = self.text_encoder(rtext[:, i], attention_mask=attention_mask[:, i],
                                            return_dict=True, mode='text')
            text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)
            r_text.append(text_feat)
        rtext_feat = torch.stack(r_text).permute(1, 0, 2)  # b * 90 * 256

        ex_image = image_feat.unsqueeze(2)  # b * 256 * 1
        ex_nimage = nimage_feat.unsqueeze(2)

        sim_TnI = (rtext_feat @ ex_nimage).squeeze()  # b * 90
        [_, sim_TnI_index] = sim_TnI.sort(dim=1, descending=True)
        padding = sim_TnI_index[:, 0]

        copy_sim_InT_index = sim_TnI_index.unsqueeze(1).repeat(1, text_dim, 1)
        replace_index = copy_sim_InT_index * torch.tril(one, diagonal=0).unsqueeze(0).repeat(batch_size, 1, 1) \
                + padding.unsqueeze(1).unsqueeze(2) * torch.triu(one, diagonal=1)
        u = zero.scatter(2, replace_index.long(), one2)  # b * 90 * 90

        rtext2 = (copy_text * (1 - u) + copy_ntext * u).long()
        attention_mask2 = (rtext2 != 0).long()
        r_text2 = []
        for i in range(text_dim):
            text_output2 = self.text_encoder(rtext2[:, i], attention_mask=attention_mask2[:, i],
                                            return_dict=True, mode='text')
            text_feat2 = F.normalize(self.text_proj(text_output2.last_hidden_state[:, 0, :]), dim=-1)
            r_text2.append(text_feat2)
        rtext_feat2 = torch.stack(r_text2).permute(1, 0, 2)  # b * 90 * 256

        sim_rTnI = rtext_feat2 @ ex_nimage  # b * 90 * 1
        sim_rTI = rtext_feat2 @ ex_image
        _, result_index = (sim_rTnI > sim_rTI).long().max(dim=1)

        result = []
        for index, feat in zip(result_index, rtext_feat2):
            result.append(feat[index])
        cftext_feat = F.normalize(torch.stack(result).squeeze())

        return cftext_feat

def blip_decoder(pretrained=False, **kwargs):
    model = BLIP_Decoder(**kwargs)
    # if pretrained:
    #     model,msg = load_checkpoint(model,pretrained)
    #     # if kwargs['args'].tokenizer == 'blip':
    #     #     print(msg.missing_keys)
    #     #     assert(len(msg.missing_keys)==0)
    #     # elif kwargs['args'].tokenizer == 'our':
    #     #     assert(len(msg.missing_keys)==4)
    return model


def blip_feature_extractor(pretrained='', **kwargs):
    model = BLIP_Base(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        # assert(len(msg.missing_keys)==0)
        print(msg.missing_keys)
    return model


def init_tokenizer(args):
    if args.bert == 'base':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif args.bert == 'sci':
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    elif args.bert == 'cli':
        tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
    # create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit == 'base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12,
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing,
                                           ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate
                                           )
        # visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12,
        #                                    num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    elif vit == 'large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24,
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing,
                                           ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate
                                           )
    return visual_encoder, vision_width


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def load_checkpoint(model, url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu')
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')

    state_dict = checkpoint['model']

    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],
                                                                   model.visual_encoder)
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                         model.visual_encoder_m)
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape != model.state_dict()[key].shape:
                # print(state_dict[key])
                print(state_dict[key].shape)
                print(model.state_dict()[key].shape)
                del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % url_or_filename)
    return model, msg


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


from typing import List


def tie_encoder_decoder_weights(encoder: nn.Module, decoder: nn.Module, base_model_prefix: str, skip_key: str):
    uninitialized_encoder_weights: List[str] = []
    if decoder.__class__ != encoder.__class__:
        logger.info(
            f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
        )

    def tie_encoder_to_decoder_recursively(
            decoder_pointer: nn.Module,
            encoder_pointer: nn.Module,
            module_name: str,
            uninitialized_encoder_weights: List[str],
            skip_key: str,
            depth=0,
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        if hasattr(decoder_pointer, "weight") and skip_key not in module_name:
            assert hasattr(encoder_pointer, "weight")
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                encoder_pointer.bias = decoder_pointer.bias
            print(module_name + ' is tied')
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert (
                    len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = set([module_name + "/" + sub_name for sub_name in encoder_modules.keys()])
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(
                            encoder_modules
                    ) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    skip_key,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights, skip_key)


