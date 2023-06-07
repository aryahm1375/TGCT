import numpy as np
import os
import sys
import math
import torch
from torch import nn
from utils import get_extended_attention_mask
import torch.nn.functional as F
value_id = 0
import pickle
device = 'cuda:2'


class LSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMAttention, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.w1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, value_id):
        output, (h_n, c_n) = self.lstm(x)
        h_n = h_n.squeeze(0)
        attn_weights = self.attention(output, h_n)
        context = torch.bmm(output.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(2)
        out = self.fc(context)

        out = torch.sigmoid(out)

        return out

    def attention(self, output, h_n):
        hidden = h_n.unsqueeze(1).repeat(1, output.size(1), 1)
        energy = F.relu(self.w1(output) + self.w2(hidden))
        attention = self.v(energy).squeeze(2)
        attention_weights = F.softmax(attention, dim=1)
        return attention_weights



class FeatureEmbedder(nn.Module):
    def __init__(self, args, number):
        super().__init__()
        self.embeddings = {}
        number = 1
        self.feature_keys = [f'dx_ints{number}',
                             f'proc_ints{number}',f'patient_id{number}']  # = args.feature_keys; #args.vocab_sizes['dx_ints'] = 14500;args.vocab_sizes['proc_ints'] =5534;print('size',args.vocab_sizes['proc_ints'])
        self.dx_embeddings = nn.Embedding(args.vocab_sizes[f'dx_ints{number}'] + 1, args.hidden_size,
                                          padding_idx=args.vocab_sizes[f'dx_ints{number}'])
        self.proc_embeddings = nn.Embedding(args.vocab_sizes[f'proc_ints{number}'] + 1, args.hidden_size,
                                            padding_idx=args.vocab_sizes[f'proc_ints{number}'])

        self.visit_embeddings = nn.Embedding(1, args.hidden_size)

        self.number = number
        ## stuff to try when everything is done as add-on
        self.layernorm = nn.LayerNorm(args.hidden_size)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, features, number):
        batch_size = features[self.feature_keys[0]].shape[0]
        embeddings = {}
        masks = {}  # ;print(features['dx_ints'])
        embeddings[f'dx_ints{number}'] = self.dx_embeddings(features[f'dx_ints{number}'])
        embeddings[f'proc_ints{number}'] = self.proc_embeddings(features[f'proc_ints{number}'])
        device = features[f'dx_ints{number}'].device
        dx_np = self.dx_embeddings(features[f'dx_ints{number}']).detach().cpu().numpy()
        rx_np = self.proc_embeddings(features[f'proc_ints{number}']).detach().cpu().numpy()


        embeddings[f'visit{number}'] = self.visit_embeddings(torch.tensor([0]).to(device))
        embeddings[f'visit{number}'] = embeddings[f'visit{number}'].unsqueeze(0).expand(batch_size, -1, -1)
        v_np = embeddings[f'visit{number}'].detach().cpu().numpy()
        masks[f'visit{number}'] = torch.ones(batch_size, 1).to(device)
        for name, embedding in embeddings.items():
            # print('EMBEDING', embedding)
            embeddings[name] = self.layernorm(embedding)
            embeddings[name] = self.dropout(embeddings[name])

        return embeddings, masks


class SelfAttention(nn.Module):
    def __init__(self, args, stack_idx):
        super().__init__()
        self.stack_idx = stack_idx
        self.num_attention_heads = args.num_heads
        self.attention_head_size = int(args.hidden_size / args.num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.key_concat = nn.Linear(2 * args.hidden_size, args.hidden_size)
        self.query_concat = nn.Linear(2 * args.hidden_size, args.hidden_size)
        self.args = args
        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)
        self.queryO = nn.Linear(args.hidden_size, self.all_head_size)
        self.keyO = nn.Linear(args.hidden_size, self.all_head_size)
        self.query_out = nn.Linear(args.hidden_size, self.all_head_size)
        self.key_out = nn.Linear(args.hidden_size, self.all_head_size)
        self.value_out = nn.Linear(args.hidden_size, self.all_head_size)
        # experiment with dropout after completion
        # self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, guide_mask=None, prior=None, output_attentions=True,
                keyP=None, queryP=None, first=False, number = None):
        #print('states',np.shape(hidden_states))
        if self.stack_idx == 0 and prior is not None:
            attention_probs = prior[:, None, :, :].expand(-1, self.num_attention_heads, -1, -1)
        else:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_query_layerO = self.queryO(hidden_states)
            mixed_key_layerO = self.keyO(hidden_states)
            query0 = self.transpose_for_scores(mixed_query_layer)
            key0 = self.transpose_for_scores(mixed_key_layer)
            queryO = self.transpose_for_scores(mixed_query_layerO)
            keyO = self.transpose_for_scores(mixed_key_layerO)
            if not first:
                query_layer = self.query_concat(torch.cat((query0, queryP), 3))
                key_layer = self.key_concat(torch.cat((key0, keyP), 3))
            else:
                query_layer = query0
                key_layer = key0
            if self.stack_idx == self.args.num_stacks - 1:
                query_layer = queryP
                key_layer = keyP


            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            if not value_id == 0:
                if self.stack_idx == self.args.num_stacks - 2:

                    attention_scores0 = torch.matmul(query0, key0.transpose(-1, -2))
                    attention_scores0 = attention_scores0/ math.sqrt(self.attention_head_size)

                    # print('score_attention',np.shape(attention_scores))
                    if attention_mask is not None:
                        attention_scores0 = attention_scores0 + attention_mask
                    attention_probs0 = nn.Softmax(dim=-1)(attention_scores0)

                    attention_scoresou = torch.matmul(queryO, keyO.transpose(-1, -2))
                    attention_scoresou = attention_scoresou / math.sqrt(self.attention_head_size)

                    # print('score_attention',np.shape(attention_scores))
                    if attention_mask is not None:
                        attention_scoresou = attention_scoresou + attention_mask
                    attention_probsou = nn.Softmax(dim=-1)(attention_scoresou)
        mixed_value_layer = self.value(hidden_states)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # dropping out entire tokens to attend to; extra experiment
        # attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        if self.stack_idx == 0 and prior is not None:
            return outputs
        else:
            return outputs, keyO, queryO, key_layer, query_layer


class RetainNN(nn.Module):
    def __init__(self, params: dict):
        super(RetainNN, self).__init__()

        self.emb_layer = nn.Linear(in_features=params["num_embeddings"], out_features=params["embedding_dim"])
        self.dropout = nn.Dropout(params["dropout_p"])
        self.variable_level_rnn = nn.GRU(params["var_rnn_hidden_size"], params["var_rnn_output_size"])
        self.visit_level_rnn = nn.GRU(params["visit_rnn_hidden_size"], params["visit_rnn_output_size"])
        self.variable_level_attention = nn.Linear(params["var_rnn_output_size"], params["var_attn_output_size"])
        self.visit_level_attention = nn.Linear(params["visit_rnn_output_size"], params["visit_attn_output_size"])
        self.output_dropout = nn.Dropout(params["output_dropout_p"])
        self.output_layer = nn.Linear(params["embedding_output_size"], 1)

        self.var_hidden_size = params["var_rnn_hidden_size"]

        self.visit_hidden_size = params["visit_rnn_hidden_size"]
        self.sigmoid = nn.Sigmoid()
        self.n_samples = params["batch_size"]
        self.reverse_rnn_feeding = params["reverse_rnn_feeding"]

    def forward(self, input, var_rnn_hidden, visit_rnn_hidden):

        v = self.emb_layer(input)

        v = self.dropout(v)


        if self.reverse_rnn_feeding:
            visit_rnn_output, visit_rnn_hidden = self.visit_level_rnn(torch.flip(v.to(device), [0]), visit_rnn_hidden)
            alpha = self.visit_level_attention(torch.flip(visit_rnn_output, [0]))
        else:
            visit_rnn_output, visit_rnn_hidden = self.visit_level_rnn(v, visit_rnn_hidden)
            alpha = self.visit_level_attention(visit_rnn_output)
        visit_attn_w = F.softmax(alpha, dim=0)

        if self.reverse_rnn_feeding:
            var_rnn_output, var_rnn_hidden = self.variable_level_rnn(torch.flip(v, [0]), var_rnn_hidden)
            beta = self.variable_level_attention(torch.flip(var_rnn_output, [0]))
        else:
            var_rnn_output, var_rnn_hidden = self.variable_level_rnn(v, var_rnn_hidden)
            beta = self.variable_level_attention(var_rnn_output)
        var_attn_w = torch.tanh(beta)

        attn_w = visit_attn_w * var_attn_w
        c = torch.sum(attn_w * v, dim=0)

        c = self.output_dropout(c)

        output = self.output_layer(c)
        output = self.sigmoid(output)


        return output, var_rnn_hidden, visit_rnn_hidden

    def init_hidden(self, current_batch_size):
        return torch.zeros(current_batch_size, self.var_hidden_size).unsqueeze(0).to(device), torch.zeros(
            current_batch_size, self.visit_hidden_size).unsqueeze(0).to(device)


def init_params(params: dict, args):
    # embedding matrix
    params["num_embeddings"] = args.hidden_size  # tedade e embedding?

    params["embedding_dim"] = int(args.hidden_size / 2)  # in ham bahashon barabre
    # embedding dropout
    params["dropout_p"] = 0.2
    # Alpha
    params["visit_rnn_hidden_size"] = int(args.hidden_size / 2)  # in dota yeki hast
    params["visit_rnn_output_size"] = int(args.hidden_size / 2)
    params["visit_attn_output_size"] = 1
    # Beta
    params["var_rnn_hidden_size"] = int(args.hidden_size / 2)  # IN3TA YEKI HAST
    params["var_rnn_output_size"] = int(args.hidden_size / 2)
    params["var_attn_output_size"] = int(args.hidden_size / 2)

    params["embedding_output_size"] = int(args.hidden_size / 2)
    params["num_class"] = 2
    params["output_dropout_p"] = 0.2

    params["batch_size"] = args.batch_size

    params["reverse_rnn_feeding"] = True

    # TODO: Customized Loss
    # TODO: REF: https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
    params["customized_loss"] = True


class SelfOutput(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.layer_norm = nn.LayerNorm(args.hidden_size)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.activation = nn.ReLU()

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.activation(self.dense(hidden_states))
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, args, stack_idx):
        super().__init__()
        self.self_attention = SelfAttention(args, stack_idx)
        self.self_output = SelfOutput(args)
        self.stack_idx = stack_idx

    def forward(self, hidden_states, attention_mask, guide_mask=None, prior=None, output_attentions=True, keyP=None,
                queryP=None, first=False, number = None):
        if self.stack_idx == 0 and prior is not None:
            self_attention_outputs = self.self_attention(hidden_states, attention_mask, guide_mask, prior,
                                                         output_attentions, keyP, queryP, first)
        else:
            self_attention_outputs, key0, query0, key_layer, query_layer = self.self_attention(hidden_states, attention_mask, guide_mask, prior,
                                                                       output_attentions, keyP, queryP, first, number)

        attention_output = self.self_output(self_attention_outputs[0], hidden_states)
        outputs = (attention_output,) + self_attention_outputs[1:]
        if self.stack_idx == 0 and prior is not None:
            return outputs
        else:
            return outputs, key0, query0, key_layer, query_layer


class IntermediateLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(args.hidden_size, args.intermediate_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class OutputLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(args.intermediate_size, args.hidden_size)
        self.layer_norm = nn.LayerNorm(args.hidden_size)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.activation = nn.ReLU()

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.activation(self.dense(hidden_states))
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class GCTLayer(nn.Module):
    def __init__(self, args, stack_idx):
        super().__init__()
        self.attention = Attention(args, stack_idx)
        self.stack_idx = stack_idx


    def forward(self, hidden_states, attention_mask=None, guide_mask=None, prior=None, output_attentions=True,
                keyP=None, queryP=None, first=False, number = None):
        if self.stack_idx == 0 and prior is not None:
            self_attention_outputs = self.attention(hidden_states, attention_mask, guide_mask, prior, output_attentions,
                                                    keyP, queryP, first)

        else:
            self_attention_outputs, key0, query0, key_layer, query_layer = self.attention(hidden_states, attention_mask, guide_mask, prior,
                                                                  output_attentions, keyP, queryP, first, number)

        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]


        outputs = (attention_output,) + outputs
        if self.stack_idx == 0 and prior is not None:
            return outputs
        else:
            return outputs, key0, query0, key_layer, query_layer
        return outputs


class Pooler(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class GraphConvolutionalTransformer(nn.Module):
    def __init__(self, args):
        super(GraphConvolutionalTransformer, self).__init__()
        self.num_labels = args.num_labels
        self.label_key = args.label_key
        self.reg_coef = args.reg_coef
        self.use_guide = args.use_guide
        self.use_prior = args.use_prior
        self.prior_scalar = args.prior_scalar
        self.batch_size = args.batch_size
        self.num_stacks = args.num_stacks
        self.max_num_codes = args.max_num_codes
        self.output_attentions = args.output_attentions
        self.output_hidden_states = args.output_hidden_states
        self.feature_keys = args.feature_keys
        self.layers1 = nn.ModuleList([GCTLayer(args, i) for i in range(args.num_stacks)])
        self.layers2 = nn.ModuleList([GCTLayer(args, i) for i in range(args.num_stacks)])
        self.layers3 = nn.ModuleList([GCTLayer(args, i) for i in range(args.num_stacks)])
        self.layer_norm = nn.LayerNorm(3 * args.hidden_size)
        self.embeddings1 = FeatureEmbedder(args, 1)
        self.embeddings2 = self.embeddings1
        self.embeddings3 = self.embeddings1
        self.pooler1 = Pooler(args)
        self.pooler2 = Pooler(args)
        self.pooler3 = Pooler(args)
        parameters = dict()
        init_params(parameters, args)
        self.retain = LSTMAttention(args.hidden_size,128)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.classifier = nn.Linear(args.hidden_size, 1)
        self.out_sigmoid = nn.Sigmoid()

    def create_matrix_vdp(self, features, masks, priors, number):
        batch_size = features[f'dx_ints{number}'].shape[0]
        device = features[f'dx_ints{number}'].device

        num_dx_ids = self.max_num_codes if self.use_prior else features[f'dx_ints{number}'].shape[-1]
        num_proc_ids = self.max_num_codes if self.use_prior else features[f'proc_ints{number}'].shape[-1]
        num_codes = 1 + num_dx_ids + num_proc_ids
        guide = None
        if self.use_guide:
            row0 = torch.cat([torch.zeros([1, 1]), torch.ones([1, num_dx_ids]), torch.zeros([1, num_proc_ids])], axis=1)
            row1 = torch.cat([torch.zeros([num_dx_ids, num_dx_ids + 1]), torch.ones([num_dx_ids, num_proc_ids])],
                             axis=1)
            row2 = torch.zeros([num_proc_ids, num_codes])

            guide = torch.cat([row0, row1, row2], axis=0)
            guide = guide + guide.t()
            guide = guide.to(device)

            guide = guide.unsqueeze(0)
            guide = guide.expand(batch_size, -1, -1)
            # print(guide)
            guide = (guide * masks.unsqueeze(-1) * masks.unsqueeze(1) + torch.eye(num_codes).to(device).unsqueeze(0))

        if self.use_prior:
            prior_idx = priors[f'indices{number}'].t()
            temp_idx = (prior_idx[:, 0] * 100000 + prior_idx[:, 1] * 1000 + prior_idx[:, 2])
            sorted_idx = torch.argsort(temp_idx)
            prior_idx = prior_idx[sorted_idx]

            prior_idx_shape = [batch_size, self.max_num_codes * 2,
                               self.max_num_codes * 2]  # ;print(prior_idx.t());print(priors['values'])
            sparse_prior = torch.sparse.FloatTensor(prior_idx.t().type(torch.LongTensor).cuda(),
                                                    priors[f'values{number}'], torch.Size(prior_idx_shape))
            prior_guide = sparse_prior.to_dense()

            visit_guide = torch.tensor([self.prior_scalar] * self.max_num_codes + [0.0] * self.max_num_codes * 1,
                                       dtype=torch.float, device=device)
            prior_guide = torch.cat([visit_guide.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1), prior_guide],
                                    axis=1)
            visit_guide = torch.cat([torch.tensor([0.0], device=device), visit_guide], axis=0)
            prior_guide = torch.cat([visit_guide.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1), prior_guide],
                                    axis=2)
            prior_guide = (prior_guide * masks.unsqueeze(-1) * masks.unsqueeze(1) + self.prior_scalar * torch.eye(
                num_codes, device=device).unsqueeze(0))
            degrees = torch.sum(prior_guide, axis=2)
            prior_guide = prior_guide / degrees.unsqueeze(-1)
        #prior_guide = None
        return guide, prior_guide

    def get_loss(self, logits, labels, attentions1, attentions2, attentions3):
        loss_fct = nn.BCELoss()
        loss = loss_fct(logits.squeeze(1).to(torch.float),
                        labels.to(torch.float))
        return loss

    def forward(self, data, priors_data, ep_nm = 0):
        embedding_dict1, mask_dict1 = self.embeddings1(data, 1)
        mask_dict1['dx_ints1'] = data['dx_masks1']
        mask_dict1['proc_ints1'] = data['proc_masks1']
        global value_id
        value_id = ep_nm
        keys1 = ['visit1'] + ['dx_ints1', 'proc_ints1']  # self.feature_keys
        hidden_states1 = torch.cat([embedding_dict1[key] for key in keys1], axis=1)
        masks1 = torch.cat([mask_dict1[key] for key in keys1], axis=1)

        guide1, prior_guide1 = self.create_matrix_vdp(data, masks1, priors_data, 1)

        all_hidden_states1 = () if self.output_hidden_states else None
        all_attentions1 = () if self.output_attentions else None
        extended_attention_mask1 = get_extended_attention_mask(masks1)
        extended_guide_mask1 = get_extended_attention_mask(guide1) if self.use_guide else None

        # ___________________________________________

        embedding_dict2, mask_dict2 = self.embeddings2(data, 2)
        mask_dict2['dx_ints2'] = data['dx_masks2']
        mask_dict2['proc_ints2'] = data['proc_masks2']

        keys2 = ['visit2'] + ['dx_ints2', 'proc_ints2']  # self.feature_keys
        hidden_states2 = torch.cat([embedding_dict2[key] for key in keys2], axis=1)
        masks2 = torch.cat([mask_dict2[key] for key in keys2], axis=1)

        guide2, prior_guide2 = self.create_matrix_vdp(data, masks2, priors_data, 2)

        all_hidden_states2 = () if self.output_hidden_states else None
        all_attentions2 = () if self.output_attentions else None
        extended_attention_mask2 = get_extended_attention_mask(masks2)
        extended_guide_mask2 = get_extended_attention_mask(guide2) if self.use_guide else None

        # ___________________________________________

        embedding_dict3, mask_dict3 = self.embeddings3(data, 3)
        mask_dict3['dx_ints3'] = data['dx_masks3']
        mask_dict3['proc_ints3'] = data['proc_masks3']

        keys3 = ['visit3'] + ['dx_ints3', 'proc_ints3']  # self.feature_keys
        hidden_states3 = torch.cat([embedding_dict3[key] for key in keys3], axis=1)
        masks3 = torch.cat([mask_dict3[key] for key in keys3], axis=1)

        guide3, prior_guide3 = self.create_matrix_vdp(data, masks3, priors_data, 3)

        all_hidden_states3 = () if self.output_hidden_states else None
        all_attentions3 = () if self.output_attentions else None
        extended_attention_mask3 = get_extended_attention_mask(masks3)
        extended_guide_mask3 = get_extended_attention_mask(guide3) if self.use_guide else None

        # ___________________________________________

        for i in range(0, len(self.layers1)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states1 + (hidden_states1,)
            if i == 0:
                layer_outputs3 = self.layers3[i](hidden_states3, extended_attention_mask3, extended_guide_mask3,
                                                 prior_guide3,
                                                 self.output_attentions, first=True)
                layer_outputs2 = self.layers2[i](hidden_states2, extended_attention_mask2, extended_guide_mask2,
                                                 prior_guide2,
                                                 self.output_attentions, first=True)
                layer_outputs1 = self.layers1[i](hidden_states1, extended_attention_mask1, extended_guide_mask1,
                                                 prior_guide1,
                                                 self.output_attentions, first=True)
                hidden_states1 = layer_outputs1[0]
                hidden_states2 = layer_outputs2[0]
                hidden_states3 = layer_outputs3[0]
            elif i == len(self.layers1) - 1:
                layer_outputs3 = self.layers3[i](hidden_states3, extended_attention_mask3, extended_guide_mask3,
                                                 prior_guide3, self.output_attentions, keyP=key_layer3,
                                                 queryP=query_layer3,
                                                 first=True)
                hidden_states3 = layer_outputs3[0][0]

                layer_outputs2 = self.layers2[i](hidden_states2, extended_attention_mask2, extended_guide_mask2,
                                                 prior_guide2, self.output_attentions, keyP=key_layer2,
                                                 queryP=query_layer2,
                                                 first=False)
                hidden_states2 = layer_outputs2[0][0]

                layer_outputs1 = self.layers1[i](hidden_states1, extended_attention_mask1, extended_guide_mask1,
                                                 prior_guide1, self.output_attentions, keyP=key_layer1,
                                                 queryP=query_layer1,
                                                 first=False)
                hidden_states1 = layer_outputs1[0][0]

            else:
                layer_outputs3 = self.layers3[i](hidden_states3, extended_attention_mask3, extended_guide_mask3,
                                                 prior_guide3, self.output_attentions, first=True, number = 3)
                hidden_states3 = layer_outputs3[0][0]
                key3 = layer_outputs3[1]
                query3 = layer_outputs3[2]
                key_layer3 = layer_outputs3[3]
                query_layer3 = layer_outputs3[4]

                layer_outputs2 = self.layers2[i](hidden_states2, extended_attention_mask2, extended_guide_mask2,
                                                 prior_guide2, self.output_attentions, keyP=key3, queryP=query3,
                                                 first=False, number = 2)
                hidden_states2 = layer_outputs2[0][0]
                key2 = layer_outputs2[1]
                query2 = layer_outputs2[2]
                key_layer2 = layer_outputs2[3]
                query_layer2 = layer_outputs2[4]

                layer_outputs1 = self.layers1[i](hidden_states1, extended_attention_mask1, extended_guide_mask1,
                                                 prior_guide1, self.output_attentions, keyP=key2, queryP=query2,
                                                 first=False, number = 1)
                hidden_states1 = layer_outputs1[0][0]
                key1 = layer_outputs1[1]
                query1 = layer_outputs1[2]
                key_layer1 = layer_outputs1[3]
                query_layer1 = layer_outputs1[4]
            # inja bebad naghese

            if self.output_attentions:
                if i == 0:
                    all_attentions1 = all_attentions1 + (layer_outputs1[1],)
                    all_attentions2 = all_attentions2 + (layer_outputs2[1],)
                    all_attentions3 = all_attentions3 + (layer_outputs3[1],)
                else:
                    all_attentions1 = all_attentions1 + (layer_outputs1[0][1],)
                    all_attentions2 = all_attentions2 + (layer_outputs2[0][1],)
                    all_attentions3 = all_attentions3 + (layer_outputs3[0][1],)

        if self.output_hidden_states:
            all_hidden_states1 = all_hidden_states1 + (hidden_states1,)
            all_hidden_states2 = all_hidden_states2 + (hidden_states2,)
            all_hidden_states3 = all_hidden_states3 + (hidden_states3,)

        pooled_output1 = self.pooler1(hidden_states1)
        pooled_output2 = self.pooler1(hidden_states2)
        pooled_output3 = self.pooler1(hidden_states3)

        pooled_output1 = self.dropout1(pooled_output1)
        pooled_output2 = self.dropout2(pooled_output2)
        pooled_output3 = self.dropout3(pooled_output3)
        pooled_output1 = pooled_output1.unsqueeze(1)
        pooled_output2 = pooled_output2.unsqueeze(1)
        pooled_output3 = pooled_output3.unsqueeze(1)
        third_tensor = torch.cat((pooled_output3, pooled_output2), 1)
        pooled = torch.cat((third_tensor, pooled_output1), 1)
        pooled = self.retain(pooled, value_id)
        logits = pooled

        loss = self.get_loss(logits, data[self.label_key], all_attentions1, all_attentions2, all_attentions3)

        return tuple(v for v in [loss, logits, all_hidden_states1, all_attentions1] if
                     v is not None)



