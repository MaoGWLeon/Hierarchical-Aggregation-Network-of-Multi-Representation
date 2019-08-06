from torch import nn
import torch
from torch.nn import init
import torch.nn.functional as F
import pickle
import numpy as np
import time

conf = {
    "batch_size": 100,
    "_EOS_": 28270,
    "max_turn_num": 10,
    "max_turn_len": 50,
    "word_embedding_size":200,
    "rnn_units":100,
    "DAM_total_words":434513
}



"""
1) DAM_embedding_file is the path of word embedding file. In our model, we use the embedding file of DAM, 
the download path: https://pan.baidu.com/s/1hakfuuwdS8xl7NyxlWzRiQ
2) Beside the embedding file, we also use the same data format as DAM. 

"""
DAM_embedding_file=""

class HAMR(nn.Module):
    def __init__(self, config=None):
        # 标准动作
        super(HAMR, self).__init__()

        self.word_embedding_size = conf["word_embedding_size"]
        self.rnn_units = conf["rnn_units"]
        self.total_words = conf["DAM_total_words"]

        self.len_max_seq = conf["max_turn_len"]
        self.d_q = conf["q_size"]
        self.d_model = conf["model_size"]
        self.d_hidden = conf["hidden_size"]

        self.max_turn_num = conf["max_turn_num"]
        print(f"max_turn_num:{self.max_turn_num}--------------------------")

        with open(DAM_embedding_file, 'rb') as f:
            embedding_matrix = pickle.load(f, encoding="bytes")
            assert embedding_matrix.shape == (434513, 200)  # 这个假设是基于DAM的词向量
        self.word_embedding = nn.Embedding(self.total_words, self.word_embedding_size)
        self.word_embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.word_embedding.weight.requires_grad = False

        #first GRU
        self.utterance_GRU = nn.GRU(self.word_embedding_size, self.rnn_units, bidirectional=True, batch_first=True)
        ih_u = (param.data for name, param in self.utterance_GRU.named_parameters() if 'weight_ih' in name)
        hh_u = (param.data for name, param in self.utterance_GRU.named_parameters() if 'weight_hh' in name)
        for k in ih_u:
            nn.init.orthogonal_(k)
        for k in hh_u:
            nn.init.orthogonal_(k)

        #second GRU
        self.utt_second_GRU = nn.GRU(self.rnn_units * 2, self.rnn_units, bidirectional=True, batch_first=True)
        ih_u = (param.data for name, param in self.utt_second_GRU.named_parameters() if 'weight_ih' in name)
        hh_u = (param.data for name, param in self.utt_second_GRU.named_parameters() if 'weight_hh' in name)
        for k in ih_u:
            nn.init.orthogonal_(k)
        for k in hh_u:
            nn.init.orthogonal_(k)

        self.emb_weight = nn.Parameter(torch.zeros(3))
        self.emb_scale = nn.Parameter(torch.ones(1))

        self.weight_1 = nn.Parameter(torch.Tensor(self.len_max_seq, self.len_max_seq * 5))
        self.bais_1 = nn.Parameter(torch.Tensor(200))
        init.kaiming_uniform_(self.weight_1)
        init.zeros_(self.bais_1)

        self.enc_GRU = nn.GRU(self.rnn_units * 2, self.rnn_units * 2, bidirectional=False, batch_first=True)
        ih_f = (param.data for name, param in self.enc_GRU.named_parameters() if 'weight_ih' in name)
        hh_f = (param.data for name, param in self.enc_GRU.named_parameters() if 'weight_hh' in name)
        for k in ih_f:
            nn.init.orthogonal_(k)
        for k in hh_f:
            nn.init.orthogonal_(k)

        self.enc_linear = nn.Linear(50, 1)

        # final GRU
        self.final_GRU = nn.GRU(self.rnn_units * 2, self.rnn_units * 2, bidirectional=False, batch_first=True)
        ih_f = (param.data for name, param in self.final_GRU.named_parameters() if 'weight_ih' in name)
        hh_f = (param.data for name, param in self.final_GRU.named_parameters() if 'weight_hh' in name)
        for k in ih_f:
            nn.init.orthogonal_(k)
        for k in hh_f:
            nn.init.orthogonal_(k)

        self.weight_linear = nn.Linear(self.max_turn_num + 1, 1)

        self.final_linear = nn.Linear(self.rnn_units * 2, 1)
        final_linear_weight = (param.data for name, param in self.final_linear.named_parameters() if "weight" in name)
        for w in final_linear_weight:
            init.xavier_uniform_(w)

    def forward(self, utterance, response):
        '''
            utterance:(batch_size, max_turn_num, max_turn_len)
            response:(batch_size, max_turn_len)
        '''
        res_emb = self.word_embedding(response)
        res_GRU_emb, _ = self.utterance_GRU(res_emb)
        res_second_GRU_emb, _ = self.utt_second_GRU(res_GRU_emb)

        softmax_weight = F.softmax(self.emb_weight)
        final_res_emb = softmax_weight[0] * res_emb + softmax_weight[1] * res_GRU_emb + \
                        softmax_weight[2] * res_second_GRU_emb
        final_res_emb = self.emb_scale * final_res_emb

        all_utt_emb = []
        utterance = utterance.permute(1, 0, 2)
        for index, each_utterance in enumerate(utterance):
            utt_emb = self.word_embedding(each_utterance)
            utt_GRU_emb, _ = self.utterance_GRU(utt_emb)
            utt_second_GRU_emb, _ = self.utt_second_GRU(utt_GRU_emb)

            final_utt_emb = softmax_weight[0] * utt_emb + softmax_weight[1] * utt_GRU_emb + \
                            softmax_weight[2] * utt_second_GRU_emb
            final_utt_emb = self.emb_scale * final_utt_emb
            all_utt_emb.append(final_utt_emb)

        enc_matching_vectors = []
        for utt_emb in all_utt_emb:
            matrix1 = torch.matmul(utt_emb, final_res_emb.permute(0, 2, 1))
            matrix1 = F.softmax(matrix1, dim=-1)
            utt_emb_attend_RES = torch.matmul(matrix1, final_res_emb)
            A_reduce_B = utt_emb - utt_emb_attend_RES
            A_multi_B = utt_emb * utt_emb_attend_RES

            FIX_utt_emb = torch.matmul(self.weight_1, torch.cat(
                [utt_emb, utt_emb_attend_RES, A_reduce_B, A_reduce_B * A_reduce_B, A_multi_B], 1)) + self.bais_1

            FIX_utt_emb = F.relu(FIX_utt_emb)

            _, last_enc_UTT_emb = self.enc_GRU(FIX_utt_emb)

            enc_matching_vectors.append(torch.squeeze(last_enc_UTT_emb))

        all_utt_emb = torch.cat(all_utt_emb, 1)
        matrix1 = torch.matmul(final_res_emb, all_utt_emb.permute(0, 2, 1))
        matrix1 = F.softmax(matrix1, dim=-1)
        res_emb_attend_UTT = torch.matmul(matrix1, all_utt_emb)
        A_reduce_B = final_res_emb - res_emb_attend_UTT
        A_multi_B = final_res_emb * res_emb_attend_UTT

        FIX_res_emb = torch.matmul(self.weight_1, torch.cat(
            [final_res_emb, res_emb_attend_UTT, A_reduce_B, A_reduce_B * A_reduce_B, A_multi_B], 1)) + self.bais_1
        FIX_res_emb = F.relu(FIX_res_emb)
        _, last_enc_RES_emb = self.enc_GRU(FIX_res_emb)
        enc_matching_vectors.append(torch.squeeze(last_enc_RES_emb))

        # (batch size,10,50)->(batch size,10,100)
        every_hidden, last_hidden = self.final_GRU(torch.stack(enc_matching_vectors, dim=1))
        # last_hidden = torch.squeeze(last_hidden)

        every_hidden = self.weight_linear(every_hidden.permute(0, 2, 1))
        every_hidden = torch.squeeze(every_hidden)
        # every_hidden = F.relu(every_hidden)
        logits = self.final_linear(every_hidden)
        y_pred = logits.view(logits.size(0))

        return y_pred