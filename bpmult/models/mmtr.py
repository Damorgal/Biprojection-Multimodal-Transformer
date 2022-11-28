import math
import torch
from torch import nn
import torch.nn.functional as F
from bpmult.models.image import ImageEncoder
from pytorch_pretrained_bert.modeling import BertModel
from transformers import BertModel as huggingBertModel
from bpmult.models.transformer import TransformerEncoder

#MAG Module is not used, but it tries to substitute the GMU module
class MAG(nn.Module):
    def __init__(self, hidden_size, beta_shift, dropout_prob):
        super(MAG, self).__init__()
        
        VISUAL_DIM = ACOUSTIC_DIM = TEXT_DIM = 768
        self.W_hv = nn.Linear(VISUAL_DIM + TEXT_DIM, TEXT_DIM)
        self.W_ha = nn.Linear(ACOUSTIC_DIM + TEXT_DIM, TEXT_DIM)

        self.W_v = nn.Linear(VISUAL_DIM, TEXT_DIM)
        self.W_a = nn.Linear(ACOUSTIC_DIM, TEXT_DIM)
        self.beta_shift = beta_shift

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, text_embedding, visual, acoustic):
        eps = 1e-6
        weight_v = F.relu(self.W_hv(torch.cat((visual, text_embedding), dim=-1)))
        weight_a = F.relu(self.W_ha(torch.cat((acoustic, text_embedding), dim=-1)))

        h_m = weight_v * self.W_v(visual) + weight_a * self.W_a(acoustic)

        em_norm = text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).cuda()#to(DEVICE)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True).cuda()#to(DEVICE)

        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        acoustic_vis_embedding = alpha * h_m

        embedding_output = self.dropout(
            self.LayerNorm(acoustic_vis_embedding + text_embedding)
        )

        return embedding_output

#Audio enconders ------------------------------------------
class AudioEncoderLarge(nn.Module):
    def __init__(self, args):
        super(AudioEncoderLarge, self).__init__()
        self.args = args
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(128, 128, 128, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, 128, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, 128, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, 128, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        #x = self.conv4(x)
        return x
    
    
class AudioEncoder(nn.Module):
    def __init__(self, args):
        super(AudioEncoder, self).__init__()
        self.args = args
        
        conv_layers = []
        
        conv_layers.append(nn.Conv1d(96, 96, 128, stride=2))
        conv_layers.append(nn.Conv1d(96, 96, 128, stride=2))
        conv_layers.append(nn.AdaptiveAvgPool1d(200))
        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x
        
class AudioEncoder_cmumosei(nn.Module):
    def __init__(self, args):
        super(AudioEncoder_cmumosei, self).__init__()
        self.args = args
        
        conv_layers = []
        
        conv_layers.append(nn.Conv1d(74, 74, 5, stride=2))
        conv_layers.append(nn.Conv1d(74, 74, 5, stride=2))
        conv_layers.append(nn.AdaptiveAvgPool1d(20))
        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x
        
class AudioEncoder_cmumosi(nn.Module):
    def __init__(self, args):
        super(AudioEncoder_cmumosi, self).__init__()
        self.args = args
        
        conv_layers = []
        conv_layers.append(nn.Conv1d(5, 5, 20, stride=1))
        conv_layers.append(nn.Conv1d(5, 5, 20, stride=1))
        conv_layers.append(nn.AdaptiveAvgPool1d(5))
        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x

# Bert encoder -------------------------------------
class BertEncoder(nn.Module):
    def __init__(self, args):
        super(BertEncoder, self).__init__()
        self.args = args
        self.bert = huggingBertModel.from_pretrained(args.bert_model)

    def forward(self, txt, mask, segment):
        
        encoded_layers, out = self.bert(
            input_ids=txt,
            token_type_ids=segment,
            attention_mask=mask,
            return_dict=False,
        )
        return encoded_layers

#GMUs Modules --------------------------------------
class GatedMultimodalLayer(nn.Module):
    """ Gated Multimodal Layer based on 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """
    def __init__(self, size_in1, size_in2, size_out):
        super(GatedMultimodalLayer, self).__init__()
        self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out
        
        self.hidden1 = nn.Linear(size_in1, size_out, bias=False)
        self.hidden2 = nn.Linear(size_in2, size_out, bias=False)
        self.x_gate = nn.Linear(size_in1+size_in2, size_out, bias=False)

    def forward(self, xs):
        h1 = torch.tanh(self.hidden1(xs[0]))
        h2 = torch.tanh(self.hidden2(xs[1]))
        x_cat = torch.cat(xs, dim=-1)
        z = torch.sigmoid(self.x_gate(x_cat))

        return z*h1 + (1-z)*h2, torch.cat((z, (1-z)), dim=-1)
        
class GatedMultimodalLayerFeatures(nn.Module):
    """ FUSION Gated Multimodal Layer based on 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) proposed by us"""
    def __init__(self, size_in1, size_in2, size_out):
        super(GatedMultimodalLayerFeatures, self).__init__()
        self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out
        
        self.hidden1 = nn.Linear(size_in1, size_out, bias=False)
        self.hidden2 = nn.Linear(size_in2, size_out, bias=False)
        self.x_gate = nn.Linear(size_in1+size_in2, size_out, bias=False)

    def forward(self, xs):
        h1 = torch.tanh(self.hidden1(xs[0]))
        h2 = torch.tanh(self.hidden2(xs[1]))
        x_cat = torch.cat(xs, dim=-1)
        z = torch.sigmoid(self.x_gate(x_cat))
        
        return z*h1*xs[0] + (1-z)*h2*xs[1], torch.cat((z, (1-z)), dim=-1)

class TextShifting4Layer(nn.Module):
    """ Layer inspired by 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """
    def __init__(self, size_in1, size_in2, size_in3, size_in4, size_out):
        super(TextShifting4Layer, self).__init__()
        self.size_in1, self.size_in2, self.size_in3, self.size_in4, self.size_out = size_in1, size_in2, size_in3, size_in4, size_out
        
        self.hidden1 = nn.Linear(size_in1, size_out, bias=False)
        self.hidden2 = nn.Linear(size_in2, size_out, bias=False)
        self.hidden3 = nn.Linear(size_in3, size_out, bias=False)
        self.hidden4 = nn.Linear(size_in4, size_out, bias=False)
        self.x1_gate = nn.Linear(size_in1+size_in2+size_in3+size_in4, size_out, bias=False)
        self.x2_gate = nn.Linear(size_in1+size_in2+size_in3+size_in4, size_out, bias=False)
        self.x3_gate = nn.Linear(size_in1+size_in2+size_in3+size_in4, size_out, bias=False)
        self.x4_gate = nn.Linear(size_in1+size_in2+size_in3+size_in4, size_out, bias=False)

    def forward(self, xs):
        h1 = torch.tanh(self.hidden1(xs[0]))
        h2 = torch.tanh(self.hidden2(xs[1]))
        h3 = torch.tanh(self.hidden3(xs[2]))
        h4 = torch.tanh(self.hidden4(xs[3]))
        x_cat = torch.cat(xs, dim=-1)
        z1 = torch.sigmoid(self.x1_gate(x_cat))
        z2 = torch.sigmoid(self.x2_gate(x_cat))
        z3 = torch.sigmoid(self.x3_gate(x_cat))
        z4 = torch.sigmoid(self.x4_gate(x_cat))

        return z1*h1 + z2*h2 + z3*h3 + z4*h4, torch.cat((z1, z2, z3, z4), dim=-1)

class TextShifting5Layer(nn.Module):
    """ Layer inspired by 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """
    def __init__(self, size_in1, size_in2, size_in3, size_in4, size_in5, size_out):
        super(TextShifting5Layer, self).__init__()
        self.size_in1, self.size_in2, self.size_in3, self.size_in4, self.size_in5, self.size_out = size_in1, size_in2, size_in3, size_in4, size_in5, size_out
        
        self.hidden1 = nn.Linear(size_in1, size_out, bias=False)
        self.hidden2 = nn.Linear(size_in2, size_out, bias=False)
        self.hidden3 = nn.Linear(size_in3, size_out, bias=False)
        self.hidden4 = nn.Linear(size_in4, size_out, bias=False)
        self.hidden5 = nn.Linear(size_in5, size_out, bias=False)
        self.x1_gate = nn.Linear(size_in1+size_in2+size_in3+size_in4+size_in5, size_out, bias=False)
        self.x2_gate = nn.Linear(size_in1+size_in2+size_in3+size_in4+size_in5, size_out, bias=False)
        self.x3_gate = nn.Linear(size_in1+size_in2+size_in3+size_in4+size_in5, size_out, bias=False)
        self.x4_gate = nn.Linear(size_in1+size_in2+size_in3+size_in4+size_in5, size_out, bias=False)
        self.x5_gate = nn.Linear(size_in1+size_in2+size_in3+size_in4+size_in5, size_out, bias=False)

    def forward(self, x1, x2, x3, x4, x5):
        h1 = F.tanh(self.hidden1(x1))
        h2 = F.tanh(self.hidden2(x2))
        h3 = F.tanh(self.hidden3(x3))
        h4 = F.tanh(self.hidden4(x4))
        h5 = F.tanh(self.hidden5(x5))
        x_cat = torch.cat((x1, x2, x3, x4, x5), dim=1)
        z1 = F.sigmoid(self.x1_gate(x_cat))
        z2 = F.sigmoid(self.x2_gate(x_cat))
        z3 = F.sigmoid(self.x3_gate(x_cat))
        z4 = F.sigmoid(self.x4_gate(x_cat))
        z5 = F.sigmoid(self.x5_gate(x_cat))

        return z1*h1 + z2*h2 + z3*h3 + z4*h4 + z5*h5, torch.cat((z1, z2, z3, z4, z5), dim=1)


class TextShiftingNLayer(nn.Module):
    """ Layer inspired by 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """
    def __init__(self, sizes_in, size_out):
        super(TextShiftingNLayer, self).__init__()
        self.sizes_in, self.size_out = sizes_in, size_out
        
        self.hiddens = nn.ModuleList([nn.Linear(size_i, size_out, bias=False) for size_i in sizes_in])
        self.x_gates = nn.ModuleList([nn.Linear(sum(sizes_in), size_out, bias=False) for i in range(len(sizes_in))])

    def forward(self, *xs):
        h = []
        for x, hidden in zip(xs, self.hiddens):
            h.append(torch.tanh(hidden(x)))
        
        x_cat = torch.cat(xs, dim=-1)
        
        z = []
        for x, gate in zip(xs, self.x_gates):
            z.append(torch.sigmoid(gate(x_cat)))
        
        fused = h[0]*z[0]
        for h_i, z_i in zip(h[1:], z[1:]):
            fused += h_i*z_i

        return fused, torch.cat(z, dim=-1)

       
# The official model-----------------------------
class MultiprojectionMMTransformerGMUClf(nn.Module):
    def __init__(self, args):
        """
        The official BPMulT model with a BERT preprocessing the text, and using video, audio, and poster. (three modalities and one extra information)
        """
        super(MultiprojectionMMTransformerGMUClf, self).__init__()
        self.args = args
        self.orig_d_l, self.orig_d_v, self.orig_d_a, self.orig_d_p = args.orig_d_l, args.orig_d_v, args.orig_d_a, args.orig_d_p
        self.d_l, self.d_a, self.d_v, self.d_m = args.hidden_sz, args.hidden_sz, args.hidden_sz, 0#768, 768, 768, 768
        self.vonly = args.vonly
        self.lonly = args.lonly
        self.aonly = args.aonly
        self.num_heads = args.num_heads
        self.layers = args.layers
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_v = args.attn_dropout_v
        self.attn_dropout_a = args.attn_dropout_a
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.out_dropout = args.out_dropout
        self.embed_dropout = args.embed_dropout
        self.attn_mask = args.attn_mask
        self.hybrid = args.hybrid
        
        combined_dim = args.hidden_sz#768 # For GMU
        output_dim = args.n_classes        # This is actually not a hyperparameter :-)
        self.reduced_dim = 32
        
        self.enc = BertEncoder(args)
#Comment following line to IMDb
        self.audio_enc = AudioEncoder(args)

        # 0. Project poster feature to 768 dim
        self.proj_poster  = nn.Linear(self.orig_d_p, self.d_v, bias=False)
        
        #--- GMU instead of sum
        #------ GMU Middle
        self.gmu_l_m = GatedMultimodalLayerFeatures(self.d_v, self.d_a, self.d_l)
        #self.gmu_l_m = GatedMultimodalUnitSoftmaxFusion(2, combined_dim, combined_dim, probs=False, residual=False)
        self.gmu_v_m = GatedMultimodalLayerFeatures(self.d_l, self.d_a, self.d_v)
        #self.gmu_v_m = GatedMultimodalUnitSoftmaxFusion(2, combined_dim, combined_dim, probs=False, residual=False)
        self.gmu_a_m = GatedMultimodalLayerFeatures(self.d_l, self.d_v, self.d_a)
        #self.gmu_a_m = GatedMultimodalUnitSoftmaxFusion(2, combined_dim, combined_dim, probs=False, residual=False)
        #------ GMU Top
        self.gmu_l = GatedMultimodalLayerFeatures(self.d_v, self.d_a, self.d_l)
        #self.gmu_l = GatedMultimodalUnitSoftmaxFusion(2, combined_dim, combined_dim, probs=False, residual=False)
        self.gmu_v = GatedMultimodalLayerFeatures(self.d_l, self.d_a, self.d_v)
        #self.gmu_v = GatedMultimodalUnitSoftmaxFusion(2, combined_dim, combined_dim, probs=False, residual=False)
        self.gmu_a = GatedMultimodalLayerFeatures(self.d_l, self.d_v, self.d_a)
        #self.gmu_a = GatedMultimodalUnitSoftmaxFusion(2, combined_dim, combined_dim, probs=False, residual=False)
        
        if self.hybrid:
            #self.gmu_early = GatedMultimodalUnitSoftmaxFusion(3, combined_dim, combined_dim, residual=False)
            self.gmu_early = TextShifting3Layer(self.d_l, self.d_v, self.d_a, combined_dim)
            self.gmu_early = TextShifting3Layer(self.reduced_dim, self.reduced_dim, self.reduced_dim, combined_dim)
        
        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
            self.trans_l_with_v2a = self.get_network(self_type='lv2a', biprojection=True)
            self.trans_l_with_a2v = self.get_network(self_type='la2v', biprojection=True)
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')
            self.trans_v_with_l2a = self.get_network(self_type='vl2a', biprojection=True)
            self.trans_v_with_a2l = self.get_network(self_type='va2l', biprojection=True)
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
            self.trans_a_with_v2l = self.get_network(self_type='av2l', biprojection=True)
            self.trans_a_with_l2v = self.get_network(self_type='al2v', biprojection=True)
        
        #MAG
        #beta_shift, dropout_prob = 1e-3, 0.5
        #self.MAG_v = MAG(self.d_v, beta_shift, dropout_prob)
        #self.MAG_a = MAG(self.d_a, beta_shift, dropout_prob)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        
        if self.hybrid:
            self.gmu = TextShiftingNLayer([combined_dim]*5, combined_dim)
        else:
            #self.gmu = GatedMultimodalUnitSoftmaxFusion(4, combined_dim, combined_dim)
            self.gmu = TextShifting4Layer(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l)
        
        self.num_vectors_l = 512
#Comment following line to moviescope/IMDb/mosei_noalign
                                    # 201 512 400
        self.num_vectors_a = 200#512#400
#Comment following line to moviescope/IMDb/mosei_noalign
                                    # 201 512 400
        self.num_vectors_v = 200#512#500
        #Transformation dimension layers
#Comment following lines to just pass a selected token
        self.transfm_a2l = nn.Linear(self.num_vectors_a, self.num_vectors_l)
        self.transfm_v2l = nn.Linear(self.num_vectors_v, self.num_vectors_l)
        self.transfm_l2a = nn.Linear(self.num_vectors_l, self.num_vectors_a)
        self.transfm_l2v = nn.Linear(self.num_vectors_l, self.num_vectors_v)
     #   self.transfm_v2a = nn.Linear(self.num_vectors_v, self.num_vectors_a)
      #  self.transfm_a2v = nn.Linear(self.num_vectors_a, self.num_vectors_v)
        
        if self.hybrid:
            self.trans_l_early = self.get_network(self_type='l_mem', layers=3)
            self.trans_v_early = self.get_network(self_type='v_mem', layers=3)
            self.trans_a_early = self.get_network(self_type='a_mem', layers=3)
            self.proj_l_e = nn.Linear(self.num_vectors_l, self.reduced_dim, bias=False)
            self.proj_v_e = nn.Linear(self.num_vectors_v, self.reduced_dim, bias=False)
            self.proj_a_e = nn.Linear(self.num_vectors_a, self.reduced_dim, bias=False)
            #self.proj_l_e = nn.Linear(combined_dim, self.reduced_dim, bias=False)
            #self.proj_v_e = nn.Linear(combined_dim, self.reduced_dim, bias=False)
            #self.proj_a_e = nn.Linear(combined_dim, self.reduced_dim, bias=False)

    def get_network(self, self_type='l', layers=-1, biprojection=False):
        if self_type in ['l', 'al', 'vl', 'av2l', 'va2l']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va', 'lv2a', 'vl2a']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av', 'la2v', 'al2v']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
        # With GMU
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        # Without GMU (normal concat)
            #embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
            #embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
            #embed_dim, attn_dropout = 2*self.d_v, self.attn_dropout
        elif self_type == 'p_mem':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask,
                                  biprojection=biprojection)

    def transfm_2dim(self, x_t, dim, out_dim):
        if x_t.size(dim) != out_dim:
            if dim == 2:
                y  = torch.zeros((x_t.size(0), x_t.size(1), out_dim-x_t.size(2))).cuda()
            elif dim == 1:
                y  = torch.zeros((x_t.size(0), out_dim-x_t.size(1), x_t.size(2))).cuda()
            elif dim == 0:
                y  = torch.zeros((out_dim-x_t.size(0), x_t.size(1), x_t.size(2))).cuda()
            x_t = torch.cat((x_t, y), dim)
        
        return x_t
        
            
    def forward(self, txt, mask, segment, img, audio, poster, output_gate=False):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_l = self.enc(txt, mask, segment)
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v = img.transpose(1, 2)
#Comment following lines to IMDb
        x_a = self.audio_enc(audio)
#        x_a = audio.transpose(1, 2) #self.audio_enc(audio)
        
        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        
        
        if proj_x_l.size(0) != self.num_vectors_l:
            proj_x_l = self.transfm_2dim(proj_x_l, 0, self.num_vectors_l)
        if proj_x_a.size(0) != self.num_vectors_a:
            proj_x_a = self.transfm_2dim(proj_x_a, 0, self.num_vectors_a)
        if proj_x_v.size(0) != self.num_vectors_v:
            proj_x_v = self.transfm_2dim(proj_x_v, 0, self.num_vectors_v)
        
        
        #Parallel fusion
        if self.hybrid:
            proj_x_l_e = self.proj_l_e(proj_x_l.permute(2, 1, 0)).permute(2, 1, 0)
            proj_x_a_e = self.proj_a_e(proj_x_a.permute(2, 1, 0)).permute(2, 1, 0)
            proj_x_v_e = self.proj_v_e(proj_x_v.permute(2, 1, 0)).permute(2, 1, 0)
            h_l = self.trans_l_early(proj_x_l_e)
            h_a = self.trans_a_early(proj_x_a_e)
            h_v = self.trans_v_early(proj_x_v_e)
            last_hl_early = h_l[0] + h_l[-1]#, zx = self.gmu_early([h_l, h_v, h_a])
            last_ha_early = h_a[0] + h_a[-1]
            last_hv_early = h_v[0] + h_v[-1]
            last_h_early, zx = self.gmu_early(last_hl_early, last_hv_early, last_ha_early)
        
#Comment following lines to IMDb
        poster   = self.proj_poster(poster)
        #poster   = self.proj_poster(poster).squeeze(1)#.transpose(1, 0)

        # First Crossmodal Transformers
        # (V,A) --> L
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a) # Dimension (V, N, d_v)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v) # Dimension (A, N, d_a)
        # (L,V) --> A
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
        # (L,A) --> V
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
        
        # Biprojections
        if self.lonly:
            # Biprojection ---------
            h_l_with_v2a = self.trans_l_with_v2a(proj_x_l, h_a_with_vs, h_a_with_vs) # Dimension (L, N, d_l)
            h_l_with_a2v = self.trans_l_with_a2v(proj_x_l, h_v_with_as, h_v_with_as) # Dimension (L, N, d_l)
            
            # GMU Middle --------
            t_h_a_with_vs =  self.transfm_a2l(h_a_with_vs.permute(2,1,0)).permute(2,1,0)#h_a_with_vs[tok_a]
            t_h_v_with_as =  self.transfm_v2l(h_v_with_as.permute(2,1,0)).permute(2,1,0)#[tok_v])
            h_l_gmu, z1_l = self.gmu_l_m([t_h_v_with_as, t_h_a_with_vs])
            
            # Residual conection level 1 to 2 -------
            h_l_with_v2a_tot = h_l_with_v2a + t_h_a_with_vs
            h_l_with_a2v_tot = h_l_with_a2v + t_h_v_with_as
            
            # Feature Fusion ---------
            h_ls_gmu, z2_l = self.gmu_l([h_l_with_a2v_tot, h_l_with_v2a_tot])
            
            # Residual conection level 1 to 3 -------
            h_ls_gmu += h_l_gmu

            #We are passing also the first token because we use the BERT encoder for text and the first token corresponds to [CLS]
            last_h_l = last_hs = h_ls_gmu[0] + h_ls_gmu[-1]

        if self.aonly:
            # Biprojection ---------
            h_a_with_v2l = self.trans_a_with_v2l(proj_x_a, h_l_with_vs, h_l_with_vs)
            h_a_with_l2v = self.trans_a_with_l2v(proj_x_a, h_v_with_ls, h_v_with_ls)
            # GMU Middle --------
            
            t_h_l_with_vs = self.transfm_l2a(h_l_with_vs.permute(2,1,0)).permute(2,1,0)
            t_h_v_with_ls = h_v_with_ls
            h_a_gmu, z1_a = self.gmu_a_m([t_h_l_with_vs, t_h_v_with_ls])
            
            # Residual conection level 1 to 2 -------
            h_a_with_v2l_tot = h_a_with_v2l + t_h_l_with_vs
            h_a_with_l2v_tot = h_a_with_l2v + t_h_v_with_ls
            
            # Feature Fusion ---------
            h_as_gmu, z2_a = self.gmu_a([h_a_with_v2l_tot, h_a_with_l2v_tot])
            
            # Residual conection level 1 to 3 -------
            h_as_gmu += h_a_gmu
            
            #We are passing also the first token because we use the BERT encoder for text and the first token corresponds to [CLS]
            last_h_a = last_hs = h_as_gmu[0] + h_as_gmu[-1]

        if self.vonly:
            # Biprojection ---------
            h_v_with_a2l = self.trans_v_with_a2l(proj_x_v, h_l_with_as, h_l_with_as)
            h_v_with_l2a = self.trans_v_with_l2a(proj_x_v, h_a_with_ls, h_a_with_ls)
            # GMU Middle --------
            
            t_h_l_with_as = self.transfm_l2v(h_l_with_as.permute(2,1,0)).permute(2,1,0)
            t_h_a_with_ls = h_a_with_ls
            h_v_gmu, z1_v = self.gmu_v_m([t_h_l_with_as, t_h_a_with_ls])
            
            # Residual conection level 1 to 2 -------
            h_v_with_a2l_tot = h_v_with_a2l + t_h_l_with_as
            h_v_with_l2a_tot = h_v_with_l2a + t_h_a_with_ls
            
            # Feature Fusion ---------
            h_vs_gmu, z2_v = self.gmu_v([h_v_with_a2l_tot, h_v_with_l2a_tot])
            
            # Residual conection level 1 to 3 -------
            h_vs_gmu += h_v_gmu
            
            #We are passing also the first token because we use the BERT encoder for text and the first token corresponds to [CLS]
            last_h_v = last_hs = h_vs_gmu[0] + h_vs_gmu[-1]
        
        
        if self.hybrid:
            last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a, poster, last_h_early)
        else:
            last_hs, z = self.gmu([last_h_l, last_h_v, last_h_a, poster])
       
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        if output_gate:
            return self.out_layer(last_hs_proj), z
        else:
            return self.out_layer(last_hs_proj)


# The official model just for 3 classes (MOSEI...)
class MultiprojectionMMTransformer3DGMUClf(nn.Module):
    def __init__(self, args):
        """
        The official BPMulT model with using text, video, and audio. (three modalities that could be the same in different channels)
        """
        super(MultiprojectionMMTransformer3DGMUClf, self).__init__()
        self.args = args
        self.orig_d_l, self.orig_d_v, self.orig_d_a = args.orig_d_l, args.orig_d_v, args.orig_d_a
        self.d_l, self.d_a, self.d_v = args.hidden_sz, args.hidden_sz, args.hidden_sz#768, 768, 768, 768
        self.low_dim = 32
        self.vonly = args.vonly
        self.lonly = args.lonly
        self.aonly = args.aonly
        self.num_heads = args.num_heads
        self.layers = args.layers
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_v = args.attn_dropout_v
        self.attn_dropout_a = args.attn_dropout_a
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.out_dropout = args.out_dropout
        self.embed_dropout = args.embed_dropout
        self.attn_mask = args.attn_mask
        self.hybrid = args.hybrid
        
        combined_dim = args.hidden_sz # For GMU
        output_dim = args.n_classes        # This is actually not a hyperparameter :-)

        
        self.enc = BertEncoder(args)
#Comment following line to IMDb
        #self.audio_enc = AudioEncoder(args)
        
        #--- GMU instead of sum
        #------ GMU Middle
        self.gmu_l_m = GatedMultimodalLayerFeatures(self.d_v, self.d_a, self.d_l)
        self.gmu_v_m = GatedMultimodalLayerFeatures(self.d_l, self.d_a, self.d_v)
        self.gmu_a_m = GatedMultimodalLayerFeatures(self.d_l, self.d_v, self.d_a)
        #------ GMU Top
        self.gmu_l = GatedMultimodalLayerFeatures(self.d_v, self.d_a, self.d_l)
        self.gmu_v = GatedMultimodalLayerFeatures(self.d_l, self.d_a, self.d_v)
        self.gmu_a = GatedMultimodalLayerFeatures(self.d_l, self.d_v, self.d_a)
        
        if self.hybrid:
            self.gmu_early = TextShifting3Layer(combined_dim, combined_dim, combined_dim, combined_dim)
        
        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
            self.trans_l_with_v2a = self.get_network(self_type='lv2a')
            self.trans_l_with_a2v = self.get_network(self_type='la2v')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')
            self.trans_v_with_l2a = self.get_network(self_type='vl2a')
            self.trans_v_with_a2l = self.get_network(self_type='va2l')
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
            self.trans_a_with_v2l = self.get_network(self_type='av2l')
            self.trans_a_with_l2v = self.get_network(self_type='al2v')
        
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
       
        if self.hybrid:
            self.gmu = TextShiftingNLayer([combined_dim]*4, combined_dim)
        else:
            self.gmu = TextShifting3Layer(self.d_l, self.d_v, self.d_a, self.d_l)
        self.num_vectors_l = 512
#Comment following line to moviescope/IMDb/mosei_noalign
                                    # 201 512 400
        self.num_vectors_a = 512#500#512#400
#Comment following line to moviescope/IMDb/mosei_noalign
                                    # 201 512 400
        self.num_vectors_v = 512#500#512#500
        #Transformation dimension layers
#Comment following lines to just pass a selected token
        self.transfm_a2l = nn.Linear(self.num_vectors_a, self.num_vectors_l)
        self.transfm_v2l = nn.Linear(self.num_vectors_v, self.num_vectors_l)
        self.transfm_l2a = nn.Linear(self.num_vectors_l, self.num_vectors_a)
        self.transfm_l2v = nn.Linear(self.num_vectors_l, self.num_vectors_v)
        #self.transfm_v2a = nn.Linear(self.num_vectors_v, self.num_vectors_a)
        #self.transfm_a2v = nn.Linear(self.num_vectors_a, self.num_vectors_v)
        
        if self.hybrid:
            self.trans_l_early = self.get_network(self_type='l_mem', layers=3)
            self.trans_v_early = self.get_network(self_type='v_mem', layers=3)
            self.trans_a_early = self.get_network(self_type='a_mem', layers=3)
            self.proj_l_e = nn.Linear(self.num_vectors_l, self.low_dim, bias=False)
            self.proj_v_e = nn.Linear(self.num_vectors_v, self.low_dim, bias=False)
            self.proj_a_e = nn.Linear(self.num_vectors_a, self.low_dim, bias=False)
            #self.proj_l_e = nn.Linear(combined_dim, self.low_dim, bias=False)
            #self.proj_v_e = nn.Linear(combined_dim, self.low_dim, bias=False)
            #self.proj_a_e = nn.Linear(combined_dim, self.low_dim, bias=False)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl', 'av2l', 'va2l']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va', 'lv2a', 'vl2a']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av', 'la2v', 'al2v']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
        # With GMU
            #embed_dim, attn_dropout = self.low_dim, self.attn_dropout
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            #embed_dim, attn_dropout = self.low_dim, self.attn_dropout
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            #embed_dim, attn_dropout = self.low_dim, self.attn_dropout
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        elif self_type == 'p_mem':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def transfm_2dim(self, x_t, dim, out_dim):
        if x_t.size(dim) != out_dim:
            if dim == 2:
                y  = torch.zeros((x_t.size(0), x_t.size(1), out_dim-x_t.size(2))).cuda()
            elif dim == 1:
                y  = torch.zeros((x_t.size(0), out_dim-x_t.size(1), x_t.size(2))).cuda()
            elif dim == 0:
                y  = torch.zeros((out_dim-x_t.size(0), x_t.size(1), x_t.size(2))).cuda()
            x_t = torch.cat((x_t, y), dim)
        
        return x_t
        
            
    def forward(self, txt, mask, segment, img, audio, output_gate=False):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        
        x_l = self.enc(txt, mask, segment)
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v = img.transpose(1, 2)
        #x_a = self.audio_enc(audio)
        x_a = audio.transpose(1, 2)
#        x_a = audio.transpose(1, 2) #self.audio_enc(audio)
        
        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        
        
        if proj_x_l.size(0) != self.num_vectors_l:
            proj_x_l = self.transfm_2dim(proj_x_l, 0, self.num_vectors_l)
        if proj_x_a.size(0) != self.num_vectors_a:
            proj_x_a = self.transfm_2dim(proj_x_a, 0, self.num_vectors_a)
        if proj_x_v.size(0) != self.num_vectors_v:
            proj_x_v = self.transfm_2dim(proj_x_v, 0, self.num_vectors_v)
        
        
        #Parallel fusion
        if self.hybrid:
            proj_x_l_e = self.proj_l_e(proj_x_l.permute(2, 1, 0)).permute(2, 1, 0)
            proj_x_a_e = self.proj_a_e(proj_x_a.permute(2, 1, 0)).permute(2, 1, 0)
            proj_x_v_e = self.proj_v_e(proj_x_v.permute(2, 1, 0)).permute(2, 1, 0)
            h_l = self.trans_l_early(proj_x_l_e)
            h_a = self.trans_a_early(proj_x_a_e)
            h_v = self.trans_v_early(proj_x_v_e)
            last_hl_early = h_l[0] + h_l[-1]#, zx = self.gmu_early([h_l, h_v, h_a])
            last_ha_early = h_a[0] + h_a[-1]
            last_hv_early = h_v[0] + h_v[-1]
            last_h_early, zx = self.gmu_early(last_hl_early, last_hv_early, last_ha_early)

        # First Crossmodal Transformers
        # (V,A) --> L
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a) # Dimension (V, N, d_v)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v) # Dimension (A, N, d_a)
        # (L,V) --> A
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
        # (L,A) --> V
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
        
        if self.lonly:
            # Biprojection
            h_l_with_v2a = self.trans_l_with_v2a(proj_x_l, h_a_with_vs, h_a_with_vs) # Dimension (L, N, d_l)
            h_l_with_a2v = self.trans_l_with_a2v(proj_x_l, h_v_with_as, h_v_with_as) # Dimension (L, N, d_l)
            
            # GMU Middle --------
            t_h_a_with_vs = h_a_with_vs #self.transfm_a2l(h_a_with_vs.permute(2,1,0)).permute(2,1,0)
            t_h_v_with_as = h_v_with_as #self.transfm_v2l(h_v_with_as.permute(2,1,0)).permute(2,1,0)
            h_l_gmu, z1_l = self.gmu_l_m([t_h_v_with_as, t_h_a_with_vs])
            
            # Residual conection level 1 to 2 -------
            h_l_with_v2a_tot = h_l_with_v2a + t_h_a_with_vs
            h_l_with_a2v_tot = h_l_with_a2v + t_h_v_with_as
            
            # Feature Fusion ---------
            h_ls_gmu, z2_l = self.gmu_l([h_l_with_a2v_tot, h_l_with_v2a_tot])
            
            # Residual conection level 1 to 3 -------
            h_ls_gmu += h_l_gmu

            last_h_l = last_hs = h_ls_gmu[0] + h_ls_gmu[-1]

        if self.aonly:
            # Biprojection ---------
            h_a_with_v2l = self.trans_a_with_v2l(proj_x_a, h_l_with_vs, h_l_with_vs)
            h_a_with_l2v = self.trans_a_with_l2v(proj_x_a, h_v_with_ls, h_v_with_ls)
            # GMU Middle --------
            
            t_h_l_with_vs = h_l_with_vs#self.transfm_l2a(h_l_with_vs.permute(2,1,0)).permute(2,1,0)
            t_h_v_with_ls = h_v_with_ls #self.transfm_v2a(h_v_with_ls.permute(2,1,0)).permute(2,1,0)
            h_a_gmu, z1_a = self.gmu_a_m([t_h_l_with_vs, t_h_v_with_ls])
            
            # Residual conection level 1 to 2 -------
            h_a_with_v2l_tot = h_a_with_v2l + t_h_l_with_vs
            h_a_with_l2v_tot = h_a_with_l2v + t_h_v_with_ls
            
            # Feature Fusion ---------
            h_as_gmu, z2_a = self.gmu_a([h_a_with_v2l_tot, h_a_with_l2v_tot])
            
            # Residual conection level 1 to 3 -------
            h_as_gmu += h_a_gmu
            
            last_h_a = last_hs = h_as_gmu[0] + h_as_gmu[-1]

        if self.vonly:
            # Biprojection ---------
            h_v_with_a2l = self.trans_v_with_a2l(proj_x_v, h_l_with_as, h_l_with_as)
            h_v_with_l2a = self.trans_v_with_l2a(proj_x_v, h_a_with_ls, h_a_with_ls)
            # GMU Middle --------
            
            t_h_l_with_as = h_l_with_as#self.transfm_l2v(h_l_with_as.permute(2,1,0)).permute(2,1,0)
            t_h_a_with_ls = h_a_with_ls #self.transfm_a2v(h_a_with_ls.permute(2,1,0)).permute(2,1,0)
            h_v_gmu, z1_v = self.gmu_v_m([t_h_l_with_as, t_h_a_with_ls])
            
            # Residual conection level 1 to 2 -------
            h_v_with_a2l_tot = h_v_with_a2l + t_h_l_with_as
            h_v_with_l2a_tot = h_v_with_l2a + t_h_a_with_ls
            
            # Feature Fusion ---------
            h_vs_gmu, z2_v = self.gmu_v([h_v_with_a2l_tot, h_v_with_l2a_tot])
            
            # Residual conection level 1 to 3 -------
            h_vs_gmu += h_v_gmu
            
            last_h_v = last_hs = h_vs_gmu[0] + h_vs_gmu[-1]
        
        if self.hybrid:
            last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a, last_h_early)
        else:
            last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        if output_gate:
            return self.out_layer(last_hs_proj), z
        else:
            return self.out_layer(last_hs_proj)
