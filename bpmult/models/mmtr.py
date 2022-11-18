import math
import torch
from torch import nn
import torch.nn.functional as F
from mmbt.models.image import ImageEncoder
from pytorch_pretrained_bert.modeling import BertModel
from transformers import BertModel as huggingBertModel
from mmbt.models.transformer import TransformerEncoder

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
    """ Gated Multimodal Layer based on 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """
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

class GatedMultimodalLayer2Features(nn.Module):
    """ Gated Multimodal Layer based on 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """
    def __init__(self, size_in1, size_in2, size_out):
        super(GatedMultimodalLayer2Features, self).__init__()
        self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out
        
        self.hidden1 = nn.Linear(size_in1, size_in1, bias=False)
        self.hidden2 = nn.Linear(size_in2, size_in1, bias=False)
        self.x_gate = nn.Linear(size_in1+size_in2, size_out, bias=False)

    def forward(self, x1, x2):
        h1 = torch.tanh(self.hidden1(x1))
        h2 = torch.tanh(self.hidden2(x2))
        x_cat = torch.cat((x1, x2), dim=-1)
        z = torch.sigmoid(self.x_gate(x_cat))
        y = torch.zeros(x1.size()).cuda()
        y1 = torch.cat((h1*x1, y), dim=-1)
        y2 = torch.cat((y, h2*x2), dim=-1)
        return z*y1 + z*y2, z #torch.cat((z, (1-z)), dim=-1)
        
class GatedMultimodalLayerFusion(nn.Module):
    """ Gated Multimodal Layer based on 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """
    def __init__(self, size_in1, size_in2, size_out):
        super(GatedMultimodalLayerFusion, self).__init__()
        self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out
        
        self.main_gate = nn.Linear(size_in1+size_in2, size_out, bias=False)

    def forward(self, x1, x2):
        x_cat = torch.cat((x1, x2), dim=-1)
        z = torch.sigmoid(self.main_gate(x_cat))
        out = z*x1 + (1-z)*x2
        
        return out, torch.cat((z, (1-z)), dim=-1)

class GatedMultimodal3LayerFeatures(nn.Module):
    """ Gated Multimodal Layer based on 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """
    def __init__(self, size_in1, size_in2, size_in3, size_out):
        super(GatedMultimodal3LayerFeatures, self).__init__()
        self.size_in1, self.size_in2, self.size_in3, self.size_out = size_in1, size_in2, size_in3, size_out
        
        self.hidden1 = nn.Linear(size_in1, size_out, bias=False)
        self.hidden2 = nn.Linear(size_in2, size_out, bias=False)
        self.hidden3 = nn.Linear(size_in3, size_out, bias=False)
        self.x1_gate = nn.Linear(size_in1+size_in2+size_in3, size_out, bias=False)
        self.x2_gate = nn.Linear(size_in1+size_in2+size_in3, size_out, bias=False)
        self.x3_gate = nn.Linear(size_in1+size_in2+size_in3, size_out, bias=False)

    def forward(self, x1, x2, x3):
        h1 = torch.tanh(self.hidden1(x1))
        h2 = torch.tanh(self.hidden2(x2))
        h3 = torch.tanh(self.hidden3(x3))
        x_cat = torch.cat((x1, x2, x3), dim=-1)
        z1 = torch.sigmoid(self.x1_gate(x_cat))
        z2 = torch.sigmoid(self.x2_gate(x_cat))
        z3 = torch.sigmoid(self.x3_gate(x_cat))

        return z1*h1*x1 + z2*h2*x2 + z3*h3*x3, torch.cat((z1, z2, z3), dim=-1)
        
class GatedMultimodal3LayerFusion(nn.Module):
    """ Gated Multimodal Layer based on 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """
    def __init__(self, size_in1, size_in2, size_in3, size_out):
        super(GatedMultimodal3LayerFusion, self).__init__()
        self.size_in1, self.size_in2, self.size_in3, self.size_out = size_in1, size_in2, size_in3, size_out
        
        self.main_gate = nn.Linear(size_in1 + size_out, size_out, bias=False)
        self.aux1_gate = nn.Linear(size_in2 + size_out, size_out, bias=False)

    def forward(self, x1, x2, x3):
        x1_cat = torch.cat((x2, x3), dim=-1)
        z1 = torch.sigmoid(self.aux1_gate(x1_cat))
        xf = z1*x2 + (1-z1)*x3
        
        x2_cat = torch.cat((x1, xf), dim=-1)
        z2 = torch.sigmoid(self.main_gate(x2_cat))
        out = z2*x1 + (1-z2)*xf

        return out, torch.cat((z2, (1-z2)*z1, (1-z2)*(1-z1)), dim=-1)

class GatedMultimodal4LayerFusion(nn.Module):
    """ Gated Multimodal Layer based on 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """
    def __init__(self, size_in1, size_in2, size_in3, size_in4, size_out):
        super(GatedMultimodal4LayerFusion, self).__init__()
        self.size_in1, self.size_in2, self.size_in3, self.size_in4, self.size_out = size_in1, size_in2, size_in3, size_in4, size_out
        
        self.main_gate = nn.Linear(size_in1 + size_out, size_out, bias=False)
        self.aux1_gate = nn.Linear(size_in2 + size_out, size_out, bias=False)
        self.aux2_gate = nn.Linear(size_in3 + size_in4, size_out, bias=False)

    def forward(self, x1, x2, x3, x4):
        x1_cat = torch.cat((x3, x4), dim=-1)
        z1 = torch.sigmoid(self.aux2_gate(x1_cat))
        xf = z1*x3 + (1-z1)*x4
        
        x2_cat = torch.cat((x2, xf), dim=-1)
        z2 = torch.sigmoid(self.aux1_gate(x2_cat))
        xf2 = z2*x2 + (1-z2)*xf
        
        x3_cat = torch.cat((x1, xf2), dim=-1)
        z3 = torch.sigmoid(self.main_gate(x3_cat))
        out = z3*x1 + (1-z3)*xf2

        return out, torch.cat((z3, (1-z3)*z2, (1-z3)*(1-z2)*z1, (1-z3)*(1-z2)*(1-z1)), dim=-1)
        
        
class TextShifting3LayerFeatures(nn.Module):
    """ Gated Multimodal Layer based on 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """
    def __init__(self, size_in1, size_in2, size_in3, size_out):
        super(TextShifting3LayerFeatures, self).__init__()
        self.size_in1, self.size_in2, self.size_in3, self.size_out = size_in1, size_in2, size_in3, size_out
        
        self.hidden1 = nn.Linear(size_in1, size_out, bias=False)
        self.hidden2 = nn.Linear(size_in2, size_out, bias=False)
        self.hidden3 = nn.Linear(size_in3, size_out, bias=False)
        self.x1_gate = nn.Linear(size_in1+size_in2+size_in3, size_out, bias=False)
        self.x2_gate = nn.Linear(size_in1+size_in2+size_in3, size_out, bias=False)
        self.x3_gate = nn.Linear(size_in1+size_in2+size_in3, size_out, bias=False)

    def forward(self, x1, x2, x3):
        '''
        y  = torch.zeros((512-x1.size(0), x1.size(1), x1.size(2))).cuda()
        x1 = torch.cat((x1, y), 0)
        y  = torch.zeros((512-x2.size(0), x2.size(1), x2.size(2))).cuda()
        x2 = torch.cat((x2, y), 0)
        y  = torch.zeros((512-x3.size(0), x3.size(1), x3.size(2))).cuda()
        x3 = torch.cat((x3, y), 0)
        '''
        h1 = torch.tanh(self.hidden1(x1))
        h2 = torch.tanh(self.hidden2(x2))
        h3 = torch.tanh(self.hidden3(x3))
        x_cat = torch.cat((x1, x2, x3), dim=-1)
        z1 = torch.sigmoid(self.x1_gate(x_cat))
        z2 = torch.sigmoid(self.x2_gate(x_cat))
        z3 = torch.sigmoid(self.x3_gate(x_cat))

        return z1*h1*x1 + z2*h2*x2 + z3*h3*x3, torch.cat((z1,z2,z3), dim=-1)

class TextShifting3LayerSimple(nn.Module):
    """ Layer inspired by 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """
    def __init__(self, size_in1, size_in2, size_in3, size_out):
        super(TextShifting3LayerSimple, self).__init__()
        self.size_in1, self.size_in2, self.size_in3, self.size_out = size_in1, size_in2, size_in3, size_out
        
        self.hidden1 = nn.Linear(size_in1, size_out, bias=False)
        self.hidden2 = nn.Linear(size_in2, size_out, bias=False)
        self.hidden3 = nn.Linear(size_in3, size_out, bias=False)
        self.x1_gate = nn.Linear(size_in1, size_out, bias=False)
        self.x2_gate = nn.Linear(size_in2, size_out, bias=False)
        self.x3_gate = nn.Linear(size_in3, size_out, bias=False)

    def forward(self, x1_, x2_, x3_):
        #print(x1_.size())
        x1 = x1_ #torch.sum(x1_, 1) #.squeeze(1)
        x2 = x2_ #torch.sum(x2_, 1) #.squeeze(1)
        x3 = x3_ #torch.sum(x3_, 1) #.squeeze(1)
        y  = torch.zeros((512-x1.size()[0], x1.size()[1], x1.size()[-1])).cuda()
        x1 = torch.cat((x1, y), 0)
        y  = torch.zeros((200-x2.size()[0], x2.size()[1], x2.size()[-1])).cuda()
        x2 = torch.cat((x2, y), 0)
        y  = torch.zeros((200-x3.size()[0], x3.size()[1], x3.size()[-1])).cuda()
        x3 = torch.cat((x3, y), 0)
        #print(x1.size(), x2.size(), x3.size())
        h1 = torch.tanh(self.hidden1(x1))
        h2 = torch.tanh(self.hidden2(x2))
        h3 = torch.tanh(self.hidden3(x3))
        
        x_cat = torch.cat((x1, x2, x3), 0)
        #print(x_cat.size())
        z1 = torch.sigmoid(self.x1_gate(x_cat))
        z2 = torch.sigmoid(self.x2_gate(x_cat))
        z3 = torch.sigmoid(self.x3_gate(x_cat))
        #print(h1.size(), h2.size(), h3.size())
        #print(z1.size(), z2.size(), z3.size())
        
        h_cat = torch.cat((h1, h2, h3))
        prod = z1*h_cat + z2*h_cat + z3*h_cat
        return torch.sum(prod, 0).unsqueeze(0), torch.cat((z1, z2, z3), dim=0)

class TextShifting3Layer(nn.Module):
    """ Layer inspired by 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """
    def __init__(self, size_in1, size_in2, size_in3, size_out):
        super(TextShifting3Layer, self).__init__()
        self.size_in1, self.size_in2, self.size_in3, self.size_out = size_in1, size_in2, size_in3, size_out
        
        self.hidden1 = nn.Linear(size_in1, size_out, bias=False)
        self.hidden2 = nn.Linear(size_in2, size_out, bias=False)
        self.hidden3 = nn.Linear(size_in3, size_out, bias=False)
        self.x1_gate = nn.Linear(size_in1+size_in2+size_in3, size_out, bias=False)
        self.x2_gate = nn.Linear(size_in1+size_in2+size_in3, size_out, bias=False)
        self.x3_gate = nn.Linear(size_in1+size_in2+size_in3, size_out, bias=False)

    def forward(self, x1, x2, x3):
        h1 = F.tanh(self.hidden1(x1))
        h2 = F.tanh(self.hidden2(x2))
        h3 = F.tanh(self.hidden3(x3))
        x_cat = torch.cat((x1, x2, x3), dim=1)
        z1 = F.sigmoid(self.x1_gate(x_cat))
        z2 = F.sigmoid(self.x2_gate(x_cat))
        z3 = F.sigmoid(self.x3_gate(x_cat))

        return z1*h1 + z2*h2 + z3*h3, torch.cat((z1, z2, z3), dim=1)
        
class TextShifting3LayerBatch(nn.Module):
    """ Layer inspired by 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """
    def __init__(self, size_in1, size_in2, size_in3, size_out):
        super(TextShifting3LayerBatch, self).__init__()
        self.size_in1, self.size_in2, self.size_in3, self.size_out = size_in1, size_in2, size_in3, size_out
        
        self.hidden1 = nn.Linear(size_in1, size_out, bias=False)
        self.hidden2 = nn.Linear(size_in2, size_out, bias=False)
        self.hidden3 = nn.Linear(size_in3, size_out, bias=False)
        self.x1_gate = nn.Linear(size_in1+size_in2+size_in3, size_out, bias=False)
        self.x2_gate = nn.Linear(size_in1+size_in2+size_in3, size_out, bias=False)
        self.x3_gate = nn.Linear(size_in1+size_in2+size_in3, size_out, bias=False)

    def forward(self, x1, x2, x3):
        h1 = F.tanh(self.hidden1(x1))
        h2 = F.tanh(self.hidden2(x2))
        h3 = F.tanh(self.hidden3(x3))
        x_cat = torch.cat((x1, x2, x3), dim=2)
        z1 = F.sigmoid(self.x1_gate(x_cat))
        z2 = F.sigmoid(self.x2_gate(x_cat))
        z3 = F.sigmoid(self.x3_gate(x_cat))

        out = z1*h1 + z2*h2 + z3*h3
        #print(out.size())
        return out#, torch.cat((z1, z2, z3), dim=2)
        
class TextShifting4LayerSimple(nn.Module):
    """ Layer inspired by 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """
    def __init__(self, size_in1, size_in2, size_in3, size_in4, size_out):
        super(TextShifting4LayerSimple, self).__init__()
        self.size_in1, self.size_in2, self.size_in3, self.size_in4, self.size_out = size_in1, size_in2, size_in3, size_in4, size_out
        
        self.hidden1 = nn.Linear(size_in1, size_out, bias=False)
        self.hidden2 = nn.Linear(size_in2, size_out, bias=False)
        self.hidden3 = nn.Linear(size_in3, size_out, bias=False)
        self.hidden4 = nn.Linear(size_in4, size_out, bias=False)
        self.x1_gate = nn.Linear(size_in1, size_out, bias=False)
        self.x2_gate = nn.Linear(size_in2, size_out, bias=False)
        self.x3_gate = nn.Linear(size_in3, size_out, bias=False)
        self.x4_gate = nn.Linear(size_in4, size_out, bias=False)

    def forward(self, x1, x2, x3, x4):
#        print(x4.size())
        #x1 = x1_ #torch.sum(x1_, 1) #.squeeze(1)
        #x2 = x2_ #torch.sum(x2_, 1) #.squeeze(1)
        #x3 = x3_ #torch.sum(x3_, 1) #.squeeze(1)
        y  = torch.zeros((512-x1.size()[0], x1.size()[1], x1.size()[-1])).cuda()
        x1 = torch.cat((x1, y), 0)
        y  = torch.zeros((200-x2.size()[0], x2.size()[1], x2.size()[-1])).cuda()
        x2 = torch.cat((x2, y), 0)
        y  = torch.zeros((200-x3.size()[0], x3.size()[1], x3.size()[-1])).cuda()
        x3 = torch.cat((x3, y), 0)
   #     y  = torch.zeros((2048-x4.size()[0], x4.size()[1], x4.size()[-1])).cuda()
        x4 = x4.unsqueeze(0) #torch.cat((x4, y), 0)
        #print(x1.size(), x2.size(), x3.size())
        h1 = torch.tanh(self.hidden1(x1))
        h2 = torch.tanh(self.hidden2(x2))
        h3 = torch.tanh(self.hidden3(x3))
        h4 = torch.tanh(self.hidden4(x4))
        
        x_cat = torch.cat((x1, x2, x3, x4), 0)
        #print(x_cat.size())
        z1 = torch.sigmoid(self.x1_gate(x_cat))
        z2 = torch.sigmoid(self.x2_gate(x_cat))
        z3 = torch.sigmoid(self.x3_gate(x_cat))
        z4 = torch.sigmoid(self.x4_gate(x_cat))
        #print(h1.size(), h2.size(), h3.size())
        #print(z1.size(), z2.size(), z3.size())
        
        h_cat = torch.cat((h1, h2, h3, h4))
        prod = z1*h_cat + z2*h_cat + z3*h_cat + z4*h_cat
        return torch.sum(prod, 0).unsqueeze(0), torch.cat((z1, z2, z3, z4), dim=0)

class TextShifting4LayerFeatures(nn.Module):
    """ Layer inspired by 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """
    def __init__(self, size_in1, size_in2, size_in3, size_in4, size_out):
        super(TextShifting4LayerFeatures, self).__init__()
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

        return z1*h1*xs[0] + z2*h2*xs[1] + z3*h3*xs[2] + z4*h4*xs[3], torch.cat((z1, z2, z3, z4), dim=-1)
        
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
        
class TextShifting4LayerFusion(nn.Module):
    """ Layer inspired by 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """
    def __init__(self, size_in1, size_in2, size_in3, size_in4, size_out):
        super(TextShifting4LayerFusion, self).__init__()
        self.size_in1, self.size_in2, self.size_in3, self.size_in4, self.size_out = size_in1, size_in2, size_in3, size_in4, size_out

        self.x1_gate = nn.Linear(size_in1+size_in2+size_in3+size_in4, size_out, bias=False)
        self.x2_gate = nn.Linear(size_in1+size_in2+size_in3+size_in4, size_out, bias=False)
        self.x3_gate = nn.Linear(size_in1+size_in2+size_in3+size_in4, size_out, bias=False)
        self.x4_gate = nn.Linear(size_in1+size_in2+size_in3+size_in4, size_out, bias=False)

    def forward(self, xs):
        x_cat = torch.cat(xs, dim=-1)
        z1 = torch.sigmoid(self.x1_gate(x_cat))
        z2 = torch.sigmoid(self.x2_gate(x_cat))
        z3 = torch.sigmoid(self.x3_gate(x_cat))
        z4 = torch.sigmoid(self.x4_gate(x_cat))

        return z1*xs[0] + z2*xs[1] + z3*xs[2] + z4*xs[3], torch.cat((z1, z2, z3, z4), dim=-1)


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


class TextShifting6Layer(nn.Module):
    """ Layer inspired by 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """
    def __init__(self, size_in1, size_in2, size_in3, size_in4, size_in5, size_in6, size_out):
        super(TextShifting6Layer, self).__init__()
        self.size_in1, self.size_in2, self.size_in3, self.size_in4, self.size_in5, self.size_in6, self.size_out = size_in1, size_in2, size_in3, size_in4, size_in5, size_in6, size_out
        
        self.hidden1 = nn.Linear(size_in1, size_out, bias=False)
        self.hidden2 = nn.Linear(size_in2, size_out, bias=False)
        self.hidden3 = nn.Linear(size_in3, size_out, bias=False)
        self.hidden4 = nn.Linear(size_in4, size_out, bias=False)
        self.hidden5 = nn.Linear(size_in5, size_out, bias=False)
        self.hidden6 = nn.Linear(size_in6, size_out, bias=False)
        combined_size = size_in1+size_in2+size_in3+size_in4+size_in5+size_in6
        self.x1_gate = nn.Linear(combined_size, size_out, bias=False)
        self.x2_gate = nn.Linear(combined_size, size_out, bias=False)
        self.x3_gate = nn.Linear(combined_size, size_out, bias=False)
        self.x4_gate = nn.Linear(combined_size, size_out, bias=False)
        self.x5_gate = nn.Linear(combined_size, size_out, bias=False)
        self.x6_gate = nn.Linear(combined_size, size_out, bias=False)

    def forward(self, x1, x2, x3, x4, x5, x6):
        h1 = F.tanh(self.hidden1(x1))
        h2 = F.tanh(self.hidden2(x2))
        h3 = F.tanh(self.hidden3(x3))
        h4 = F.tanh(self.hidden4(x4))
        h5 = F.tanh(self.hidden5(x5))
        h6 = F.tanh(self.hidden5(x6))
        x_cat = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        z1 = F.sigmoid(self.x1_gate(x_cat))
        z2 = F.sigmoid(self.x2_gate(x_cat))
        z3 = F.sigmoid(self.x3_gate(x_cat))
        z4 = F.sigmoid(self.x4_gate(x_cat))
        z5 = F.sigmoid(self.x5_gate(x_cat))
        z6 = F.sigmoid(self.x6_gate(x_cat))

        return z1*h1 + z2*h2 + z3*h3 + z4*h4 + z5*h5 + z6*h6, torch.cat((z1, z2, z3, z4, z5, z6), dim=1)


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
        
class GatedMultimodalUnitSoftmaxFusion(nn.Module):
    """ Gated Multimodal Layer based on 'Gated multimodal networks, Arevalo1 et al. (https://arxiv.org/abs/1702.01992)
    """
    def __init__(self, M:int, size_in:int, size_out:int, probs=True, residual=False):
        super(GatedMultimodalUnitSoftmaxFusion, self).__init__()
        assert(type(M)  == int)
        assert(type(size_in)  == int)
        assert(type(size_out) == int)
        
        self.M, self.size_in, self.size_out, self.probs, self.residual = M, size_in, size_out, probs, residual
        self.hidden = []
        self.gate = []
        for i in range(M):
            if probs:
                self.hidden += [nn.Linear(size_in, size_out, bias=False)]
            if i > 0:
                self.gate += [nn.Linear(M * size_in, size_out, bias=False)]
        if probs:
            self.hidden = nn.ModuleList(self.hidden)
        self.gate = nn.ModuleList(self.gate)

    def forward(self, xs:list):
        assert(type(xs) == list)
        # Number of modalities
        M = len(xs)
        assert(M == self.M)
        
        # Asserting Seq size is the same for all
        seq_size = xs[0].size(0)
        for i in range(1, M):
            assert(xs[i].size(0) == seq_size)
                
        # Hidden layers
        hs = []
        for i in range(M):
            if self.probs:
                hs += [torch.tanh(self.hidden[i](xs[i]))]
            else:
                hs += [xs[i]]
                
        # Concatenate by Seq dimension
        x_cat = torch.cat(xs, dim=-1)
        
        # Activation
        #G = self.gate(x_cat)
        zs = []
        for i in range(M-1):
            #zs += [torch.sigmoid(G[..., (self.size_out*i):(self.size_out*(i+1))])]
            zs += [torch.sigmoid(self.gate[i](x_cat))]
        
        if self.residual:
            ans = [zs[0]*hs[0]*xs[0]]
            for i in range(1, M):
                ans += [1-zs[0]]
            for i in range(1, M-1):
                ans[i] *= zs[i]*hs[i]*xs[i]
                for j in range(i+1, M):
                    ans[j] *= 1-zs[i]
            ans[-1] *= hs[-1]*xs[-1]
        else:
            ans = [zs[0]*hs[0]]
            for i in range(1, M):
                ans += [1-zs[0]]
            for i in range(1, M-1):
                ans[i] *= zs[i]*hs[i]
                for j in range(i+1, M):
                    ans[j] *= 1-zs[i]
            ans[-1] *= hs[-1]
            ANS = ans[0]
            for i in range(1, M):
                ANS += ans[i]
       
        # Activations
        act = [zs[0]]
        for i in range(1, M):
            act += [1-zs[0]]
        for i in range(1, M-1):
            act[i] *= zs[i]
            for j in range(i+1, M):
                act[j] *= 1-zs[i]
        act = torch.cat(act, dim=-1)
            
        return ANS, act

class GatedMultimodalUnitSoftmaxFusionDiego(nn.Module):
    """ Gated Multimodal Layer based on 'Gated multimodal networks, Arevalo1 et al. (https://arxiv.org/abs/1702.01992)
    """
    def __init__(self, M:int, size_in:int, size_out:int, probs=True, residual=True):
        super(GatedMultimodalUnitSoftmaxFusionDiego, self).__init__()
        assert(type(M)  == int)
        assert(type(size_in)  == int)
        assert(type(size_out) == int)
        
        self.M, self.size_in, self.size_out, self.probs, self.residual = M, size_in, size_out, probs, residual
        self.hidden = []
        self.gate = []
        for i in range(M):
            if probs:
                self.hidden += [nn.Linear(size_in, size_out, bias=False)]
            self.gate   += [nn.Linear(M * size_in, size_out, bias=False)]
        if probs:
            self.hidden = nn.ModuleList(self.hidden)
        self.gate   = nn.ModuleList(self.gate)

    def forward(self, xs:list):
        assert(type(xs) == list)
        # Number of modalities
        M = len(xs)
        assert(M == self.M)
        
        # Asserting Seq size is the same for all
        seq_size = xs[0].size(0)
        for i in range(1, M):
            assert(xs[i].size(0) == seq_size)
                
        # Hidden layers
        hs = []
        for i in range(M):
            if self.probs:
                hs += [torch.tanh(self.hidden[i](xs[i]))]
            else:
                hs += [xs[i]]
                
        # Concatenate by Seq dimension
        x_cat = torch.cat(xs, dim=-1)
        
        # Activation
        zs = []
        for i in range(M):
            zs += [self.gate[i](x_cat).unsqueeze(-1)]
        z = torch.softmax(torch.cat(zs, dim=-1), dim=-1)
        
        if self.residual:
            ans = z[..., 0]*hs[0]*xs[0]
            for i in range(1, M):
                ans += z[..., i]*hs[i]*xs[i]
        else:
            ans = z[..., 0]*hs[0]
            for i in range(1, M):
                ans += z[..., i]*hs[i]
       
        # Activations
        act = z[..., 0]
        for i in range(1, M):
            act = torch.cat((act, z[..., i]), dim=-1)
            
        return ans, act
        
class GatedMultimodalUnitSoftmaxAbdiel(nn.Module):
    """ Gated Multimodal Layer based on 'Gated multimodal networks, Arevalo1 et al. (https://arxiv.org/abs/1702.01992)
    """
    def __init__(self, M:int, size_in:int, size_out:int, probs=True, residual=True):
        super(GatedMultimodalUnitSoftmaxAbdiel, self).__init__()
        assert(type(M)  == int)
        assert(type(size_in)  == int)
        assert(type(size_out) == int)
        
        self.M, self.size_in, self.size_out, self.probs, self.residual = M, size_in, size_out, probs, residual
        self.hidden = []
        for i in range(M):
            self.hidden += [nn.Linear(size_in, size_out, bias=False)]
        self.hidden = nn.ModuleList(self.hidden)
        self.gate = nn.Linear(size_in, size_out, bias=False)

    def forward(self, xs:list):
        assert(type(xs) == list)
        # Number of modalities
        M = len(xs)
        assert(M == self.M)
        
        # Asserting tensor size is [Seq, Batch, Feat] instead of [Batch, Feat]
        for i in range(M):
            if xs[i].dim() == 1:
                xs[i] = xs[i].unsqueeze(0)
            if xs[i].dim() == 2:
                xs[i] = xs[i].unsqueeze(0)
        
        # Asserting Seq size is the same for all
        seq_size = xs[0].size(0)
        for i in range(1, M):
            assert(xs[i].size(0) == seq_size)
                
        # Hidden layers
        hs = []
        for i in range(M):
            if self.probs:
                hs += [torch.tanh(self.hidden[i](xs[i]))]
            else:
                hs += [self.hidden[i](xs[i])]
                
        # Concatenate by Seq dimension
        x_cat = torch.cat(xs, dim=0)
        h_cat = torch.cat(hs, dim=0)
        z = torch.softmax(self.gate(x_cat), dim=0)
        
        if self.residual:
            ans = z[0:seq_size]*h_cat[0:seq_size]*xs[0]
            for i in range(1, M):
                ans += z[(i*seq_size):((i+1)*seq_size)]*h_cat[(i*seq_size):((i+1)*seq_size)]*xs[i]
        else:
            ans = z[0:seq_size]*h_cat[0:seq_size]
            for i in range(1, M):
                ans += z[(i*seq_size):((i+1)*seq_size)]*h_cat[(i*seq_size):((i+1)*seq_size)]
       
        # Activations
        act = z[0:seq_size]
        for i in range(1, M):
            act = torch.cat((act, z[(i*seq_size):((i+1)*seq_size)]), dim=-1)
            
        return ans, act

# este es el MMTransformerGMUHybridClf
class MMTransformerGMUHybridClf(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model for Text, Video frames and Audio spectrogram with GMU late fusion.
        """
        super(MMTransformerGMUHybridClf, self).__init__()
        self.args = args
        self.orig_d_l, self.orig_d_v, self.orig_d_a, args.orig_d_m = args.orig_d_l, args.orig_d_v, args.orig_d_a, 312
        self.d_l, self.d_a, self.d_v, self.d_m = 768, 768, 768, 768
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
        
        self.enc = BertEncoder(args)
        self.audio_enc = AudioEncoder(args)

        # 0. Project poster feature to 768 dim
        self.proj_poster  = nn.Linear(self.orig_d_v, self.d_v, bias=False)
        #self.proj2_poster = nn.Linear(self.orig_d_v, self.d_v, bias=False)
        # 0. Temporal Linear layers for Early fusion
        self.proj_l_e = nn.Linear(512, 64, bias=False)
        self.proj_v_e = nn.Linear(200, 64, bias=False)
        self.proj_a_e = nn.Linear(200, 64, bias=False)
        self.proj2_l_e = nn.Linear(self.orig_d_l, self.d_l, bias=False)
        self.proj2_v_e = nn.Linear(self.orig_d_v, self.d_v, bias=False)
        self.proj2_a_e = nn.Linear(self.orig_d_a, self.d_a, bias=False)
        

        combined_dim = 768 # For GMU
        
        output_dim = args.n_classes        # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        self.proj12 = nn.Linear(combined_dim, combined_dim)
        self.proj22 = nn.Linear(combined_dim, combined_dim)
        self.out_layer2 = nn.Linear(combined_dim, output_dim)
        self.out_layer_final = nn.Linear(output_dim, output_dim)
        
        # GMU layer for fusing text and image and audio information
        #self.gmu = TextShifting3Layer(self.d_l*2, self.d_v*2, self.d_a*2, self.d_l)
        #self.gmu = TextShifting4Layer(self.d_l*2, self.d_v*2, self.d_v*2, self.d_v, self.d_l)
        self.gmu = TextShifting5Layer(self.d_l*2, self.d_v*2, self.d_v*2, self.d_v, self.d_l, self.d_l)
        #self.gmuSimple = TextShifting4LayerSimple(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l)
        self.gmuFeat = TextShifting3LayerFeatures(self.d_l, self.d_v, self.d_a, self.d_l)
        self.trans_h_mem = self.get_network(self_type='h_mem', layers=3)
        #self.gmuSimple = TextShifting3LayerSimple(self.d_l, self.d_v, self.d_a, self.d_l)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.d_v, self.attn_dropout
        elif self_type == 'h_mem':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
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
            
    def forward(self, txt, mask, segment, img, audio, poster, output_gate=False):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        #txt = torch.cat((txt, txt), 0)
        #mask = torch.cat((mask, mask), 0)
        #segment = torch.cat((segment, segment), 0)
        #img = torch.cat((img, img), 0)
        #audio = torch.cat((audio, audio), 0)
        #poster = torch.cat((poster, poster), 0)
        x_l = self.enc(txt, mask, segment)
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v = img.transpose(1, 2)
        x_a = self.audio_enc(audio)
        
        if x_l.size(2) != 512:
            y  = torch.zeros((x_l.size(0), x_l.size(1), 512-x_l.size(2))).cuda()
            x1 = torch.cat((x_l, y), 2)
        else:
            x1 = x_l
        if x_v.size(2) != 200:
            y  = torch.zeros((x_v.size(0), x_v.size(1), 200-x_v.size(2))).cuda()
            x2 = torch.cat((x_v, y), 2)
        else:
            x2 = x_v
        if x_a.size(2) != 200:
            y  = torch.zeros((x_a.size(0), x_a.size(1), 200-x_a.size(2))).cuda()
            x3 = torch.cat((x_a, y), 2)
        else:
            x3 = x_a
        proj_x_l_e = self.proj_l_e(x1)
        proj_x_v_e = self.proj_v_e(x2)
        proj_x_a_e = self.proj_a_e(x3)
        proj_x_l_e = self.proj2_l_e(proj_x_l_e.permute(2, 0, 1))
        proj_x_a_e = self.proj2_a_e(proj_x_a_e.permute(2, 0, 1))
        proj_x_v_e = self.proj2_v_e(proj_x_v_e.permute(2, 0, 1))
        #proj_x_p_e = self.proj2_poster(poster)
        
        #Early fusion
        #h_early, zx = self.gmuSimple(proj_x_l_e, proj_x_a_e, proj_x_v_e, proj_x_p_e)
        h_early = self.gmuFeat(proj_x_l_e, proj_x_a_e, proj_x_v_e)
        h_early = self.trans_h_mem(h_early)
        if type(h_early) == tuple:
            h_early = h_early[0]
        last_h_early = last_hs = h_early[-1]

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)

        if self.lonly:
            # (V,A) --> L
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)    # Dimension (L, N, d_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)    # Dimension (L, N, d_l)
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            h_ls = self.trans_l_mem(h_ls)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]   # Take the last output for prediction

        if self.aonly:
            # (L,V) --> A
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
            h_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

        if self.vonly:
            # (L,A) --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            h_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]
        
        #last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a)
        #last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a, self.proj_poster(poster), h_early.squeeze(0))
        last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a, self.proj_poster(poster), last_h_early)
        #print(last_hs.size())
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)
        return output
        # A residual block
        '''
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        last_hs_proj2 = self.proj22(F.dropout(F.relu(self.proj12(h_early)), p=self.out_dropout, training=self.training))
        last_hs_proj2 += h_early
        
        output = self.out_layer(last_hs_proj)
        output2 = self.out_layer2(last_hs_proj2)
        if output.size() != output2.size():
            output2 = output2.squeeze(0)
        output = self.out_layer_final(torch.cat((output, output2)))
        if output_gate:
            size_aux = output.size()[0]
            return output[:int(size_aux/2),:] + output[int(size_aux/2):,:], z #torch.sum(output, 0).unsqueeze(0), z
        else:
            size_aux = output.size()[0]
            return output[:int(size_aux/2),:] + output[int(size_aux/2):,:] #torch.sum(output, 0).unsqueeze(0)
        '''
     
class MMTransformerGMUClfVAPT(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model for Text, Video frames and Audio spectrogram with GMU late fusion.
        """
        super(MMTransformerGMUClfVAPT, self).__init__()
        self.args = args
        self.orig_d_l, self.orig_d_v, self.orig_d_a, args.orig_d_m = args.orig_d_l, args.orig_d_v, args.orig_d_a, 312
        self.d_l, self.d_a, self.d_v, self.d_m = 768, 768, 768, 768
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
        
        self.enc = BertEncoder(args)
        self.audio_enc = AudioEncoder(args)

        # 0. Project poster feature to 768 dim
        self.proj_poster  = nn.Linear(self.orig_d_v, self.d_v, bias=False)
        #self.proj2_poster = nn.Linear(self.orig_d_v, self.d_v, bias=False)
        # 0. Temporal Linear layers for Early fusion
        #------For the crossmodal layers sum
        #self.proj_v2a = nn.Linear(self.d_l, self.d_l)#, bias=False)
        #self.proj_a2v = nn.Linear(self.d_l, self.d_l)#, bias=False)
        #self.proj_v2l = nn.Linear(self.d_a, self.d_a)#, bias=False)
        #self.proj_l2v = nn.Linear(self.d_a, self.d_a)#, bias=False)
        #self.proj_l2a = nn.Linear(self.d_v, self.d_v)#, bias=False)
        #self.proj_a2l = nn.Linear(self.d_v, self.d_v)#, bias=False)
        #--- GMU instead of sum
        #------ GMU Middle
        #self.gmu_l_m = GatedMultimodalLayerFeatures(self.d_v, self.d_a, self.d_l)
        #self.gmu_v_m = GatedMultimodalLayerFeatures(self.d_l, self.d_a, self.d_v)
        #self.gmu_a_m = GatedMultimodalLayerFeatures(self.d_l, self.d_v, self.d_a)
        #------ GMU Top
        self.gmu_l = GatedMultimodalLayerFeatures(self.d_l, self.d_l, self.d_l)
        self.gmu_v = GatedMultimodalLayerFeatures(self.d_v, self.d_v, self.d_v)
        self.gmu_a = GatedMultimodalLayerFeatures(self.d_a, self.d_a, self.d_a)
        #--------- GMU Top with sum
        #self.gmu_l = GatedMultimodal3LayerFeatures(self.d_l, self.d_l, self.d_l, self.d_l)
        #self.gmu_v = GatedMultimodal3LayerFeatures(self.d_v, self.d_v, self.d_v, self.d_l)
        #self.gmu_a = GatedMultimodal3LayerFeatures(self.d_a, self.d_a, self.d_a, self.d_l)
        #-------------
        #self.proj_va2l = nn.Linear(self.d_l * 2, self.d_l, bias=False)
        #self.proj_vl2a = nn.Linear(self.d_a * 2, self.d_a, bias=False)
        #self.proj_la2v = nn.Linear(self.d_v * 2, self.d_v, bias=False)

        combined_dim = 768 # For GMU
        
        output_dim = args.n_classes        # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
       
        # Projection layers
        self.combined_dim = 768
        self.proj1 = nn.Linear(self.combined_dim, self.combined_dim)
        self.proj2 = nn.Linear(self.combined_dim, self.combined_dim)
        self.out_layer = nn.Linear(self.combined_dim, output_dim)
        
        # GMU layer for fusing text and image information
        # For Normal Concatenation
        #self.gmu = TextShifting4Layer(2*self.d_l, 2*self.d_v, 2*self.d_v, self.d_v, self.d_l)
        # For GMU Features Attention Fusion
        self.gmu = TextShifting4Layer(self.d_l, self.d_v, self.d_v, self.d_v, self.d_l)
        
    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            # For Normal Concatenation
            #embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
            # For GMU Features Attention Fusion
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            #embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            #embed_dim, attn_dropout = 2*self.d_v, self.attn_dropout
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
            
    def forward(self, txt, mask, segment, img, audio, poster, output_gate=False):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_l = self.enc(txt, mask, segment)
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v = img.transpose(1, 2)
        x_a = self.audio_enc(audio)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)

        if self.lonly:
            # (V,A) --> L
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)    # Dimension (L, N, d_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)    # Dimension (L, N, d_l)
            # Normal concatenation
            #h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            # GMU Features Attention Fusion
            h_ls, z_l = self.gmu_l(h_l_with_vs, h_l_with_as)
            h_ls = self.trans_l_mem(h_ls)
            # GMU Features Attention Fusion for residual conection
            #h_ls_gmu = self.gmu_l(h_l_with_vs, h_l_with_as)
            #h_ls = self.trans_l_mem(h_ls_gmu)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1] #+ h_ls_gmu[-1]  # Take the last output for prediction

        if self.aonly:
            # (L,V) --> A
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            # Normal concatenation
            #h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
            # GMU Features Attention Fusion
            h_as, z_a = self.gmu_a(h_a_with_ls, h_a_with_vs)
            h_as = self.trans_a_mem(h_as)
            # GMU Features Attention Fusion for residual conection
            #h_as_gmu = self.gmu_a(h_a_with_ls, h_a_with_vs)
            #h_as = self.trans_a_mem(h_as_gmu)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1] #+ h_as_gmu[-1]

        if self.vonly:
            # (L,A) --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            # Normal concatenation
            #h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            # GMU Features Attention Fusion
            h_vs, z_v = self.gmu_v(h_v_with_ls, h_v_with_as)
            h_vs = self.trans_v_mem(h_vs)
            # GMU Features Attention Fusion for residual conection
            #h_vs_gmu = self.gmu_v(h_v_with_ls, h_v_with_as)
            #h_vs = self.trans_v_mem(h_vs_gmu)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1] #+ h_vs_gmu[-1]
        pos = self.proj_poster(poster)
        #print(last_h_l.size(), last_h_v.size(), last_h_a.size(), pos.size())
        last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a, pos)
        #print("last_hs:", last_hs.size(), "Z:", z.size())
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
                
        if output_gate:
            return self.out_layer(last_hs_proj), torch.cat((z_l[-1], z_v[-1], z_a[-1], z), dim=1)
        else:
            return self.out_layer(last_hs_proj)
   
# For CMU_MOSEI
class TranslatingMMTransformerGMUClf(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model for Text, Video frames and Audio spectrogram with GMU late fusion.
        """
        super(TranslatingMMTransformerGMUClf, self).__init__()
        self.args = args
        self.orig_d_l, self.orig_d_v, self.orig_d_a, args.orig_d_m = args.orig_d_l, args.orig_d_v, args.orig_d_a, 20 #args.orig_d_a, 312
        self.d_l, self.d_a, self.d_v, self.d_m = args.hidden_sz, args.hidden_sz, args.hidden_sz, 0#768, 768, 768, 768#300,300,300,300#768, 768, 768, 768
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
        
        self.enc = BertEncoder(args)
        #self.audio_enc = AudioEncoder_cmumosei(args)
 #       self.audio_enc = AudioEncoder_cmumosi(args)

        # 0. Project poster feature to 768 dim
        #self.proj_poster  = nn.Linear(self.orig_d_v, self.d_v, bias=False)
        #self.proj2_poster = nn.Linear(self.orig_d_v, self.d_v, bias=False)
        # 0. Temporal Linear layers for Early fusion
        #------For the crossmodal layers sum
        #self.proj_v2a = nn.Linear(self.d_l, self.d_l)#, bias=False)
        #self.proj_a2v = nn.Linear(self.d_l, self.d_l)#, bias=False)
        #self.proj_v2l = nn.Linear(self.d_a, self.d_a)#, bias=False)
        #self.proj_l2v = nn.Linear(self.d_a, self.d_a)#, bias=False)
        #self.proj_l2a = nn.Linear(self.d_v, self.d_v)#, bias=False)
        #self.proj_a2l = nn.Linear(self.d_v, self.d_v)#, bias=False)
        #--- GMU instead of sum
        #------ GMU Middle
        self.gmu_l_m = GatedMultimodalLayerFeatures(self.d_v, self.d_a, self.d_l)
        self.gmu_v_m = GatedMultimodalLayerFeatures(self.d_l, self.d_a, self.d_v)
        self.gmu_a_m = GatedMultimodalLayerFeatures(self.d_l, self.d_v, self.d_a)
        #------ GMU Top
        self.gmu_l = GatedMultimodalLayerFeatures(self.d_v, self.d_a, self.d_l)
        self.gmu_v = GatedMultimodalLayerFeatures(self.d_l, self.d_a, self.d_v)
        self.gmu_a = GatedMultimodalLayerFeatures(self.d_l, self.d_v, self.d_a)
        #--------- GMU Top with sum
        self.gmu_early = GatedMultimodal3LayerFeatures(self.d_l, self.d_v, self.d_a, self.d_l)
        #self.gmu_v = GatedMultimodal3LayerFeatures(self.d_v, self.d_v, self.d_v, self.d_l)
        #self.gmu_a = GatedMultimodal3LayerFeatures(self.d_a, self.d_a, self.d_a, self.d_l)
        #-------------
        #self.proj_va2l = nn.Linear(self.d_l * 2, self.d_l, bias=False)
        #self.proj_vl2a = nn.Linear(self.d_a * 2, self.d_a, bias=False)
        #self.proj_la2v = nn.Linear(self.d_v * 2, self.d_v, bias=False)

        combined_dim = args.hidden_sz #768 # For GMU
        
        output_dim = args.n_classes        # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        # For Parallel Fusion
       # self.proj2_l_e = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        #self.proj2_v_e = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        #self.proj2_a_e = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)

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
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        #self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        #self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
        #self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        
        # For CMU-MOSI
       # self.out_layer = nn.Linear(combined_dim, 1)
        
        #self.sigmoid = nn.Sigmoid()
        #self.proj12 = nn.Linear(combined_dim, combined_dim)
        #self.proj22 = nn.Linear(combined_dim, combined_dim)
        #self.out_layer2 = nn.Linear(combined_dim, output_dim)
        #self.out_layer_final = nn.Linear(output_dim, output_dim)
        
        # GMU layer for fusing text and image and audio information
        #self.gmu = TextShifting3Layer(self.d_l, self.d_v, self.d_a, self.d_l)
        self.gmu = TextShifting4Layer(self.d_l, self.d_v, self.d_a, self.d_l, self.d_l)
        #self.gmu = TextShifting5Layer(self.d_l*2, self.d_v*2, self.d_v*2, self.d_v, self.d_l, self.d_l)
        #self.gmuSimple = TextShifting4LayerSimple(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l)
        #self.gmuSimple = TextShifting3LayerSimple(self.d_l, self.d_v, self.d_a, self.d_l)
        
        #Transformation dimension layers
        self.num_vectors = 500
        self.trans_l_early = self.get_network(self_type='l_mem', layers=3)
        self.trans_v_early = self.get_network(self_type='v_mem', layers=3)
        self.trans_a_early = self.get_network(self_type='a_mem', layers=3)
        self.proj_l_e = nn.Linear(512, 32, bias=False)
        self.proj_v_e = nn.Linear(self.num_vectors, 32, bias=False)
        self.proj_a_e = nn.Linear(self.num_vectors, 32, bias=False)
        '''
        self.transfm_a2l = nn.Linear(self.num_vectors, 512)
        self.transfm_v2l = nn.Linear(self.num_vectors, 512)
        self.transfm_l2a = nn.Linear(512, self.num_vectors)
        self.transfm_l2v = nn.Linear(512, self.num_vectors)
        self.transfm_v2a = nn.Linear(self.num_vectors, self.num_vectors)
        self.transfm_a2v = nn.Linear(self.num_vectors, self.num_vectors)
        '''
        self.transfm_a2l = nn.Linear(combined_dim, combined_dim)
        self.transfm_v2l = nn.Linear(combined_dim, combined_dim)
        self.transfm_l2a = nn.Linear(combined_dim, combined_dim)
        self.transfm_l2v = nn.Linear(combined_dim, combined_dim)
        self.transfm_v2a = nn.Linear(combined_dim, combined_dim)
        self.transfm_a2v = nn.Linear(combined_dim, combined_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl', 'av2l', 'va2l']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va', 'lv2a', 'vl2a']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av', 'la2v', 'al2v']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
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
        #txt = torch.cat((txt, txt), 0)
        #mask = torch.cat((mask, mask), 0)
        #segment = torch.cat((segment, segment), 0)
        #img = torch.cat((img, img), 0)
        #audio = torch.cat((audio, audio), 0)
        #poster = torch.cat((poster, poster), 0)
        x_l = self.enc(txt, mask, segment)
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v = img.transpose(1, 2)
        #x_a = self.audio_enc(audio)
        x_a = audio.transpose(1, 2)
        #print(x_l.size(), x_v.size(), x_a.size())
       # proj_x_p_e = self.proj2_poster(poster)
       
        tok_l = tok_a = tok_v = 0#-1

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        #print(proj_x_l.size(), proj_x_v.size(), proj_x_a.size())
        if proj_x_l.size(0) != 512:
            proj_x_l = self.transfm_2dim(proj_x_l, 0, 512)
        if proj_x_a.size(0) != self.num_vectors:
            proj_x_a = self.transfm_2dim(proj_x_a, 0, self.num_vectors)
        if proj_x_v.size(0) != self.num_vectors:
            proj_x_v = self.transfm_2dim(proj_x_v, 0, self.num_vectors)
            
        #Parallel fusion
        proj_x_l_e = self.proj_l_e(proj_x_l.permute(2, 1, 0)).permute(2, 1, 0)
        proj_x_a_e = self.proj_a_e(proj_x_a.permute(2, 1, 0)).permute(2, 1, 0)
        proj_x_v_e = self.proj_v_e(proj_x_v.permute(2, 1, 0)).permute(2, 1, 0)
        h_l = self.trans_l_early(proj_x_l_e)[tok_l]
        h_a = self.trans_a_early(proj_x_a_e)[tok_a]
        h_v = self.trans_v_early(proj_x_v_e)[tok_v]
        last_h_early, zx = self.gmu_early(h_l, h_v, h_a)

        if self.lonly:
            # (V,A) --> L
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)    # Dimension (L, N, d_v)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)    # Dimension (L, N, d_a)

            # Feature Dimension Transformation
            #t_h_a_with_vs = self.transfm_a2l(h_a_with_vs.permute(2, 1, 0)).permute(2, 1, 0)
            #t_h_v_with_as = self.transfm_v2l(h_v_with_as.permute(2, 1, 0)).permute(2, 1, 0)
            t_h_a_with_vs = self.transfm_a2l(h_a_with_vs[tok_a])
            t_h_v_with_as = self.transfm_v2l(h_v_with_as[tok_v])
            # GMU Middle --------
            h_l_gmu, z1_l = self.gmu_l_m(t_h_v_with_as, t_h_a_with_vs)
            #h_ls = self.trans_l_with_v2a(proj_x_l, h_l_gmu, h_l_gmu)
            # GMU Top ---------
            h_l_with_v2a = self.trans_l_with_v2a(proj_x_l, h_a_with_vs, h_a_with_vs)    # Dimension (L, N, d_l)
            h_l_with_a2v = self.trans_l_with_a2v(proj_x_l, h_v_with_as, h_v_with_as)    # Dimension (L, N, d_l)
            # Residual conection
            h_l_with_v2a[tok_l] += t_h_a_with_vs
            h_l_with_a2v[tok_l] += t_h_v_with_as
            # Option 1 ---------
            h_ls_gmu, z2_l = self.gmu_l(h_l_with_a2v, h_l_with_v2a)
            h_ls_gmu += h_l_gmu
            # Option 2 ---------
            #sum_ls = self.proj_v2a(h_l_with_v2a) + self.proj_a2v(h_l_with_a2v)
            #h_ls = self.gmu_l(h_l_with_v2a, h_l_with_a2v, sum_ls)
            # Option 3 ---------
            #h_ls = self.proj_va2l(torch.cat([h_l_with_v2a, h_l_with_a2v], dim=2))
            #print(h_ls.size(), h_l_gmu.size())
            #------------
            # Residual conection
            #h_ls[:200,:,:] += h_l_gmu
            #h_ls = self.trans_l_mem(h_ls_gmu)
            #h_ls += h_ls_gmu
            #if type(h_ls) == tuple:
             #   h_ls = h_ls[0]
            last_h_l = last_hs = h_ls_gmu[tok_l]  # Take the last output for prediction
            #print(last_h_l.size())

        if self.aonly:
            # (L,V) --> A
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
            
            # Feature Dimension Transformation
            #t_h_l_with_vs = self.transfm_l2a(h_l_with_vs.permute(2, 1, 0)).permute(2, 1, 0)
            #t_h_v_with_ls = self.transfm_v2a(h_v_with_ls.permute(2, 1, 0)).permute(2, 1, 0)
            t_h_l_with_vs = self.transfm_l2a(h_l_with_vs[tok_l])
            t_h_v_with_ls = self.transfm_v2a(h_v_with_ls[tok_v])
            # GMU Middle --------
            h_a_gmu, z1_a = self.gmu_a_m(t_h_l_with_vs, t_h_v_with_ls)
            #h_as = self.trans_a_with_l2v(proj_x_a, h_a_gmu, h_a_gmu)
            # GMU Top --------
            h_a_with_v2l = self.trans_a_with_v2l(proj_x_a, h_l_with_vs, h_l_with_vs)
            h_a_with_l2v = self.trans_a_with_l2v(proj_x_a, h_v_with_ls, h_v_with_ls)
            # Residual conection
            h_a_with_v2l[tok_a] += t_h_l_with_vs
            h_a_with_l2v[tok_a] += t_h_v_with_ls
            # Option 1 ---------
            h_as_gmu, z2_a = self.gmu_a(h_a_with_v2l, h_a_with_l2v)
            h_as_gmu += h_a_gmu
            # Option 2 ---------
            #sum_as = self.proj_l2v(h_a_with_l2v) + self.proj_v2l(h_a_with_v2l)
            #h_as = self.gmu_a(h_a_with_l2v, h_a_with_v2l, sum_as)
            # Option 3 ---------
            #h_as = self.proj_vl2a(torch.cat([h_a_with_l2v, h_a_with_v2l], dim=2))
            #print(h_as.size())
            #------------
            # Residual conection
            #h_as = h_a_gmu[:200,:,:] + h_as
            #h_as = self.trans_a_mem(h_as_gmu)
            #h_as += h_as_gmu
            #if type(h_as) == tuple:
             #   h_as = h_as[0]
            last_h_a = last_hs = h_as_gmu[tok_a]
            #print(last_h_a.size())

        if self.vonly:
            # (L,A) --> V
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
            
            # Feature Dimension Transformation
            #t_h_l_with_as = self.transfm_l2v(h_l_with_as.permute(2, 1, 0)).permute(2, 1, 0)
            #t_h_a_with_ls = self.transfm_a2v(h_a_with_ls.permute(2, 1, 0)).permute(2, 1, 0)
            t_h_l_with_as = self.transfm_l2v(h_l_with_as[tok_l])
            t_h_a_with_ls = self.transfm_a2v(h_a_with_ls[tok_a])
            # GMU Middle --------
            h_v_gmu, z1_v = self.gmu_v_m(t_h_l_with_as, t_h_a_with_ls)
            #h_vs = self.trans_v_with_l2a(proj_x_v, h_v_gmu, h_v_gmu)
            # GMU Top --------
            h_v_with_a2l = self.trans_v_with_a2l(proj_x_v, h_l_with_as, h_l_with_as)
            h_v_with_l2a = self.trans_v_with_l2a(proj_x_v, h_a_with_ls, h_a_with_ls)
            # Residual conection
            h_v_with_a2l[tok_v] += t_h_l_with_as
            h_v_with_l2a[tok_v] += t_h_a_with_ls
            # Option 1 ---------
            h_vs_gmu, z2_v = self.gmu_v(h_v_with_a2l, h_v_with_l2a)
            h_vs_gmu += h_v_gmu
            # Option 2 ---------
            #sum_vs = self.proj_l2a(h_v_with_l2a) + self.proj_a2l(h_v_with_a2l)
            #h_vs = self.gmu_v(h_v_with_l2a, h_v_with_a2l, sum_vs)
            # Option 3 ---------
            #h_vs = self.proj_la2v(torch.cat([h_v_with_l2a, h_v_with_a2l], dim=2))
            #print(h_vs.size())
            #------------
            # Residual conection
            #h_vs = h_v_gmu[:200,:,:] + h_vs
            #h_vs = self.trans_v_mem(h_vs_gmu)
            #h_vs += h_vs_gmu
            #if type(h_vs) == tuple:
             #   h_vs = h_vs[0]
            last_h_v = last_hs = h_vs_gmu[tok_v]
            #print(last_h_v.size())
        
        #last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a)
        last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a, last_h_early)#, self.proj_poster(poster)) #, h_early.squeeze(0))
        #print(last_hs.size())
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        if output_gate:
            #return self.out_layer(last_hs_proj), z
            #For CMU_MOSI
            #print(last_hs_proj.size())
            #print(out.size())
            return self.out_layer(last_hs_proj), z
        else:
            return self.out_layer(last_hs_proj)
   

class TranslatingMMTransformerGMUClf_residual_v4T(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model for Text, Video frames and Audio spectrogram with GMU late fusion.
        """
        super(TranslatingMMTransformerGMUClf_residual_v4T, self).__init__()
        self.args = args
        self.orig_d_l, self.orig_d_v, self.orig_d_a, args.orig_d_m = args.orig_d_l, args.orig_d_v, args.orig_d_a, 312
        self.d_l, self.d_a, self.d_v, self.d_m = 768, 768, 768, 768
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
        
        self.enc = BertEncoder(args)
        self.audio_enc = AudioEncoder(args)

        # 0. Project poster feature to 768 dim
        self.proj_poster  = nn.Linear(self.orig_d_v, self.d_v, bias=False)
        #self.proj2_poster = nn.Linear(self.orig_d_v, self.d_v, bias=False)
        # 0. Temporal Linear layers for Early fusion
        #------For the crossmodal layers sum
        #self.proj_v2a = nn.Linear(self.d_l, self.d_l)#, bias=False)
        #self.proj_a2v = nn.Linear(self.d_l, self.d_l)#, bias=False)
        #self.proj_v2l = nn.Linear(self.d_a, self.d_a)#, bias=False)
        #self.proj_l2v = nn.Linear(self.d_a, self.d_a)#, bias=False)
        #self.proj_l2a = nn.Linear(self.d_v, self.d_v)#, bias=False)
        #self.proj_a2l = nn.Linear(self.d_v, self.d_v)#, bias=False)
        #--- GMU instead of sum
        #------ GMU Middle
        #self.gmu_l_m = GatedMultimodalLayerFeatures(self.d_v, self.d_a, self.d_l)
        #self.gmu_v_m = GatedMultimodalLayerFeatures(self.d_l, self.d_a, self.d_v)
        #self.gmu_a_m = GatedMultimodalLayerFeatures(self.d_l, self.d_v, self.d_a)
        #------ Transformer Middle
   #     self.trans_l_with_va = self.get_network(self_type='v&a', layers=3)
   #     self.trans_v_with_la = self.get_network(self_type='l&a', layers=3)
   #     self.trans_a_with_lv = self.get_network(self_type='l&v', layers=3)
        #------ GMU Top
        #self.gmu_l = GatedMultimodalLayerFeatures(self.d_v, self.d_a, self.d_l)
        #self.gmu_v = GatedMultimodalLayerFeatures(self.d_l, self.d_a, self.d_v)
        #self.gmu_a = GatedMultimodalLayerFeatures(self.d_l, self.d_v, self.d_a)
        #--------- GMU Top with sum
        #self.gmu_l = GatedMultimodal3LayerFeatures(self.d_l, self.d_l, self.d_l, self.d_l)
        #self.gmu_v = GatedMultimodal3LayerFeatures(self.d_v, self.d_v, self.d_v, self.d_l)
        #self.gmu_a = GatedMultimodal3LayerFeatures(self.d_a, self.d_a, self.d_a, self.d_l)
        #-------------
        #self.proj_va2l = nn.Linear(self.d_l * 2, self.d_l, bias=False)
        #self.proj_vl2a = nn.Linear(self.d_a * 2, self.d_a, bias=False)
        #self.proj_la2v = nn.Linear(self.d_v * 2, self.d_v, bias=False)

        combined_dim = 768 # For GMU
        
        output_dim = args.n_classes        # This is actually not a hyperparameter :-)

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
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        #self.proj12 = nn.Linear(combined_dim, combined_dim)
        #self.proj22 = nn.Linear(combined_dim, combined_dim)
        #self.out_layer2 = nn.Linear(combined_dim, output_dim)
        #self.out_layer_final = nn.Linear(output_dim, output_dim)
        
        # GMU layer for fusing text and image and audio information
        self.gmu = TextShifting4Layer(self.d_l*2, self.d_v*2, self.d_a*2, self.d_v, self.d_l)
        #self.gmu = TextShifting4Layer(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l)
        #self.gmu = TextShifting5Layer(self.d_l*2, self.d_v*2, self.d_v*2, self.d_v, self.d_l, self.d_l)
        #self.gmuSimple = TextShifting4LayerSimple(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l)
        #self.gmuSimple = TextShifting3LayerSimple(self.d_l, self.d_v, self.d_a, self.d_l)
        
        #Transformation dimension layers
        self.transfm_a2l = nn.Linear(200, 512)
        self.transfm_v2l = nn.Linear(200, 512)
        self.transfm_l2a = nn.Linear(512, 200)
        self.transfm_l2v = nn.Linear(512, 200)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl', 'av2l', 'va2l']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va', 'lv2a', 'vl2a']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av', 'la2v', 'al2v']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type in ['l_mem', 'v&a']:
        # With GMU
            #embed_dim, attn_dropout = self.d_l, self.attn_dropout
        # Without GMU (normal concat)
            embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
        elif self_type in ['a_mem', 'l&v']:
            #embed_dim, attn_dropout = self.d_a, self.attn_dropout
            embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout
        elif self_type in ['v_mem', 'l&a']:
            #embed_dim, attn_dropout = self.d_v, self.attn_dropout
            embed_dim, attn_dropout = 2*self.d_v, self.attn_dropout
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
        
            
    def forward(self, txt, mask, segment, img, audio, poster, output_gate=False):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        #txt = torch.cat((txt, txt), 0)
        #mask = torch.cat((mask, mask), 0)
        #segment = torch.cat((segment, segment), 0)
        #img = torch.cat((img, img), 0)
        #audio = torch.cat((audio, audio), 0)
        #poster = torch.cat((poster, poster), 0)
        x_l = self.enc(txt, mask, segment)
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v = img.transpose(1, 2)
        x_a = self.audio_enc(audio)
        
       # proj_x_l_e = self.proj_l_e(x_l.permute(2, 0, 1))
       # proj_x_a_e = self.proj_a_e(x_a.permute(2, 0, 1))
       # proj_x_v_e = self.proj_v_e(x_v.permute(2, 0, 1))
       # proj_x_p_e = self.proj2_poster(poster)
        
        #Early fusion
       # h_early, zx = self.gmuSimple(proj_x_l_e, proj_x_a_e, proj_x_v_e, proj_x_p_e)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        
        if proj_x_l.size(0) != 512:
            proj_x_l = self.transfm_2dim(proj_x_l, 0, 512)
        if proj_x_a.size(0) != 200:
            proj_x_a = self.transfm_2dim(proj_x_a, 0, 200)
        if proj_x_v.size(0) != 200:
            proj_x_v = self.transfm_2dim(proj_x_v, 0, 200)

        if self.lonly:
            # (V,A) --> L
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)    # Dimension (L, N, d_v)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)    # Dimension (L, N, d_a)
            
            # Feature Dimension Transformation
            t_h_a_with_vs = self.transfm_a2l(h_a_with_vs.permute(2, 1, 0)).permute(2, 1, 0)
            t_h_v_with_as = self.transfm_v2l(h_v_with_as.permute(2, 1, 0)).permute(2, 1, 0)
            # GMU Middle --------
            #h_l_gmu, z1_l = self.gmu_l_m(t_h_v_with_as, t_h_a_with_vs)
     #       h_l_trans = self.trans_l_with_va(torch.cat([t_h_v_with_as, t_h_a_with_vs], dim=2))
            # GMU Top ---------
            h_l_with_v2a = self.trans_l_with_v2a(proj_x_l, h_a_with_vs, h_a_with_vs)    # Dimension (L, N, d_l)
            h_l_with_a2v = self.trans_l_with_a2v(proj_x_l, h_v_with_as, h_v_with_as)    # Dimension (L, N, d_l)
            # Residual conection
      ##      h_l_with_v2a += t_h_a_with_vs
        ##    h_l_with_a2v += t_h_v_with_as
            # Option 1 ---------
            h_ls_trans = self.trans_l_mem(torch.cat((h_l_with_a2v, h_l_with_v2a), dim=2))
     #       h_ls_trans[-1] += h_l_trans[-1]
            # Option 2 ---------
            #sum_ls = self.proj_v2a(h_l_with_v2a) + self.proj_a2v(h_l_with_a2v)
            #h_ls = self.gmu_l(h_l_with_v2a, h_l_with_a2v, sum_ls)
            # Option 3 ---------
            #h_ls = self.proj_va2l(torch.cat([h_l_with_v2a, h_l_with_a2v], dim=2))
            #print(h_ls.size(), h_l_gmu.size())
            # Normal Option --------
            #h_ls = self.trans_l_mem(torch.cat((h_l_with_v2a, h_l_with_a2v), dim=2))
            #h_ls += h_ls_gmu
            #if type(h_ls) == tuple:
             #   h_ls = h_ls[0]
            #last_h_l = last_hs = h_ls[-1] #+ h_ls_gmu[-1]
            #------------
            '''
            # Residual conection
            #h_ls[:200,:,:] += h_l_gmu
            h_ls = self.trans_l_mem(h_ls_gmu)
            #h_ls += h_ls_gmu
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1] + h_ls_gmu[-1]  # Take the last output for prediction
            '''
            last_h_l = last_hs = h_ls_trans[-1]
            #last_h_l = last_hs = h_l_with_a2v[-1]

        if self.aonly:
            # (L,V) --> A
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
            
            # Feature Dimension Transformation
            t_h_l_with_vs = self.transfm_l2a(h_l_with_vs.permute(2, 1, 0)).permute(2, 1, 0)
            # GMU Middle --------
            #h_a_gmu, z1_a = self.gmu_a_m(t_h_l_with_vs, h_v_with_ls)
      #      h_a_trans = self.trans_a_with_lv(torch.cat([t_h_l_with_vs, h_v_with_ls], dim=2))
            #h_as = self.trans_a_with_l2v(proj_x_a, h_a_gmu, h_a_gmu)
            # GMU Top --------
            h_a_with_v2l = self.trans_a_with_v2l(proj_x_a, h_l_with_vs, h_l_with_vs)
            h_a_with_l2v = self.trans_a_with_l2v(proj_x_a, h_v_with_ls, h_v_with_ls)
            # Residual conection
      ##      h_a_with_v2l += t_h_l_with_vs
        ##    h_a_with_l2v += h_v_with_ls
            # Option 1 ---------
            #h_as_gmu, z2_a = self.gmu_a(h_a_with_v2l, h_a_with_l2v)
            h_as_trans = self.trans_a_mem(torch.cat((h_a_with_v2l, h_a_with_l2v), dim=2))
     #       h_as_trans[-1] += h_a_trans[-1]
            # Option 2 ---------
            #sum_as = self.proj_l2v(h_a_with_l2v) + self.proj_v2l(h_a_with_v2l)
            #h_as = self.gmu_a(h_a_with_l2v, h_a_with_v2l, sum_as)
            # Option 3 ---------
            #h_as = self.proj_vl2a(torch.cat([h_a_with_l2v, h_a_with_v2l], dim=2))
            #print(h_as.size())
            # Normal Option --------
            #h_as = self.trans_a_mem(torch.cat((h_a_with_v2l, h_a_with_l2v), dim=2))
            #h_ls += h_ls_gmu
            #if type(h_as) == tuple:
             #   h_as = h_as[0]
            #last_h_a = last_hs = h_as[-1]
            #------------
            '''
            # Residual conection
            #h_as = h_a_gmu[:200,:,:] + h_as
            h_as = self.trans_a_mem(h_as_gmu)
            #h_as += h_as_gmu
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1] + h_as_gmu[-1]
            '''
            last_h_a = last_hs = h_as_trans[-1]
            #last_h_a = last_hs = h_a_with_v2l[-1]
            #print(last_h_a.size())

        if self.vonly:
            # (L,A) --> V
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
            
            # Feature Dimension Transformation
            t_h_l_with_as = self.transfm_l2v(h_l_with_as.permute(2, 1, 0)).permute(2, 1, 0)
            # GMU Middle --------
            #h_v_gmu, z1_v = self.gmu_v_m(t_h_l_with_as, h_a_with_ls)
       #     h_v_trans = self.trans_v_with_la(torch.cat([t_h_l_with_as, h_a_with_ls], dim=2))
            #h_vs = self.trans_v_with_l2a(proj_x_v, h_v_gmu, h_v_gmu)
            # GMU Top --------
            h_v_with_a2l = self.trans_v_with_a2l(proj_x_v, h_l_with_as, h_l_with_as)
            h_v_with_l2a = self.trans_v_with_l2a(proj_x_v, h_a_with_ls, h_a_with_ls)
            # Residual conection
      ##      h_v_with_a2l += t_h_l_with_as
        ##    h_v_with_l2a += h_a_with_ls
            # Option 1 ---------
            #h_vs_gmu, z2_v = self.gmu_v(h_v_with_a2l, h_v_with_l2a)
            #h_vs_gmu += h_v_gmu
            h_vs_trans = self.trans_v_mem(torch.cat((h_v_with_a2l, h_v_with_l2a), dim=2))
      #      h_vs_trans[-1] += h_v_trans[-1]
            # Option 2 ---------
            #sum_vs = self.proj_l2a(h_v_with_l2a) + self.proj_a2l(h_v_with_a2l)
            #h_vs = self.gmu_v(h_v_with_l2a, h_v_with_a2l, sum_vs)
            # Option 3 ---------
            #h_vs = self.proj_la2v(torch.cat([h_v_with_l2a, h_v_with_a2l], dim=2))
            #print(h_vs.size())
            # Normal Option --------
            #h_vs = self.trans_v_mem(torch.cat((h_v_with_a2l, h_v_with_l2a), dim=2))
            #if type(h_vs) == tuple:
             #   h_vs = h_vs[0]
            #last_h_v = last_hs = h_vs[-1]
            #------------
            '''
            # Residual conection
            #h_vs = h_v_gmu[:200,:,:] + h_vs
            h_vs = self.trans_v_mem(h_vs_gmu)
            h_vs += h_vs_gmu
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1] + h_vs_gmu[-1]
            '''
            last_h_v = last_hs = h_vs_trans[-1]
            #last_h_v = last_hs = h_v_with_l2a[-1]
        
        #last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a)
        last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a, self.proj_poster(poster)) #, h_early.squeeze(0))
        #print(last_hs.size())
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        if output_gate:
            return self.out_layer(last_hs_proj), z#torch.cat((z1_l[-1], z2_l[-1],
                                                   #         z1_v[-1], z2_v[-1],
                                                    #        z1_a[-1], z2_a[-1]), dim=1)
        else:
            return self.out_layer(last_hs_proj)
            
class TranslatingMMTransformerGMUClf_residual_v4T_v2(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model for Text, Video frames and Audio spectrogram with GMU late fusion.
        """
        super(TranslatingMMTransformerGMUClf_residual_v4T_v2, self).__init__()
        self.args = args
        self.orig_d_l, self.orig_d_v, self.orig_d_a, args.orig_d_m = args.orig_d_l, args.orig_d_v, args.orig_d_a, 312
        self.d_l, self.d_a, self.d_v, self.d_m = 768, 768, 768, 768
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
        
        self.enc = BertEncoder(args)
        self.audio_enc = AudioEncoder(args)

        # 0. Project poster feature to 768 dim
        self.proj_poster  = nn.Linear(self.orig_d_v, self.d_v, bias=False)
        #self.proj2_poster = nn.Linear(self.orig_d_v, self.d_v, bias=False)
        # 0. Temporal Linear layers for Early fusion
        #------For the crossmodal layers sum
        #self.proj_v2a = nn.Linear(self.d_l, self.d_l)#, bias=False)
        #self.proj_a2v = nn.Linear(self.d_l, self.d_l)#, bias=False)
        #self.proj_v2l = nn.Linear(self.d_a, self.d_a)#, bias=False)
        #self.proj_l2v = nn.Linear(self.d_a, self.d_a)#, bias=False)
        #self.proj_l2a = nn.Linear(self.d_v, self.d_v)#, bias=False)
        #self.proj_a2l = nn.Linear(self.d_v, self.d_v)#, bias=False)
        #--- GMU instead of sum
        #------ GMU Middle
        #self.gmu_l_m = GatedMultimodalLayerFeatures(self.d_v, self.d_a, self.d_l)
        #self.gmu_v_m = GatedMultimodalLayerFeatures(self.d_l, self.d_a, self.d_v)
        #self.gmu_a_m = GatedMultimodalLayerFeatures(self.d_l, self.d_v, self.d_a)
        #------ Transformer Middle
        self.trans_l_with_va = self.get_network(self_type='v&a', layers=3)
        self.trans_v_with_la = self.get_network(self_type='l&a', layers=3)
        self.trans_a_with_lv = self.get_network(self_type='l&v', layers=3)
        #------ GMU Top
        #self.gmu_l = GatedMultimodalLayerFeatures(self.d_v, self.d_a, self.d_l)
        #self.gmu_v = GatedMultimodalLayerFeatures(self.d_l, self.d_a, self.d_v)
        #self.gmu_a = GatedMultimodalLayerFeatures(self.d_l, self.d_v, self.d_a)
        #--------- GMU Top with sum
        #self.gmu_l = GatedMultimodal3LayerFeatures(self.d_l, self.d_l, self.d_l, self.d_l)
        #self.gmu_v = GatedMultimodal3LayerFeatures(self.d_v, self.d_v, self.d_v, self.d_l)
        #self.gmu_a = GatedMultimodal3LayerFeatures(self.d_a, self.d_a, self.d_a, self.d_l)
        #-------------
        #self.proj_va2l = nn.Linear(self.d_l * 2, self.d_l, bias=False)
        #self.proj_vl2a = nn.Linear(self.d_a * 2, self.d_a, bias=False)
        #self.proj_la2v = nn.Linear(self.d_v * 2, self.d_v, bias=False)

        combined_dim = 768 # For GMU
        
        output_dim = args.n_classes        # This is actually not a hyperparameter :-)

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
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        #self.proj12 = nn.Linear(combined_dim, combined_dim)
        #self.proj22 = nn.Linear(combined_dim, combined_dim)
        #self.out_layer2 = nn.Linear(combined_dim, output_dim)
        #self.out_layer_final = nn.Linear(output_dim, output_dim)
        
        # GMU layer for fusing text and image and audio information
        self.gmu = TextShifting4Layer(self.d_l*2, self.d_v*2, self.d_a*2, self.d_v, self.d_l)
        #self.gmu = TextShifting4Layer(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l)
        #self.gmu = TextShifting5Layer(self.d_l*2, self.d_v*2, self.d_v*2, self.d_v, self.d_l, self.d_l)
        #self.gmuSimple = TextShifting4LayerSimple(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l)
        #self.gmuSimple = TextShifting3LayerSimple(self.d_l, self.d_v, self.d_a, self.d_l)
        
        #Transformation dimension layers
        self.transfm_a2l = nn.Linear(200, 512)
        self.transfm_v2l = nn.Linear(200, 512)
        self.transfm_l2a = nn.Linear(512, 200)
        self.transfm_l2v = nn.Linear(512, 200)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl', 'av2l', 'va2l']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va', 'lv2a', 'vl2a']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av', 'la2v', 'al2v']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type in ['l_mem', 'v&a']:
        # With GMU
            #embed_dim, attn_dropout = self.d_l, self.attn_dropout
        # Without GMU (normal concat)
            embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
        elif self_type in ['a_mem', 'l&v']:
            #embed_dim, attn_dropout = self.d_a, self.attn_dropout
            embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout
        elif self_type in ['v_mem', 'l&a']:
            #embed_dim, attn_dropout = self.d_v, self.attn_dropout
            embed_dim, attn_dropout = 2*self.d_v, self.attn_dropout
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
        
            
    def forward(self, txt, mask, segment, img, audio, poster, output_gate=False):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        #txt = torch.cat((txt, txt), 0)
        #mask = torch.cat((mask, mask), 0)
        #segment = torch.cat((segment, segment), 0)
        #img = torch.cat((img, img), 0)
        #audio = torch.cat((audio, audio), 0)
        #poster = torch.cat((poster, poster), 0)
        x_l = self.enc(txt, mask, segment)
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v = img.transpose(1, 2)
        x_a = self.audio_enc(audio)
        
       # proj_x_l_e = self.proj_l_e(x_l.permute(2, 0, 1))
       # proj_x_a_e = self.proj_a_e(x_a.permute(2, 0, 1))
       # proj_x_v_e = self.proj_v_e(x_v.permute(2, 0, 1))
       # proj_x_p_e = self.proj2_poster(poster)
        
        #Early fusion
       # h_early, zx = self.gmuSimple(proj_x_l_e, proj_x_a_e, proj_x_v_e, proj_x_p_e)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        
        if proj_x_l.size(0) != 512:
            proj_x_l = self.transfm_2dim(proj_x_l, 0, 512)
        if proj_x_a.size(0) != 200:
            proj_x_a = self.transfm_2dim(proj_x_a, 0, 200)
        if proj_x_v.size(0) != 200:
            proj_x_v = self.transfm_2dim(proj_x_v, 0, 200)

        if self.lonly:
            # (V,A) --> L
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)    # Dimension (L, N, d_v)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)    # Dimension (L, N, d_a)
            
            # Feature Dimension Transformation
            t_h_a_with_vs = self.transfm_a2l(h_a_with_vs.permute(2, 1, 0)).permute(2, 1, 0)
            t_h_v_with_as = self.transfm_v2l(h_v_with_as.permute(2, 1, 0)).permute(2, 1, 0)
            # GMU Middle --------
            #h_l_gmu, z1_l = self.gmu_l_m(t_h_v_with_as, t_h_a_with_vs)
            h_l_trans = self.trans_l_with_va(torch.cat([t_h_v_with_as, t_h_a_with_vs], dim=2))
            # GMU Top ---------
            h_l_with_v2a = self.trans_l_with_v2a(proj_x_l, h_a_with_vs, h_a_with_vs)    # Dimension (L, N, d_l)
            h_l_with_a2v = self.trans_l_with_a2v(proj_x_l, h_v_with_as, h_v_with_as)    # Dimension (L, N, d_l)
            # Residual conection
            h_l_with_v2a += t_h_a_with_vs
            h_l_with_a2v += t_h_v_with_as
            # Option 1 ---------
            h_ls_trans = self.trans_l_mem(torch.cat((h_l_with_a2v, h_l_with_v2a), dim=2))
            h_ls_trans[-1] += h_l_trans[-1]
            # Option 2 ---------
            #sum_ls = self.proj_v2a(h_l_with_v2a) + self.proj_a2v(h_l_with_a2v)
            #h_ls = self.gmu_l(h_l_with_v2a, h_l_with_a2v, sum_ls)
            # Option 3 ---------
            #h_ls = self.proj_va2l(torch.cat([h_l_with_v2a, h_l_with_a2v], dim=2))
            #print(h_ls.size(), h_l_gmu.size())
            # Normal Option --------
            #h_ls = self.trans_l_mem(torch.cat((h_l_with_v2a, h_l_with_a2v), dim=2))
            #h_ls += h_ls_gmu
            #if type(h_ls) == tuple:
             #   h_ls = h_ls[0]
            #last_h_l = last_hs = h_ls[-1] #+ h_ls_gmu[-1]
            #------------
            '''
            # Residual conection
            #h_ls[:200,:,:] += h_l_gmu
            h_ls = self.trans_l_mem(h_ls_gmu)
            #h_ls += h_ls_gmu
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1] + h_ls_gmu[-1]  # Take the last output for prediction
            '''
            last_h_l = last_hs = h_ls_trans[-1]
            #last_h_l = last_hs = h_l_with_a2v[-1]

        if self.aonly:
            # (L,V) --> A
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
            
            # Feature Dimension Transformation
            t_h_l_with_vs = self.transfm_l2a(h_l_with_vs.permute(2, 1, 0)).permute(2, 1, 0)
            # GMU Middle --------
            #h_a_gmu, z1_a = self.gmu_a_m(t_h_l_with_vs, h_v_with_ls)
            h_a_trans = self.trans_a_with_lv(torch.cat([t_h_l_with_vs, h_v_with_ls], dim=2))
            #h_as = self.trans_a_with_l2v(proj_x_a, h_a_gmu, h_a_gmu)
            # GMU Top --------
            h_a_with_v2l = self.trans_a_with_v2l(proj_x_a, h_l_with_vs, h_l_with_vs)
            h_a_with_l2v = self.trans_a_with_l2v(proj_x_a, h_v_with_ls, h_v_with_ls)
            # Residual conection
            h_a_with_v2l += t_h_l_with_vs
            h_a_with_l2v += h_v_with_ls
            # Option 1 ---------
            #h_as_gmu, z2_a = self.gmu_a(h_a_with_v2l, h_a_with_l2v)
            h_as_trans = self.trans_a_mem(torch.cat((h_a_with_v2l, h_a_with_l2v), dim=2))
            h_as_trans[-1] += h_a_trans[-1]
            # Option 2 ---------
            #sum_as = self.proj_l2v(h_a_with_l2v) + self.proj_v2l(h_a_with_v2l)
            #h_as = self.gmu_a(h_a_with_l2v, h_a_with_v2l, sum_as)
            # Option 3 ---------
            #h_as = self.proj_vl2a(torch.cat([h_a_with_l2v, h_a_with_v2l], dim=2))
            #print(h_as.size())
            # Normal Option --------
            #h_as = self.trans_a_mem(torch.cat((h_a_with_v2l, h_a_with_l2v), dim=2))
            #h_ls += h_ls_gmu
            #if type(h_as) == tuple:
             #   h_as = h_as[0]
            #last_h_a = last_hs = h_as[-1]
            #------------
            '''
            # Residual conection
            #h_as = h_a_gmu[:200,:,:] + h_as
            h_as = self.trans_a_mem(h_as_gmu)
            #h_as += h_as_gmu
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1] + h_as_gmu[-1]
            '''
            last_h_a = last_hs = h_as_trans[-1]
            #last_h_a = last_hs = h_a_with_v2l[-1]
            #print(last_h_a.size())

        if self.vonly:
            # (L,A) --> V
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
            
            # Feature Dimension Transformation
            t_h_l_with_as = self.transfm_l2v(h_l_with_as.permute(2, 1, 0)).permute(2, 1, 0)
            # GMU Middle --------
            #h_v_gmu, z1_v = self.gmu_v_m(t_h_l_with_as, h_a_with_ls)
            h_v_trans = self.trans_v_with_la(torch.cat([t_h_l_with_as, h_a_with_ls], dim=2))
            #h_vs = self.trans_v_with_l2a(proj_x_v, h_v_gmu, h_v_gmu)
            # GMU Top --------
            h_v_with_a2l = self.trans_v_with_a2l(proj_x_v, h_l_with_as, h_l_with_as)
            h_v_with_l2a = self.trans_v_with_l2a(proj_x_v, h_a_with_ls, h_a_with_ls)
            # Residual conection
            h_v_with_a2l += t_h_l_with_as
            h_v_with_l2a += h_a_with_ls
            # Option 1 ---------
            #h_vs_gmu, z2_v = self.gmu_v(h_v_with_a2l, h_v_with_l2a)
            #h_vs_gmu += h_v_gmu
            h_vs_trans = self.trans_v_mem(torch.cat((h_v_with_a2l, h_v_with_l2a), dim=2))
            h_vs_trans[-1] += h_v_trans[-1]
            # Option 2 ---------
            #sum_vs = self.proj_l2a(h_v_with_l2a) + self.proj_a2l(h_v_with_a2l)
            #h_vs = self.gmu_v(h_v_with_l2a, h_v_with_a2l, sum_vs)
            # Option 3 ---------
            #h_vs = self.proj_la2v(torch.cat([h_v_with_l2a, h_v_with_a2l], dim=2))
            #print(h_vs.size())
            # Normal Option --------
            #h_vs = self.trans_v_mem(torch.cat((h_v_with_a2l, h_v_with_l2a), dim=2))
            #if type(h_vs) == tuple:
             #   h_vs = h_vs[0]
            #last_h_v = last_hs = h_vs[-1]
            #------------
            '''
            # Residual conection
            #h_vs = h_v_gmu[:200,:,:] + h_vs
            h_vs = self.trans_v_mem(h_vs_gmu)
            h_vs += h_vs_gmu
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1] + h_vs_gmu[-1]
            '''
            last_h_v = last_hs = h_vs_trans[-1]
            #last_h_v = last_hs = h_v_with_l2a[-1]
        
        #last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a)
        last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a, self.proj_poster(poster)) #, h_early.squeeze(0))
        #print(last_hs.size())
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        if output_gate:
            return self.out_layer(last_hs_proj), z#torch.cat((z1_l[-1], z2_l[-1],
                                                   #         z1_v[-1], z2_v[-1],
                                                    #        z1_a[-1], z2_a[-1]), dim=1)
        else:
            return self.out_layer(last_hs_proj)

class TranslatingMMTransformerGMUClf_early_fusion(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model for Text, Video frames and Audio spectrogram with GMU late fusion.
        """
        super(TranslatingMMTransformerGMUClf_early_fusion, self).__init__()
        self.args = args
        self.orig_d_l, self.orig_d_v, self.orig_d_a, args.orig_d_m = args.orig_d_l, args.orig_d_v, args.orig_d_a, 312
        self.d_l, self.d_a, self.d_v, self.d_m = 768, 768, 768, 768
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
        
        self.enc = BertEncoder(args)
        self.audio_enc = AudioEncoder(args)

        # 0. Project poster feature to 768 dim
        self.proj_poster  = nn.Linear(self.orig_d_v, self.d_v, bias=False)
        # 0. Temporal Linear layers for Early fusion
        self.proj_l_e = nn.Linear(512, 64, bias=False)
        self.proj_v_e = nn.Linear(200, 64, bias=False)
        self.proj_a_e = nn.Linear(200, 64, bias=False)
        self.proj2_l_e = nn.Linear(self.orig_d_l, self.d_l, bias=False)
        self.proj2_v_e = nn.Linear(self.orig_d_v, self.d_v, bias=False)
        self.proj2_a_e = nn.Linear(self.orig_d_a, self.d_a, bias=False)
        #--- GMU instead of sum
        #------ GMU Middle
    #    self.gmu_l_m = GatedMultimodalLayerFeatures(self.d_v, self.d_a, self.d_l)
     #   self.gmu_v_m = GatedMultimodalLayerFeatures(self.d_l, self.d_a, self.d_v)
      #  self.gmu_a_m = GatedMultimodalLayerFeatures(self.d_l, self.d_v, self.d_a)
        #------ GMU Top
       # self.gmu_l = GatedMultimodalLayerFeatures(self.d_v, self.d_a, self.d_l)
        #self.gmu_v = GatedMultimodalLayerFeatures(self.d_l, self.d_a, self.d_v)
        #self.gmu_a = GatedMultimodalLayerFeatures(self.d_l, self.d_v, self.d_a)
        #--------- GMU Top with sum
        #self.gmu_l = GatedMultimodal3LayerFeatures(self.d_l, self.d_l, self.d_l, self.d_l)
        #self.gmu_v = GatedMultimodal3LayerFeatures(self.d_v, self.d_v, self.d_v, self.d_l)
        #self.gmu_a = GatedMultimodal3LayerFeatures(self.d_a, self.d_a, self.d_a, self.d_l)
        #-------------
        #self.proj_va2l = nn.Linear(self.d_l * 2, self.d_l, bias=False)
        #self.proj_vl2a = nn.Linear(self.d_a * 2, self.d_a, bias=False)
        #self.proj_la2v = nn.Linear(self.d_v * 2, self.d_v, bias=False)

        combined_dim = 768 # For GMU
        
        output_dim = args.n_classes        # This is actually not a hyperparameter :-)
        '''
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
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        #self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        #self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
        #self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        #self.trans_h_mem = self.get_network(self_type='h_mem', layers=3)
        '''
        self.trans_l_early = self.get_network(self_type='l_mem', layers=3)
        self.trans_v_early = self.get_network(self_type='v_mem', layers=3)
        self.trans_a_early = self.get_network(self_type='a_mem', layers=3)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        #self.proj12 = nn.Linear(combined_dim, combined_dim)
        #self.proj22 = nn.Linear(combined_dim, combined_dim)
        #self.out_layer2 = nn.Linear(combined_dim, output_dim)
        #self.out_layer_final = nn.Linear(output_dim, output_dim)
        
        # GMU layer for fusing text and image and audio information
        #self.gmu = TextShifting4Layer(self.d_l*2, self.d_v*2, self.d_a*2, self.d_v, self.d_l)
        #self.gmu = TextShifting4Layer(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l)
        #self.gmu = TextShifting5Layer(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l, self.d_l)
        self.gmu = GatedMultimodalLayer(self.d_v, self.d_l, self.d_l)
        #self.gmuSimple = TextShifting4LayerSimple(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l)
        self.gmu_early = GatedMultimodal3LayerFeatures(self.d_l, self.d_v, self.d_a, self.d_l)
                
        
        #Transformation dimension layers
        self.transfm_a2l = nn.Linear(200, 512)
        self.transfm_v2l = nn.Linear(200, 512)
        self.transfm_l2a = nn.Linear(512, 200)
        self.transfm_l2v = nn.Linear(512, 200)

    def get_network(self, self_type='l', layers=-1):
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
        elif self_type == 'h_mem':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
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
        
            
    def forward(self, txt, mask, segment, img, audio, poster, output_gate=False):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        #txt = torch.cat((txt, txt), 0)
        #mask = torch.cat((mask, mask), 0)
        #segment = torch.cat((segment, segment), 0)
        #img = torch.cat((img, img), 0)
        #audio = torch.cat((audio, audio), 0)
        #poster = torch.cat((poster, poster), 0)
        x_l = self.enc(txt, mask, segment)
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v = img.transpose(1, 2)
        x_a = self.audio_enc(audio)
        
        # Early Fusion
        if x_l.size(2) != 512:
            x1 = self.transfm_2dim(x_l, 2, 512)
        else: x1 = x_l
        if x_v.size(2) != 200:
            x2 = self.transfm_2dim(x_v, 2, 200)
        else: x2 = x_v
        if x_a.size(2) != 200:
            x3 = self.transfm_2dim(x_a, 2, 200)
        else: x3 = x_a
        proj_x_l_e = self.proj_l_e(x1)
        proj_x_v_e = self.proj_v_e(x2)
        proj_x_a_e = self.proj_a_e(x3)
        proj_x_l_e = self.proj2_l_e(proj_x_l_e.permute(2, 0, 1))
        proj_x_a_e = self.proj2_a_e(proj_x_a_e.permute(2, 0, 1))
        proj_x_v_e = self.proj2_v_e(proj_x_v_e.permute(2, 0, 1))
       # proj_x_l_e = self.proj_l_e(x_l.permute(2, 0, 1))
       # proj_x_a_e = self.proj_a_e(x_a.permute(2, 0, 1))
       # proj_x_v_e = self.proj_v_e(x_v.permute(2, 0, 1))
       # proj_x_p_e = self.proj2_poster(poster)
       # h_early, zx = self.gmuSimple(proj_x_l_e, proj_x_a_e, proj_x_v_e, proj_x_p_e)
        h_early, z_h = self.gmu_early(self.trans_l_early(proj_x_l_e),
                                      self.trans_v_early(proj_x_v_e),
                                      self.trans_a_early(proj_x_a_e))
        if type(h_early) == tuple:
            h_early = h_early[0]
        last_h_early = last_hs = h_early[-1]

        
        last_hs, z = self.gmu(self.proj_poster(poster), last_h_early)
        #print(last_hs.size())
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        if output_gate:
            return self.out_layer(last_hs_proj), z#torch.cat((z1_l[-1], z2_l[-1],
                                                   #         z1_v[-1], z2_v[-1],
                                                    #        z1_a[-1], z2_a[-1]), dim=1)
        else:
            return self.out_layer(last_hs_proj)
          
            
class TranslatingMMTransformerMAGClf(nn.Module):
    def __init__(self, args):
        """
        Construct a TMulT model for Text, Video frames and Audio spectrogram with MAG fusion.
        """
        super(TranslatingMMTransformerMAGClf, self).__init__()
        self.args = args
        self.orig_d_l, self.orig_d_v, self.orig_d_a, args.orig_d_m = args.orig_d_l, args.orig_d_v, args.orig_d_a, 312
        self.d_l, self.d_a, self.d_v, self.d_m = 768, 768, 768, 768
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
        
        self.enc = BertEncoder(args)
        self.audio_enc = AudioEncoder(args)

        # 0. Project poster feature to 768 dim
        self.proj_poster  = nn.Linear(self.orig_d_v, self.d_v, bias=False)
        #self.proj2_poster = nn.Linear(self.orig_d_v, self.d_v, bias=False)
        # 0. Temporal Linear layers for Early fusion
        #------For the crossmodal layers sum
        #self.proj_v2a = nn.Linear(self.d_l, self.d_l)#, bias=False)
        #self.proj_a2v = nn.Linear(self.d_l, self.d_l)#, bias=False)
        #self.proj_v2l = nn.Linear(self.d_a, self.d_a)#, bias=False)
        #self.proj_l2v = nn.Linear(self.d_a, self.d_a)#, bias=False)
        #self.proj_l2a = nn.Linear(self.d_v, self.d_v)#, bias=False)
        #self.proj_a2l = nn.Linear(self.d_v, self.d_v)#, bias=False)
        #--- GMU instead of sum
        #------ GMU Middle
        self.gmu_l_m = GatedMultimodalLayerFeatures(self.d_v, self.d_a, self.d_l)
        self.gmu_v_m = GatedMultimodalLayerFeatures(self.d_l, self.d_a, self.d_v)
        self.gmu_a_m = GatedMultimodalLayerFeatures(self.d_l, self.d_v, self.d_a)
        #------ MAG Top
        beta_shift, dropout_prob = 1e-3, 0.5
        self.gmu_l = MAG(self.d_l, beta_shift, dropout_prob)
        self.gmu_v = MAG(self.d_v, beta_shift, dropout_prob)
        self.gmu_a = MAG(self.d_a, beta_shift, dropout_prob)
        #--------- GMU Top with sum
        #self.gmu_l = GatedMultimodal3LayerFeatures(self.d_l, self.d_l, self.d_l, self.d_l)
        #self.gmu_v = GatedMultimodal3LayerFeatures(self.d_v, self.d_v, self.d_v, self.d_l)
        #self.gmu_a = GatedMultimodal3LayerFeatures(self.d_a, self.d_a, self.d_a, self.d_l)
        #-------------
        #self.proj_va2l = nn.Linear(self.d_l * 2, self.d_l, bias=False)
        #self.proj_vl2a = nn.Linear(self.d_a * 2, self.d_a, bias=False)
        #self.proj_la2v = nn.Linear(self.d_v * 2, self.d_v, bias=False)

        combined_dim = 768 # For GMU
        
        output_dim = args.n_classes        # This is actually not a hyperparameter :-)

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
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        #self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        #self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
        #self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        #self.proj12 = nn.Linear(combined_dim, combined_dim)
        #self.proj22 = nn.Linear(combined_dim, combined_dim)
        #self.out_layer2 = nn.Linear(combined_dim, output_dim)
        #self.out_layer_final = nn.Linear(output_dim, output_dim)
        
        # GMU layer for fusing text and image and audio information
        #self.gmu = TextShifting4Layer(self.d_l*2, self.d_v*2, self.d_a*2, self.d_v, self.d_l)
        self.gmu = TextShifting4Layer(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l)
        #self.gmu = TextShifting5Layer(self.d_l*2, self.d_v*2, self.d_v*2, self.d_v, self.d_l, self.d_l)
        #self.gmuSimple = TextShifting4LayerSimple(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l)
        #self.gmuSimple = TextShifting3LayerSimple(self.d_l, self.d_v, self.d_a, self.d_l)
        
        #Transformation dimension layers
        self.transfm_a2l = nn.Linear(200, 512)
        self.transfm_v2l = nn.Linear(200, 512)
        self.transfm_l2a = nn.Linear(512, 200)
        self.transfm_l2v = nn.Linear(512, 200)

    def get_network(self, self_type='l', layers=-1):
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
        
            
    def forward(self, txt, mask, segment, img, audio, poster, output_gate=False):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        #txt = torch.cat((txt, txt), 0)
        #mask = torch.cat((mask, mask), 0)
        #segment = torch.cat((segment, segment), 0)
        #img = torch.cat((img, img), 0)
        #audio = torch.cat((audio, audio), 0)
        #poster = torch.cat((poster, poster), 0)
        x_l = self.enc(txt, mask, segment)
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v = img.transpose(1, 2)
        x_a = self.audio_enc(audio)
        
       # proj_x_l_e = self.proj_l_e(x_l.permute(2, 0, 1))
       # proj_x_a_e = self.proj_a_e(x_a.permute(2, 0, 1))
       # proj_x_v_e = self.proj_v_e(x_v.permute(2, 0, 1))
       # proj_x_p_e = self.proj2_poster(poster)
        
        #Early fusion
       # h_early, zx = self.gmuSimple(proj_x_l_e, proj_x_a_e, proj_x_v_e, proj_x_p_e)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        
        if proj_x_l.size(0) != 512:
            proj_x_l = self.transfm_2dim(proj_x_l, 0, 512)
        if proj_x_a.size(0) != 200:
            proj_x_a = self.transfm_2dim(proj_x_a, 0, 200)
        if proj_x_v.size(0) != 200:
            proj_x_v = self.transfm_2dim(proj_x_v, 0, 200)

        if self.lonly:
            # (V,A) --> L
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)    # Dimension (V, N, d_v)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)    # Dimension (A, N, d_a)
    #        attention_CLS = torch.cat((h_v_with_as[0], h_a_with_vs[0]), dim=1)
            
            # Feature Dimension Transformation
            t_h_a_with_vs = self.transfm_a2l(h_a_with_vs.permute(2, 1, 0)).permute(2, 1, 0)
            t_h_v_with_as = self.transfm_v2l(h_v_with_as.permute(2, 1, 0)).permute(2, 1, 0)
            # GMU Middle --------
   #         h_l_gmu, z1_l = self.gmu_l_m(t_h_v_with_as, t_h_a_with_vs)
            #h_ls = self.trans_l_with_v2a(proj_x_l, h_l_gmu, h_l_gmu)
            # GMU Top ---------
            h_l_with_v2a = self.trans_l_with_v2a(proj_x_l, h_a_with_vs, h_a_with_vs)    # Dimension (L, N, d_l)
            h_l_with_a2v = self.trans_l_with_a2v(proj_x_l, h_v_with_as, h_v_with_as)    # Dimension (L, N, d_l)
    #        attention_CLS = torch.cat((attention_CLS, h_l_with_a2v[0], h_l_with_v2a[0]), dim=1)
            # Residual conection
            h_l_with_v2a += t_h_a_with_vs
            h_l_with_a2v += t_h_v_with_as
            # Option 1 ---------
            #h_ls_gmu, z2_l = self.gmu_l(proj_x_l, h_l_with_a2v, h_l_with_v2a)
            h_ls_gmu = self.gmu_l(proj_x_l, h_l_with_a2v, h_l_with_v2a)
            h_ls = h_ls_gmu #+ h_l_gmu
            #h_ls = 0.6*F.normalize(h_ls_gmu) + 0.4*F.normalize(h_l_gmu)
            '''
            mean1 = torch.mean(z1_l[-1][:, :self.d_l], dim=1, keepdim=False)
            mean2 = torch.mean(z1_l[-1][:, self.d_l:], dim=1, keepdim=False)
            alpha1_l = 1 - (mean1 - mean2)
            mean1 = torch.mean(z2_l[-1][:, :self.d_l], dim=1, keepdim=False)
            mean2 = torch.mean(z2_l[-1][:, self.d_l:], dim=1, keepdim=False)
            alpha2_l = 1 - (mean1 - mean2)
            #print(h_ls_gmu.size(), h_l_gmu.size())
            h_ls = alpha2_l*F.normalize(h_ls_gmu.permute(0, 2, 1)) + alpha1_l*F.normalize(h_l_gmu.permute(0, 2, 1))
            h_ls = h_ls.permute(0, 2, 1)
           # '''
            # Option 2 ---------
            #sum_ls = self.proj_v2a(h_l_with_v2a) + self.proj_a2v(h_l_with_a2v)
            #h_ls = self.gmu_l(h_l_with_v2a, h_l_with_a2v, sum_ls)
            # Option 3 ---------
            #h_ls = self.proj_va2l(torch.cat([h_l_with_v2a, h_l_with_a2v], dim=2))
            #print(h_ls.size(), h_l_gmu.size())
            # Normal Option --------
            #h_ls = self.trans_l_mem(torch.cat((h_l_with_v2a, h_l_with_a2v), dim=2))
            #h_ls += h_ls_gmu
            #if type(h_ls) == tuple:
             #   h_ls = h_ls[0]
            #last_h_l = last_hs = h_ls[-1] #+ h_ls_gmu[-1]
            #------------
            '''
            # Residual conection
            #h_ls[:200,:,:] += h_l_gmu
            h_ls = self.trans_l_mem(h_ls_gmu)
            #h_ls += h_ls_gmu
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1] + h_ls_gmu[-1]  # Take the last output for prediction
            '''
            last_h_l = last_hs = h_ls[0]
            #last_h_l = last_hs = h_l_with_a2v[-1]

        if self.aonly:
            # (L,V) --> A
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
     #       attention_CLS = torch.cat((attention_CLS, h_v_with_ls[0], h_l_with_vs[0]), dim=1)
            
            # Feature Dimension Transformation
            t_h_l_with_vs = self.transfm_l2a(h_l_with_vs.permute(2, 1, 0)).permute(2, 1, 0)
            # GMU Middle --------
    #        h_a_gmu, z1_a = self.gmu_a_m(t_h_l_with_vs, h_v_with_ls)
            #h_as = self.trans_a_with_l2v(proj_x_a, h_a_gmu, h_a_gmu)
            # GMU Top --------
            h_a_with_v2l = self.trans_a_with_v2l(proj_x_a, h_l_with_vs, h_l_with_vs)
            h_a_with_l2v = self.trans_a_with_l2v(proj_x_a, h_v_with_ls, h_v_with_ls)
    #        attention_CLS = torch.cat((attention_CLS, h_a_with_l2v[0], h_a_with_v2l[0]), dim=1)
            
            # Residual conection
            h_a_with_v2l += t_h_l_with_vs
            h_a_with_l2v += h_v_with_ls
            # Option 1 ---------
            #h_as_gmu, z2_a = self.gmu_a(proj_x_a, h_a_with_v2l, h_a_with_l2v)
            h_as_gmu = self.gmu_a(proj_x_a, h_a_with_v2l, h_a_with_l2v)
            h_as = h_as_gmu #+ h_a_gmu
            '''
            mean1 = torch.mean(z1_a[-1][:, :self.d_a], dim=1, keepdim=False)
            mean2 = torch.mean(z1_a[-1][:, self.d_a:], dim=1, keepdim=False)
            alpha1_a = 1 - (mean1 - mean2)
            mean1 = torch.mean(z2_a[-1][:, :self.d_a], dim=1, keepdim=False)
            mean2 = torch.mean(z2_a[-1][:, self.d_a:], dim=1, keepdim=False)
            alpha2_a = 1 - (mean1 - mean2)
            h_as = alpha2_a*F.normalize(h_as_gmu.permute(0, 2, 1)) + alpha1_a*F.normalize(h_a_gmu.permute(0, 2, 1))
            h_as = h_as.permute(0, 2, 1)
            #'''
            # Option 2 ---------
            #sum_as = self.proj_l2v(h_a_with_l2v) + self.proj_v2l(h_a_with_v2l)
            #h_as = self.gmu_a(h_a_with_l2v, h_a_with_v2l, sum_as)
            # Option 3 ---------
            #h_as = self.proj_vl2a(torch.cat([h_a_with_l2v, h_a_with_v2l], dim=2))
            #print(h_as.size())
            # Normal Option --------
            #h_as = self.trans_a_mem(torch.cat((h_a_with_v2l, h_a_with_l2v), dim=2))
            #h_ls += h_ls_gmu
            #if type(h_as) == tuple:
             #   h_as = h_as[0]
            #last_h_a = last_hs = h_as[-1]
            #------------
            '''
            # Residual conection
            #h_as = h_a_gmu[:200,:,:] + h_as
            h_as = self.trans_a_mem(h_as_gmu)
            #h_as += h_as_gmu
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1] + h_as_gmu[-1]
            '''
            last_h_a = last_hs = h_as[0]
            #last_h_a = last_hs = h_a_with_v2l[0]
            #print(last_h_a.size())

        if self.vonly:
            # (L,A) --> V
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
    #        attention_CLS = torch.cat((attention_CLS, h_a_with_ls[0], h_l_with_as[0]), dim=1)
            
            # Feature Dimension Transformation
            t_h_l_with_as = self.transfm_l2v(h_l_with_as.permute(2, 1, 0)).permute(2, 1, 0)
            # GMU Middle --------
    #        h_v_gmu, z1_v = self.gmu_v_m(t_h_l_with_as, h_a_with_ls)
            #h_vs = self.trans_v_with_l2a(proj_x_v, h_v_gmu, h_v_gmu)
            # GMU Top --------
            h_v_with_a2l = self.trans_v_with_a2l(proj_x_v, h_l_with_as, h_l_with_as)
            h_v_with_l2a = self.trans_v_with_l2a(proj_x_v, h_a_with_ls, h_a_with_ls)
    #        attention_CLS = torch.cat((attention_CLS, h_v_with_l2a[0], h_v_with_a2l[0]), dim=1)
            
            # Residual conection
            h_v_with_a2l += t_h_l_with_as
            h_v_with_l2a += h_a_with_ls
            # Option 1 ---------
            #h_vs_gmu, z2_v = self.gmu_v(proj_x_v, h_v_with_a2l, h_v_with_l2a)
            h_vs_gmu = self.gmu_v(proj_x_v, h_v_with_a2l, h_v_with_l2a)
            h_vs = h_vs_gmu# + h_v_gmu
            '''
            mean1 = torch.mean(z1_v[-1][:, :self.d_v], dim=1, keepdim=False)
            mean2 = torch.mean(z1_v[-1][:, self.d_v:], dim=1, keepdim=False)
            alpha1_v = 1 - (mean1 - mean2)
            mean1 = torch.mean(z2_v[-1][:, :self.d_v], dim=1, keepdim=False)
            mean2 = torch.mean(z2_v[-1][:, self.d_v:], dim=1, keepdim=False)
            alpha2_v = 1 - (mean1 - mean2)
            h_vs = alpha2_v*F.normalize(h_vs_gmu.permute(0, 2, 1)) + alpha1_v*F.normalize(h_v_gmu.permute(0, 2, 1))
            h_vs = h_vs.permute(0, 2, 1)
            #'''
            # Option 2 ---------
            #sum_vs = self.proj_l2a(h_v_with_l2a) + self.proj_a2l(h_v_with_a2l)
            #h_vs = self.gmu_v(h_v_with_l2a, h_v_with_a2l, sum_vs)
            # Option 3 ---------
            #h_vs = self.proj_la2v(torch.cat([h_v_with_l2a, h_v_with_a2l], dim=2))
            #print(h_vs.size())
            # Normal Option --------
            #h_vs = self.trans_v_mem(torch.cat((h_v_with_a2l, h_v_with_l2a), dim=2))
            #if type(h_vs) == tuple:
             #   h_vs = h_vs[0]
            #last_h_v = last_hs = h_vs[-1]
            #------------
            '''
            # Residual conection
            #h_vs = h_v_gmu[:200,:,:] + h_vs
            h_vs = self.trans_v_mem(h_vs_gmu)
            h_vs += h_vs_gmu
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1] + h_vs_gmu[-1]
            '''
            last_h_v = last_hs = h_vs[0]
            #last_h_v = last_hs = h_v_gmu[0]
        
        #last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a)
        last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a, self.proj_poster(poster)) #, h_early.squeeze(0))
        #print(last_hs.size())
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        if output_gate:
            # Return attention from crossmodal transformers (token CLS)
            '''
            return self.out_layer(last_hs_proj), attention_CLS
            '''
            # Return GMU activation (Final GMU)
            #'''
            return self.out_layer(last_hs_proj), z
            #'''
            # Return GMU activation (Middle GMUs)
            '''
            which_token = 0 #-1
            return self.out_layer(last_hs_proj), torch.cat((z1_l[which_token], z2_l[which_token], z1_v[which_token], z2_v[which_token], z1_a[which_token], z2_a[which_token]), dim=1)
            '''
            
        else:
            return self.out_layer(last_hs_proj)

class TranslatingMMTransformerGMUClf_residual_v4_hybrid(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model for Text, Video frames and Audio spectrogram with GMU late fusion.
        """
        super(TranslatingMMTransformerGMUClf_residual_v4_hybrid, self).__init__()
        self.args = args
        self.orig_d_l, self.orig_d_v, self.orig_d_a, args.orig_d_m = args.orig_d_l, args.orig_d_v, args.orig_d_a, 312
        self.d_l, self.d_a, self.d_v, self.d_m = 768, 768, 768, 768
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
        
        self.enc = BertEncoder(args)
        self.audio_enc = AudioEncoder(args)

        # 0. Project poster feature to 768 dim
        self.proj_poster  = nn.Linear(self.orig_d_v, self.d_v, bias=False)
        # 0. Temporal Linear layers for Early fusion
        self.proj_l_e = nn.Linear(512, 16, bias=False)
        self.proj_v_e = nn.Linear(200, 16, bias=False)
        self.proj_a_e = nn.Linear(200, 16, bias=False)
        self.proj2_l_e = nn.Linear(self.orig_d_l, self.d_l, bias=False)
        self.proj2_v_e = nn.Linear(self.orig_d_v, self.d_v, bias=False)
        self.proj2_a_e = nn.Linear(self.orig_d_a, self.d_a, bias=False)
        #--- GMU instead of sum
        #------ GMU Middle
        self.gmu_l_m = GatedMultimodalLayerFeatures(self.d_v, self.d_a, self.d_l)
        self.gmu_v_m = GatedMultimodalLayerFeatures(self.d_l, self.d_a, self.d_v)
        self.gmu_a_m = GatedMultimodalLayerFeatures(self.d_l, self.d_v, self.d_a)
        #------ GMU Top
        self.gmu_l = GatedMultimodalLayerFeatures(self.d_v, self.d_a, self.d_l)
        self.gmu_v = GatedMultimodalLayerFeatures(self.d_l, self.d_a, self.d_v)
        self.gmu_a = GatedMultimodalLayerFeatures(self.d_l, self.d_v, self.d_a)
        #--------- GMU Top with sum
        #self.gmu_l = GatedMultimodal3LayerFeatures(self.d_l, self.d_l, self.d_l, self.d_l)
        #self.gmu_v = GatedMultimodal3LayerFeatures(self.d_v, self.d_v, self.d_v, self.d_l)
        #self.gmu_a = GatedMultimodal3LayerFeatures(self.d_a, self.d_a, self.d_a, self.d_l)
        #-------------
        #self.proj_va2l = nn.Linear(self.d_l * 2, self.d_l, bias=False)
        #self.proj_vl2a = nn.Linear(self.d_a * 2, self.d_a, bias=False)
        #self.proj_la2v = nn.Linear(self.d_v * 2, self.d_v, bias=False)

        combined_dim = 768 # For GMU
        
        output_dim = args.n_classes        # This is actually not a hyperparameter :-)

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
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        #self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        #self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
        #self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        #self.trans_h_mem = self.get_network(self_type='h_mem', layers=3)
        self.trans_l_early = self.get_network(self_type='l_mem', layers=3)
        self.trans_v_early = self.get_network(self_type='v_mem', layers=3)
        self.trans_a_early = self.get_network(self_type='a_mem', layers=3)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        #self.proj12 = nn.Linear(combined_dim, combined_dim)
        #self.proj22 = nn.Linear(combined_dim, combined_dim)
        #self.out_layer2 = nn.Linear(combined_dim, output_dim)
        #self.out_layer_final = nn.Linear(output_dim, output_dim)
        
        # GMU layer for fusing text and image and audio information
        #self.gmu = TextShifting4Layer(self.d_l*2, self.d_v*2, self.d_a*2, self.d_v, self.d_l)
        #self.gmu = TextShifting4Layer(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l)
        self.gmu = TextShifting5Layer(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l, self.d_l)
        #self.gmuSimple = TextShifting4LayerSimple(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l)
        self.gmu_early = GatedMultimodal3LayerFeatures(self.d_l, self.d_v, self.d_a, self.d_l)
        
        #Transformation dimension layers
        self.transfm_a2l = nn.Linear(200, 512)
        self.transfm_v2l = nn.Linear(200, 512)
        self.transfm_l2a = nn.Linear(512, 200)
        self.transfm_l2v = nn.Linear(512, 200)

    def get_network(self, self_type='l', layers=-1):
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
        elif self_type == 'h_mem':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
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
        
            
    def forward(self, txt, mask, segment, img, audio, poster, output_gate=False):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        #txt = torch.cat((txt, txt), 0)
        #mask = torch.cat((mask, mask), 0)
        #segment = torch.cat((segment, segment), 0)
        #img = torch.cat((img, img), 0)
        #audio = torch.cat((audio, audio), 0)
        #poster = torch.cat((poster, poster), 0)
        x_l = self.enc(txt, mask, segment)
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v = img.transpose(1, 2)
        x_a = self.audio_enc(audio)
        
        # Early Fusion
        if x_l.size(2) != 512:
            x1 = self.transfm_2dim(x_l, 2, 512)
        else: x1 = x_l
        if x_v.size(2) != 200:
            x2 = self.transfm_2dim(x_v, 2, 200)
        else: x2 = x_v
        if x_a.size(2) != 200:
            x3 = self.transfm_2dim(x_a, 2, 200)
        else: x3 = x_a
        proj_x_l_e = self.proj_l_e(x1)
        proj_x_v_e = self.proj_v_e(x2)
        proj_x_a_e = self.proj_a_e(x3)
        proj_x_l_e = self.proj2_l_e(proj_x_l_e.permute(2, 0, 1))
        proj_x_a_e = self.proj2_a_e(proj_x_a_e.permute(2, 0, 1))
        proj_x_v_e = self.proj2_v_e(proj_x_v_e.permute(2, 0, 1))
       # proj_x_l_e = self.proj_l_e(x_l.permute(2, 0, 1))
       # proj_x_a_e = self.proj_a_e(x_a.permute(2, 0, 1))
       # proj_x_v_e = self.proj_v_e(x_v.permute(2, 0, 1))
       # proj_x_p_e = self.proj2_poster(poster)
       # h_early, zx = self.gmuSimple(proj_x_l_e, proj_x_a_e, proj_x_v_e, proj_x_p_e)
        h_early, z_h = self.gmu_early(self.trans_l_early(proj_x_l_e),
                                      self.trans_v_early(proj_x_v_e),
                                      self.trans_a_early(proj_x_a_e))
        if type(h_early) == tuple:
            h_early = h_early[0]
        last_h_early = last_hs = h_early[-1]

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        
        if proj_x_l.size(0) != 512:
            proj_x_l = self.transfm_2dim(proj_x_l, 0, 512)
        if proj_x_a.size(0) != 200:
            proj_x_a = self.transfm_2dim(proj_x_a, 0, 200)
        if proj_x_v.size(0) != 200:
            proj_x_v = self.transfm_2dim(proj_x_v, 0, 200)

        if self.lonly:
            # (V,A) --> L
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)    # Dimension (L, N, d_v)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)    # Dimension (L, N, d_a)
            
            # Feature Dimension Transformation
            t_h_a_with_vs = self.transfm_a2l(h_a_with_vs.permute(2, 1, 0)).permute(2, 1, 0)
            t_h_v_with_as = self.transfm_v2l(h_v_with_as.permute(2, 1, 0)).permute(2, 1, 0)
            # GMU Middle --------
            h_l_gmu, z1_l = self.gmu_l_m(t_h_v_with_as, t_h_a_with_vs)
            #h_ls = self.trans_l_with_v2a(proj_x_l, h_l_gmu, h_l_gmu)
            # GMU Top ---------
            h_l_with_v2a = self.trans_l_with_v2a(proj_x_l, h_a_with_vs, h_a_with_vs)    # Dimension (L, N, d_l)
            h_l_with_a2v = self.trans_l_with_a2v(proj_x_l, h_v_with_as, h_v_with_as)    # Dimension (L, N, d_l)
            # Residual conection
            h_l_with_v2a += t_h_a_with_vs
            h_l_with_a2v += t_h_v_with_as
            # Option 1 ---------
            h_ls_gmu, z2_l = self.gmu_l(h_l_with_a2v, h_l_with_v2a)
            h_ls = h_ls_gmu + h_l_gmu
            '''
            mean1 = torch.mean(z1_l[-1][:, :self.d_l], dim=1, keepdim=False)
            mean2 = torch.mean(z1_l[-1][:, self.d_l:], dim=1, keepdim=False)
            alpha1_l = 1 - (mean1 - mean2)
            mean1 = torch.mean(z2_l[-1][:, :self.d_l], dim=1, keepdim=False)
            mean2 = torch.mean(z2_l[-1][:, self.d_l:], dim=1, keepdim=False)
            alpha2_l = 1 - (mean1 - mean2)
            #print(h_ls_gmu.size(), h_l_gmu.size())
            h_ls = alpha2_l*F.normalize(h_ls_gmu.permute(0, 2, 1)) + alpha1_l*F.normalize(h_l_gmu.permute(0, 2, 1))
            h_ls = h_ls.permute(0, 2, 1)
           # '''
            # Option 2 ---------
            #sum_ls = self.proj_v2a(h_l_with_v2a) + self.proj_a2v(h_l_with_a2v)
            #h_ls = self.gmu_l(h_l_with_v2a, h_l_with_a2v, sum_ls)
            # Option 3 ---------
            #h_ls = self.proj_va2l(torch.cat([h_l_with_v2a, h_l_with_a2v], dim=2))
            #print(h_ls.size(), h_l_gmu.size())
            # Normal Option --------
            #h_ls = self.trans_l_mem(torch.cat((h_l_with_v2a, h_l_with_a2v), dim=2))
            #h_ls += h_ls_gmu
            #if type(h_ls) == tuple:
             #   h_ls = h_ls[0]
            #last_h_l = last_hs = h_ls[-1] #+ h_ls_gmu[-1]
            #------------
            '''
            # Residual conection
            #h_ls[:200,:,:] += h_l_gmu
            h_ls = self.trans_l_mem(h_ls_gmu)
            #h_ls += h_ls_gmu
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1] + h_ls_gmu[-1]  # Take the last output for prediction
            '''
            last_h_l = last_hs = h_ls[-1]
            #last_h_l = last_hs = h_l_with_a2v[-1]

        if self.aonly:
            # (L,V) --> A
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
            
            # Feature Dimension Transformation
            t_h_l_with_vs = self.transfm_l2a(h_l_with_vs.permute(2, 1, 0)).permute(2, 1, 0)
            # GMU Middle --------
            h_a_gmu, z1_a = self.gmu_a_m(t_h_l_with_vs, h_v_with_ls)
            #h_as = self.trans_a_with_l2v(proj_x_a, h_a_gmu, h_a_gmu)
            # GMU Top --------
            h_a_with_v2l = self.trans_a_with_v2l(proj_x_a, h_l_with_vs, h_l_with_vs)
            h_a_with_l2v = self.trans_a_with_l2v(proj_x_a, h_v_with_ls, h_v_with_ls)
            # Residual conection
            h_a_with_v2l += t_h_l_with_vs
            h_a_with_l2v += h_v_with_ls
            # Option 1 ---------
            h_as_gmu, z2_a = self.gmu_a(h_a_with_v2l, h_a_with_l2v)
            h_as = h_as_gmu + h_a_gmu
            '''
            mean1 = torch.mean(z1_a[-1][:, :self.d_a], dim=1, keepdim=False)
            mean2 = torch.mean(z1_a[-1][:, self.d_a:], dim=1, keepdim=False)
            alpha1_a = 1 - (mean1 - mean2)
            mean1 = torch.mean(z2_a[-1][:, :self.d_a], dim=1, keepdim=False)
            mean2 = torch.mean(z2_a[-1][:, self.d_a:], dim=1, keepdim=False)
            alpha2_a = 1 - (mean1 - mean2)
            h_as = alpha2_a*F.normalize(h_as_gmu.permute(0, 2, 1)) + alpha1_a*F.normalize(h_a_gmu.permute(0, 2, 1))
            h_as = h_as.permute(0, 2, 1)
            #'''
            # Option 2 ---------
            #sum_as = self.proj_l2v(h_a_with_l2v) + self.proj_v2l(h_a_with_v2l)
            #h_as = self.gmu_a(h_a_with_l2v, h_a_with_v2l, sum_as)
            # Option 3 ---------
            #h_as = self.proj_vl2a(torch.cat([h_a_with_l2v, h_a_with_v2l], dim=2))
            #print(h_as.size())
            # Normal Option --------
            #h_as = self.trans_a_mem(torch.cat((h_a_with_v2l, h_a_with_l2v), dim=2))
            #h_ls += h_ls_gmu
            #if type(h_as) == tuple:
             #   h_as = h_as[0]
            #last_h_a = last_hs = h_as[-1]
            #------------
            '''
            # Residual conection
            #h_as = h_a_gmu[:200,:,:] + h_as
            h_as = self.trans_a_mem(h_as_gmu)
            #h_as += h_as_gmu
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1] + h_as_gmu[-1]
            '''
            last_h_a = last_hs = h_as[-1]
            #last_h_a = last_hs = h_a_with_v2l[-1]
            #print(last_h_a.size())

        if self.vonly:
            # (L,A) --> V
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
            
            # Feature Dimension Transformation
            t_h_l_with_as = self.transfm_l2v(h_l_with_as.permute(2, 1, 0)).permute(2, 1, 0)
            # GMU Middle --------
            h_v_gmu, z1_v = self.gmu_v_m(t_h_l_with_as, h_a_with_ls)
            #h_vs = self.trans_v_with_l2a(proj_x_v, h_v_gmu, h_v_gmu)
            # GMU Top --------
            h_v_with_a2l = self.trans_v_with_a2l(proj_x_v, h_l_with_as, h_l_with_as)
            h_v_with_l2a = self.trans_v_with_l2a(proj_x_v, h_a_with_ls, h_a_with_ls)
            # Residual conection
            h_v_with_a2l += t_h_l_with_as
            h_v_with_l2a += h_a_with_ls
            # Option 1 ---------
            h_vs_gmu, z2_v = self.gmu_v(h_v_with_a2l, h_v_with_l2a)
            h_vs = h_vs_gmu + h_v_gmu
            '''
            mean1 = torch.mean(z1_v[-1][:, :self.d_v], dim=1, keepdim=False)
            mean2 = torch.mean(z1_v[-1][:, self.d_v:], dim=1, keepdim=False)
            alpha1_v = 1 - (mean1 - mean2)
            mean1 = torch.mean(z2_v[-1][:, :self.d_v], dim=1, keepdim=False)
            mean2 = torch.mean(z2_v[-1][:, self.d_v:], dim=1, keepdim=False)
            alpha2_v = 1 - (mean1 - mean2)
            h_vs = alpha2_v*F.normalize(h_vs_gmu.permute(0, 2, 1)) + alpha1_v*F.normalize(h_v_gmu.permute(0, 2, 1))
            h_vs = h_vs.permute(0, 2, 1)
            #'''
            # Option 2 ---------
            #sum_vs = self.proj_l2a(h_v_with_l2a) + self.proj_a2l(h_v_with_a2l)
            #h_vs = self.gmu_v(h_v_with_l2a, h_v_with_a2l, sum_vs)
            # Option 3 ---------
            #h_vs = self.proj_la2v(torch.cat([h_v_with_l2a, h_v_with_a2l], dim=2))
            #print(h_vs.size())
            # Normal Option --------
            #h_vs = self.trans_v_mem(torch.cat((h_v_with_a2l, h_v_with_l2a), dim=2))
            #if type(h_vs) == tuple:
             #   h_vs = h_vs[0]
            #last_h_v = last_hs = h_vs[-1]
            #------------
            '''
            # Residual conection
            #h_vs = h_v_gmu[:200,:,:] + h_vs
            h_vs = self.trans_v_mem(h_vs_gmu)
            h_vs += h_vs_gmu
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1] + h_vs_gmu[-1]
            '''
            last_h_v = last_hs = h_vs[-1]
            #last_h_v = last_hs = h_v_with_l2a[-1]
        
        #last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a)
        last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a, self.proj_poster(poster), last_h_early)
        #print(last_hs.size())
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        if output_gate:
            return self.out_layer(last_hs_proj), z#torch.cat((z1_l[-1], z2_l[-1],
                                                   #         z1_v[-1], z2_v[-1],
                                                    #        z1_a[-1], z2_a[-1]), dim=1)
        else:
            return self.out_layer(last_hs_proj)
            
# V2 attention, see: Notability
class MultiprojectionMMTransformerGMUClf_V2(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model with a Transformer preprocessing for video and audio
        """
        super(MultiprojectionMMTransformerGMUClf_V2, self).__init__()
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
        
        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
            self.trans_l_with_v2a = self.get_network(self_type='lv2a')#, biprojection=True)
            self.trans_l_with_a2v = self.get_network(self_type='la2v')#, biprojection=True)
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')
            self.trans_v_with_l2a = self.get_network(self_type='vl2a')#, biprojection=True)
            self.trans_v_with_a2l = self.get_network(self_type='va2l')#, biprojection=True)
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
            self.trans_a_with_v2l = self.get_network(self_type='av2l')#, biprojection=True)
            self.trans_a_with_l2v = self.get_network(self_type='al2v')#, biprojection=True)
        
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        
        # GMU layer for fusing text and image and audio information
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
   #     self.transfm_a2l = nn.Linear(self.num_vectors_a, self.num_vectors_l)
    #    self.transfm_v2l = nn.Linear(self.num_vectors_v, self.num_vectors_l)
     #   self.transfm_l2a = nn.Linear(self.num_vectors_l, self.num_vectors_a)
      #  self.transfm_l2v = nn.Linear(self.num_vectors_l, self.num_vectors_v)
     #   self.transfm_v2a = nn.Linear(self.num_vectors_v, self.num_vectors_a)
      #  self.transfm_a2v = nn.Linear(self.num_vectors_a, self.num_vectors_v)
        
        if self.hybrid:
            self.trans_l_early = self.get_network(self_type='l_mem', layers=3)
            self.trans_v_early = self.get_network(self_type='v_mem', layers=3)
            self.trans_a_early = self.get_network(self_type='a_mem', layers=3)
            #self.proj_l_e = nn.Linear(self.num_vectors_l, 32, bias=False)
            #self.proj_v_e = nn.Linear(self.num_vectors_v, 32, bias=False)
            #self.proj_a_e = nn.Linear(self.num_vectors_a, 32, bias=False)
        
        # Transformer self-attention for preprosessing
        self.trans_v = self.get_network(self_type='v_mem')
        self.trans_a = self.get_network(self_type='a_mem')

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
        
        '''
        if proj_x_l.size(0) != self.num_vectors_l:
            proj_x_l = self.transfm_2dim(proj_x_l, 0, self.num_vectors_l)
        if proj_x_a.size(0) != self.num_vectors_a:
            proj_x_a = self.transfm_2dim(proj_x_a, 0, self.num_vectors_a)
        if proj_x_v.size(0) != self.num_vectors_v:
            proj_x_v = self.transfm_2dim(proj_x_v, 0, self.num_vectors_v)
        '''
        
        #Parallel fusion
        if self.hybrid:
            proj_x_l_e = proj_x_l#self.proj_l_e(proj_x_l.permute(2, 1, 0)).permute(2, 1, 0)
            proj_x_a_e = proj_x_a#self.proj_a_e(proj_x_a.permute(2, 1, 0)).permute(2, 1, 0)
            proj_x_v_e = proj_x_v#self.proj_v_e(proj_x_v.permute(2, 1, 0)).permute(2, 1, 0)
            h_l = self.trans_l_early(proj_x_l_e)
            h_a = self.trans_a_early(proj_x_a_e)
            h_v = self.trans_v_early(proj_x_v_e)
            last_hl_early = h_l[0] + h_l[-1]#, zx = self.gmu_early([h_l, h_v, h_a])
            last_ha_early = h_a[0] + h_a[-1]
            last_hv_early = h_v[0] + h_v[-1]
            last_h_early, zx = self.gmu_early(last_hl_early, last_hv_early, last_ha_early)
            
        poster   = self.proj_poster(poster)

        # First Crossmodal Transformers
        # (V,A) --> L
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)    # Dimension (V, N, d_v)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)    # Dimension (A, N, d_a)
        # (L,V) --> A
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
        # (L,A) --> V
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
        
        if self.lonly:
            # Biprojection
            # Since x_l is BERT based we do not use a transformer to preprocess
            h_l_with_v2a = self.trans_l_with_v2a(proj_x_l[0].unsqueeze(0), h_a_with_vs, h_a_with_vs)
            h_l_with_a2v = self.trans_l_with_a2v(proj_x_l[0].unsqueeze(0), h_v_with_as, h_v_with_as) # Dimension (L, N, d_l)
            
            # GMU Middle --------------
           # t_h_a_with_vs =  self.transfm_a2l(h_a_with_vs.permute(2,1,0)).permute(2,1,0)#h_a_with_vs[tok_a]
           # t_h_v_with_as =  self.transfm_v2l(h_v_with_as.permute(2,1,0)).permute(2,1,0)#[tok_v])
   #         h_l_gmu, z1_l = self.gmu_l_m([h_l_with_vs, h_l_with_as]) #[t_h_v_with_as, t_h_a_with_vs])
            
            # Residual conection level 1 to 2 -------
            h_l_with_v2a_tot = h_l_with_v2a# + h_l_with_as#t_h_a_with_vs
            h_l_with_a2v_tot = h_l_with_a2v# + h_l_with_vs#t_h_v_with_as
            # Feature Fusion ---------
            h_ls_gmu, z2_l = self.gmu_l([h_l_with_a2v_tot, h_l_with_v2a_tot])
            # Residual conection level 1 to 3 -------
    #        h_ls_gmu += h_l_gmu

            last_h_l = last_hs = h_ls_gmu[0]# + h_ls_gmu[-1]

        if self.aonly:
            # Transformer preprocess
            proj_x_a_trans = self.trans_a(proj_x_a)[-1].unsqueeze(0)#, proj_x_a, proj_x_a)
            # Biprojection ---------
            h_a_with_v2l = self.trans_a_with_v2l(proj_x_a_trans, h_l_with_vs, h_l_with_vs)
            h_a_with_l2v = self.trans_a_with_l2v(proj_x_a_trans, h_v_with_ls, h_v_with_ls)
            # GMU Middle --------
            
        #    t_h_l_with_vs = self.transfm_l2a(h_l_with_vs.permute(2,1,0)).permute(2,1,0)#[tok_l])
         #   t_h_v_with_ls = h_v_with_ls #self.transfm_v2a(h_v_with_ls.permute(2,1,0)).permute(2,1,0)#[tok_v])
  #          h_a_gmu, z1_a = self.gmu_a_m([h_a_with_ls, h_a_with_vs])#t_h_l_with_vs, t_h_v_with_ls])

            # Residual conection level 1 to 2 -------
            h_a_with_v2l_tot = h_a_with_v2l# + h_a_with_ls#t_h_l_with_vs
            h_a_with_l2v_tot = h_a_with_l2v# + h_a_with_vs#t_h_v_with_ls
            # Feature Fusion ---------
            h_as_gmu, z2_a = self.gmu_a([h_a_with_v2l_tot, h_a_with_l2v_tot])
            # Residual conection level 1 to 3 -------
   #         h_as_gmu += h_a_gmu
            last_h_a = last_hs = h_as_gmu[0]# + h_as_gmu[-1]

        if self.vonly:
            # Transformer preprocess
            proj_x_v_trans = self.trans_v(proj_x_v)[-1].unsqueeze(0)#, proj_x_v, proj_x_v)
            # Biprojection ---------
            h_v_with_a2l = self.trans_v_with_a2l(proj_x_v_trans, h_l_with_as, h_l_with_as)
            h_v_with_l2a = self.trans_v_with_l2a(proj_x_v_trans, h_a_with_ls, h_a_with_ls)
            # GMU Middle --------
            
          #  t_h_l_with_as = self.transfm_l2v(h_l_with_as.permute(2,1,0)).permute(2,1,0)#[tok_l])
           # t_h_a_with_ls = h_a_with_ls #self.transfm_a2v(h_a_with_ls.permute(2,1,0)).permute(2,1,0)#[tok_a])
  #          h_v_gmu, z1_v = self.gmu_v_m([h_v_with_ls, h_v_with_as])#t_h_l_with_as, t_h_a_with_ls])
            
            # Residual conection level 1 to 2 -------
            h_v_with_a2l_tot = h_v_with_a2l# + h_v_with_ls#t_h_l_with_as
            h_v_with_l2a_tot = h_v_with_l2a# + h_v_with_as#t_h_a_with_ls
            # Feature Fusion ---------
            h_vs_gmu, z2_v = self.gmu_v([h_v_with_a2l_tot, h_v_with_l2a_tot])
            # Residual conection level 1 to 3 -------
   #         h_vs_gmu += h_v_gmu
            last_h_v = last_hs = h_vs_gmu[0]# + h_vs_gmu[-1]
        
        if self.hybrid:
            last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a, poster, last_h_early)#last_hl_early, last_hv_early, last_ha_early)
        else:
            last_hs, z = self.gmu([last_h_l, last_h_v, last_h_a, poster])
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        if output_gate:
            return self.out_layer(last_hs_proj), z
            
        else:
            return self.out_layer(last_hs_proj)
        
# el chingon
class MultiprojectionMMTransformerGMUClf_V1(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model with a Transformer preprocessing for video and audio
        """
        super(MultiprojectionMMTransformerGMUClf_V1, self).__init__()
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
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        #self.trans_p_ppro = self.get_network(self_type='p_mem', layers=3)
 #       self.trans_v_ppro = self.get_network(self_type='v_mem', layers=3)
  #      self.trans_a_ppro = self.get_network(self_type='a_mem', layers=3)
   #     self.CLS_v = nn.Linear(combined_dim, combined_dim)
    #    self.CLS_a = nn.Linear(combined_dim, combined_dim)
        #MAG
        #beta_shift, dropout_prob = 1e-3, 0.5
        #self.MAG_v = MAG(self.d_v, beta_shift, dropout_prob)
        #self.MAG_a = MAG(self.d_a, beta_shift, dropout_prob)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        #self.proj12 = nn.Linear(combined_dim, combined_dim)
        #self.proj22 = nn.Linear(combined_dim, combined_dim)
        #self.out_layer2 = nn.Linear(combined_dim, output_dim)
        #self.out_layer_final = nn.Linear(output_dim, output_dim)
        
        # GMU layer for fusing text and image and audio information
        #self.gmu = TextShifting4Layer(self.d_l*2, self.d_v*2, self.d_a*2, self.d_v, self.d_l)
 #       self.gmu = TextShifting4Layer(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l)
    #self.gmu = TextShiftingNLayer([combined_dim]*7, combined_dim)
        if self.hybrid:
            self.gmu = TextShiftingNLayer([combined_dim]*5, combined_dim)
        else:
            #self.gmu = GatedMultimodalUnitSoftmaxFusion(4, combined_dim, combined_dim)
            self.gmu = TextShifting4Layer(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l)
        #self.gmu = TextShifting5Layer(self.d_l*2, self.d_v*2, self.d_v*2, self.d_v, self.d_l, self.d_l)
        #self.gmuSimple = TextShifting4LayerSimple(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l)
        #self.gmuSimple = TextShifting3LayerSimple(self.d_l, self.d_v, self.d_a, self.d_l)
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
            
        # Trasformer preprocessing and prepend CLS Trainable Tokens
        '''
        ones_v = torch.ones(1, proj_x_v.size(1), proj_x_v.size(2)).cuda()
        ones_a = torch.ones(1, proj_x_a.size(1), proj_x_a.size(2)).cuda()
        
        proj_x_a = self.trans_a_ppro(torch.cat((proj_x_l[0].unsqueeze(0), proj_x_a), dim=0))
        proj_x_v = self.trans_v_ppro(torch.cat((proj_x_l[0].unsqueeze(0), proj_x_v), dim=0))
    #
        cls_a = self.CLS_a(proj_x_a)#.permute(2,1,0)).permute(2,1,0)
        cls_v = self.CLS_v(proj_x_v)#.permute(2,1,0)).permute(2,1,0)
    #
        cls_a = self.CLS_a(ones_a)
        cls_v = self.CLS_v(ones_v)
        
        proj_x_a = self.trans_a_ppro(torch.cat((cls_a[0].unsqueeze(0), proj_x_a), dim=0))
        proj_x_v = self.trans_v_ppro(torch.cat((cls_v[0].unsqueeze(0), proj_x_v), dim=0))
        '''
        
#Comment following lines to IMDb
        poster   = self.proj_poster(poster)
        #poster   = self.proj_poster(poster).squeeze(1)#.transpose(1, 0) #self.trans_p_ppro(torch.cat((proj_x_l[0].unsqueeze(0), self.proj_poster(poster).unsqueeze(0)), dim=0))[0]

        # First Crossmodal Transformers
        # (V,A) --> L
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)    # Dimension (V, N, d_v)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)    # Dimension (A, N, d_a)
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
            
            t_h_a_with_vs =  self.transfm_a2l(h_a_with_vs.permute(2,1,0)).permute(2,1,0)#h_a_with_vs[tok_a]
            t_h_v_with_as =  self.transfm_v2l(h_v_with_as.permute(2,1,0)).permute(2,1,0)#[tok_v])
            h_l_gmu, z1_l = self.gmu_l_m([t_h_v_with_as, t_h_a_with_vs])
            '''
            #t_h_a_with_vs = h_a_with_vs[0] + h_a_with_vs[-1]
            #t_h_v_with_as = h_v_with_as[0]  + h_v_with_as[-1]
            h_l_gmu, z1_l = self.gmu_l_m([h_a_with_vs[0], h_a_with_vs[-1], h_v_with_as[0], h_v_with_as[-1]])
            '''
            # Residual conection level 1 to 2 -------
            h_l_with_v2a_tot = h_l_with_v2a + t_h_a_with_vs #h_l_with_v2a[0] + h_l_with_v2a[-1] + t_h_a_with_vs[0] + t_h_a_with_vs[-1]
            #h_l_with_v2a_tot2 = h_l_with_v2a[-1] + h_a_with_vs[-1]
            h_l_with_a2v_tot = h_l_with_a2v + t_h_v_with_as #h_l_with_a2v[0] + h_l_with_a2v[-1] + t_h_v_with_as[0] + t_h_v_with_as[-1]
            #h_l_with_a2v_tot2 = h_l_with_a2v[-1] + h_v_with_as[-1]
            
            # Feature Fusion ---------
            h_ls_gmu, z2_l = self.gmu_l([h_l_with_a2v_tot, h_l_with_v2a_tot])
            #h_ls_gmu, z2_l = self.gmu_l([h_l_with_v2a_tot1, h_l_with_v2a_tot2, h_l_with_a2v_tot1, h_l_with_a2v_tot2])
            
            # Residual conection level 1 to 3 -------
            h_ls_gmu += h_l_gmu
            #h_ls_gmu[-1] += h_l_gmu[-1]

            last_h_l = last_hs = h_ls_gmu[0] + h_ls_gmu[-1]

        if self.aonly:
            # Biprojection ---------
            h_a_with_v2l = self.trans_a_with_v2l(proj_x_a, h_l_with_vs, h_l_with_vs)
            h_a_with_l2v = self.trans_a_with_l2v(proj_x_a, h_v_with_ls, h_v_with_ls)
            # GMU Middle --------
            
            t_h_l_with_vs = self.transfm_l2a(h_l_with_vs.permute(2,1,0)).permute(2,1,0)#[tok_l])
            t_h_v_with_ls = h_v_with_ls #self.transfm_v2a(h_v_with_ls.permute(2,1,0)).permute(2,1,0)#[tok_v])
            h_a_gmu, z1_a = self.gmu_a_m([t_h_l_with_vs, t_h_v_with_ls])
            '''
            #t_h_l_with_vs = h_l_with_vs[0] + h_l_with_vs[-1]
            #t_h_v_with_ls = h_v_with_ls[0] + h_v_with_ls[-1]
            h_a_gmu, z1_a = self.gmu_a_m([h_l_with_vs[0], h_l_with_vs[-1], h_v_with_ls[0], h_v_with_ls[-1]])
            '''
            # Residual conection level 1 to 2 -------
            h_a_with_v2l_tot = h_a_with_v2l + t_h_l_with_vs #h_a_with_v2l[0] + h_a_with_v2l[-1] + t_h_l_with_vs #[tok_a] += t_h_l_with_vs
           # h_a_with_v2l_tot2 = h_a_with_v2l[-1] + h_l_with_vs[-1]
            h_a_with_l2v_tot = h_a_with_l2v + t_h_v_with_ls#h_a_with_l2v[0] + h_a_with_l2v[-1] + t_h_v_with_ls #[tok_a] += t_h_v_with_ls
         #   h_a_with_l2v_tot2 = h_a_with_l2v[-1] + h_v_with_ls[-1]
            # Feature Fusion ---------
            h_as_gmu, z2_a = self.gmu_a([h_a_with_v2l_tot, h_a_with_l2v_tot])
            # Residual conection level 1 to 3 -------
            h_as_gmu += h_a_gmu
            #h_as_gmu[-1] += h_a_gmu[-1]
            
            last_h_a = last_hs = h_as_gmu[0] + h_as_gmu[-1]

        if self.vonly:
            # Biprojection ---------
            h_v_with_a2l = self.trans_v_with_a2l(proj_x_v, h_l_with_as, h_l_with_as)
            h_v_with_l2a = self.trans_v_with_l2a(proj_x_v, h_a_with_ls, h_a_with_ls)
            # GMU Middle --------
            
            t_h_l_with_as = self.transfm_l2v(h_l_with_as.permute(2,1,0)).permute(2,1,0)#[tok_l])
            t_h_a_with_ls = h_a_with_ls #self.transfm_a2v(h_a_with_ls.permute(2,1,0)).permute(2,1,0)#[tok_a])
            h_v_gmu, z1_v = self.gmu_v_m([t_h_l_with_as, t_h_a_with_ls])
            '''
        #    t_h_l_with_as = h_l_with_as[0] + h_l_with_as[-1]
         #   t_h_a_with_ls = h_a_with_ls[0] + h_a_with_ls[-1]
            h_v_gmu, z1_v = self.gmu_v_m([h_l_with_as[0], h_l_with_as[-1], h_a_with_ls[0],h_a_with_ls[-1]])
            '''
            # Residual conection level 1 to 2 -------
            h_v_with_a2l_tot = h_v_with_a2l + t_h_l_with_as #h_v_with_a2l[0] + h_v_with_a2l[-1] + t_h_l_with_as #[tok_v] += t_h_l_with_as
        #    h_v_with_a2l_tot2 = h_v_with_a2l[-1] + h_l_with_as[-1]
            h_v_with_l2a_tot = h_v_with_l2a + t_h_a_with_ls #h_v_with_l2a[0] + h_v_with_l2a[-1] + t_h_a_with_ls #[tok_v] += t_h_a_with_ls
         #   h_v_with_l2a_tot2 = h_v_with_l2a[-1] + h_a_with_ls[-1]
            # Feature Fusion ---------
            h_vs_gmu, z2_v = self.gmu_v([h_v_with_a2l_tot, h_v_with_l2a_tot])
            # Residual conection level 1 to 3 -------
            h_vs_gmu += h_v_gmu
           # h_vs_gmu[-1] += h_v_gmu[-1]
            
            last_h_v = last_hs = h_vs_gmu[0] + h_vs_gmu[-1]
        
        #last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a)
 #       last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a, poster) #, h_early.squeeze(0))
        #last_hs, z = self.gmu(last_h_l1, last_h_l2, last_h_v1, last_h_v2, last_h_a1, last_h_a2, h_v_with_as[tok], h_a_with_vs[tok], h_v_with_ls[tok], h_l_with_vs[tok], h_a_with_ls[tok], h_l_with_as[tok], poster)
        if self.hybrid:
            last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a, poster, last_h_early)#last_hl_early, last_hv_early, last_ha_early)
        else:
            #print(last_h_l.size(), last_h_v.size(), last_h_a.size(), poster.size())
            last_hs, z = self.gmu([last_h_l, last_h_v, last_h_a, poster])
       # last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a, last_h_v2, last_h_a1, last_h_a2, poster)
        #print(last_hs.size())
        #last_hs = last_hs.squeeze(0)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        if output_gate:
            # Return attention from crossmodal transformers (token CLS)
            '''
            return self.out_layer(last_hs_proj), attention_CLS
            '''
            # Return GMU activation (Final GMU)
            #'''
            return self.out_layer(last_hs_proj), z
            #'''
            # Return GMU activation (Middle GMUs)
            '''
            which_token = 0 #-1
            return self.out_layer(last_hs_proj), torch.cat((z1_l[which_token], z2_l[which_token], z1_v[which_token], z2_v[which_token], z1_a[which_token], z2_a[which_token]), dim=1)
            '''
            
        else:
            return self.out_layer(last_hs_proj)

#El chingon pero pa experimentar
class MultiprojectionMMTransformerGMUClf(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model with a Transformer preprocessing for video and audio
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
        self.reduced_dim = 150
        
        self.enc = BertEncoder(args)
#Comment following line to IMDb
        self.audio_enc = AudioEncoder(args)

        # 0. Project poster feature to 768 dim
        self.proj_poster  = nn.Linear(self.orig_d_p, self.d_v, bias=False)
        
        #--- GMU instead of sum
        #------ GMU Middle
#        self.gmu_l_m = GatedMultimodalLayerFeatures(self.d_v, self.d_a, self.d_l)
        #self.gmu_l_m = TextShifting4LayerFeatures(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l)
        #self.gmu_l_m = GatedMultimodalUnitSoftmaxFusion(2, combined_dim, combined_dim, probs=False, residual=False)
#        self.gmu_v_m = GatedMultimodalLayerFeatures(self.d_l, self.d_a, self.d_v)
        #self.gmu_v_m = TextShifting4LayerFeatures(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l)
        #self.gmu_v_m = GatedMultimodalUnitSoftmaxFusion(2, combined_dim, combined_dim, probs=False, residual=False)
#        self.gmu_a_m = GatedMultimodalLayerFeatures(self.d_l, self.d_v, self.d_a)
       # self.gmu_a_m = TextShifting4LayerFeatures(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l)
        #self.gmu_a_m = GatedMultimodalUnitSoftmaxFusion(2, combined_dim, combined_dim, probs=False, residual=False)
        #------ GMU Top
#        self.gmu_l = GatedMultimodalLayerFeatures(self.d_v, self.d_a, self.d_l)
        self.gmu_l = TextShifting4LayerFeatures(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l)
        #self.gmu_l = GatedMultimodalUnitSoftmaxFusion(2, combined_dim, combined_dim, probs=False, residual=False)
#        self.gmu_v = GatedMultimodalLayerFeatures(self.d_l, self.d_a, self.d_v)
        self.gmu_v = TextShifting4LayerFeatures(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l)
        #self.gmu_v = GatedMultimodalUnitSoftmaxFusion(2, combined_dim, combined_dim, probs=False, residual=False)
#        self.gmu_a = GatedMultimodalLayerFeatures(self.d_l, self.d_v, self.d_a)
        self.gmu_a = TextShifting4LayerFeatures(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l)
        #self.gmu_a = GatedMultimodalUnitSoftmaxFusion(2, combined_dim, combined_dim, probs=False, residual=False)
        
        if self.hybrid:
            #self.gmu_early = GatedMultimodalUnitSoftmaxFusion(3, combined_dim, combined_dim, residual=False)
            #self.gmu_early = TextShifting3Layer(self.d_l, self.d_v, self.d_a, combined_dim)
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
            #self.proj_l_e = nn.Linear(self.num_vectors_l, self.reduced_dim, bias=False)
            #self.proj_v_e = nn.Linear(self.num_vectors_v, self.reduced_dim, bias=False)
            #self.proj_a_e = nn.Linear(self.num_vectors_a, self.reduced_dim, bias=False)
            self.proj_l_e = nn.Linear(combined_dim, self.reduced_dim, bias=False)
            self.proj_v_e = nn.Linear(combined_dim, self.reduced_dim, bias=False)
            self.proj_a_e = nn.Linear(combined_dim, self.reduced_dim, bias=False)

    def get_network(self, self_type='l', layers=-1, biprojection=False):
        if self_type in ['l', 'al', 'vl', 'av2l', 'va2l']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va', 'lv2a', 'vl2a']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av', 'la2v', 'al2v']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
        # With GMU
            embed_dim, attn_dropout = self.reduced_dim, self.attn_dropout
        # Without GMU (normal concat)
            #embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.reduced_dim, self.attn_dropout
            #embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = self.reduced_dim, self.attn_dropout
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
            proj_x_l_e = self.proj_l_e(proj_x_l)#.permute(2, 1, 0)).permute(2, 1, 0)
            proj_x_a_e = self.proj_a_e(proj_x_a)#.permute(2, 1, 0)).permute(2, 1, 0)
            proj_x_v_e = self.proj_v_e(proj_x_v)#.permute(2, 1, 0)).permute(2, 1, 0)
            h_l = self.trans_l_early(proj_x_l_e)
            h_a = self.trans_a_early(proj_x_a_e)
            h_v = self.trans_v_early(proj_x_v_e)
            last_hl_early = h_l[0] + h_l[-1]#, zx = self.gmu_early([h_l, h_v, h_a])
            last_ha_early = h_a[0] + h_a[-1]
            last_hv_early = h_v[0] + h_v[-1]
            last_h_early, zx = self.gmu_early(last_hl_early, last_hv_early, last_ha_early)
        
#Comment following lines to IMDb
        poster   = self.proj_poster(poster)

        # First Crossmodal Transformers
        # (V,A) --> L
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)    # Dimension (V, N, d_v)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)    # Dimension (A, N, d_a)
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
#            t_h_a_with_vs =  self.transfm_a2l(h_a_with_vs.permute(2,1,0)).permute(2,1,0)#h_a_with_vs[tok_a]
#            t_h_v_with_as =  self.transfm_v2l(h_v_with_as.permute(2,1,0)).permute(2,1,0)#[tok_v])
#            h_l_gmu, z1_l = self.gmu_l_m([h_l_with_vs, h_l_with_as])
            '''
            #t_h_a_with_vs = h_a_with_vs[0] + h_a_with_vs[-1]
            #t_h_v_with_as = h_v_with_as[0]  + h_v_with_as[-1]
            h_l_gmu, z1_l = self.gmu_l_m([h_a_with_vs[0], h_a_with_vs[-1], h_v_with_as[0], h_v_with_as[-1]])
            '''
            # Residual conection level 1 to 2 -------
#            h_l_with_v2a_tot = h_l_with_v2a + t_h_a_with_vs #h_l_with_v2a[0] + h_l_with_v2a[-1] + t_h_a_with_vs[0] + t_h_a_with_vs[-1]
            #h_l_with_v2a_tot2 = h_l_with_v2a[-1] + h_a_with_vs[-1]
#            h_l_with_a2v_tot = h_l_with_a2v + t_h_v_with_as #h_l_with_a2v[0] + h_l_with_a2v[-1] + t_h_v_with_as[0] + t_h_v_with_as[-1]
            #h_l_with_a2v_tot2 = h_l_with_a2v[-1] + h_v_with_as[-1]
            
            # Feature Fusion ---------
#            h_ls_gmu, z2_l = self.gmu_l([h_l_with_a2v_tot, h_l_with_v2a_tot])
            h_ls_gmu, z2_l = self.gmu_l([h_l_with_a2v, h_l_with_v2a, h_l_with_vs, h_l_with_as])
            
            # Residual conection level 1 to 3 -------
#            h_ls_gmu += h_l_gmu
            #h_ls_gmu[-1] += h_l_gmu[-1]

            last_h_l = last_hs = h_ls_gmu[0] + h_ls_gmu[-1]

        if self.aonly:
            # Biprojection ---------
            h_a_with_v2l = self.trans_a_with_v2l(proj_x_a, h_l_with_vs, h_l_with_vs)
            h_a_with_l2v = self.trans_a_with_l2v(proj_x_a, h_v_with_ls, h_v_with_ls)
            # GMU Middle --------
            
#            t_h_l_with_vs = self.transfm_l2a(h_l_with_vs.permute(2,1,0)).permute(2,1,0)#[tok_l])
#            t_h_v_with_ls = h_v_with_ls #self.transfm_v2a(h_v_with_ls.permute(2,1,0)).permute(2,1,0)#[tok_v])
#            h_a_gmu, z1_a = self.gmu_a_m([h_a_with_ls, h_a_with_vs])
            '''
            #t_h_l_with_vs = h_l_with_vs[0] + h_l_with_vs[-1]
            #t_h_v_with_ls = h_v_with_ls[0] + h_v_with_ls[-1]
            h_a_gmu, z1_a = self.gmu_a_m([h_l_with_vs[0], h_l_with_vs[-1], h_v_with_ls[0], h_v_with_ls[-1]])
            '''
            # Residual conection level 1 to 2 -------
#            h_a_with_v2l_tot = h_a_with_v2l + t_h_l_with_vs #h_a_with_v2l[0] + h_a_with_v2l[-1] + t_h_l_with_vs #[tok_a] += t_h_l_with_vs
           #h_a_with_v2l_tot2 = h_a_with_v2l[-1] + h_l_with_vs[-1]
#            h_a_with_l2v_tot = h_a_with_l2v + t_h_v_with_ls#h_a_with_l2v[0] + h_a_with_l2v[-1] + t_h_v_with_ls #[tok_a] += t_h_v_with_ls
            #h_a_with_l2v_tot2 = h_a_with_l2v[-1] + h_v_with_ls[-1]
            # Feature Fusion ---------
#            h_as_gmu, z2_a = self.gmu_a([h_a_with_v2l_tot, h_a_with_l2v_tot])
            h_as_gmu, z2_a = self.gmu_a([h_a_with_v2l, h_a_with_l2v, h_a_with_ls, h_a_with_vs])
            # Residual conection level 1 to 3 -------
#            h_as_gmu += h_a_gmu
            #h_as_gmu[-1] += h_a_gmu[-1]
            
            last_h_a = last_hs = h_as_gmu[0] + h_as_gmu[-1]

        if self.vonly:
            # Biprojection ---------
            h_v_with_a2l = self.trans_v_with_a2l(proj_x_v, h_l_with_as, h_l_with_as)
            h_v_with_l2a = self.trans_v_with_l2a(proj_x_v, h_a_with_ls, h_a_with_ls)
            # GMU Middle --------
            
#            t_h_l_with_as = self.transfm_l2v(h_l_with_as.permute(2,1,0)).permute(2,1,0)#[tok_l])
#            t_h_a_with_ls = h_a_with_ls #self.transfm_a2v(h_a_with_ls.permute(2,1,0)).permute(2,1,0)#[tok_a])
#            h_v_gmu, z1_v = self.gmu_v_m([h_v_with_ls, h_v_with_as])
            '''
        #    t_h_l_with_as = h_l_with_as[0] + h_l_with_as[-1]
         #   t_h_a_with_ls = h_a_with_ls[0] + h_a_with_ls[-1]
            h_v_gmu, z1_v = self.gmu_v_m([h_l_with_as[0], h_l_with_as[-1], h_a_with_ls[0],h_a_with_ls[-1]])
            '''
            # Residual conection level 1 to 2 -------
#            h_v_with_a2l_tot = h_v_with_a2l + t_h_l_with_as #h_v_with_a2l[0] + h_v_with_a2l[-1] + t_h_l_with_as #[tok_v] += t_h_l_with_as
            #h_v_with_a2l_tot2 = h_v_with_a2l[-1] + h_l_with_as[-1]
#            h_v_with_l2a_tot = h_v_with_l2a + t_h_a_with_ls #h_v_with_l2a[0] + h_v_with_l2a[-1] + t_h_a_with_ls #[tok_v] += t_h_a_with_ls
            #h_v_with_l2a_tot2 = h_v_with_l2a[-1] + h_a_with_ls[-1]
            # Feature Fusion ---------
#            h_vs_gmu, z2_v = self.gmu_v([h_v_with_a2l_tot, h_v_with_l2a_tot])
            h_vs_gmu, z2_v = self.gmu_v([h_v_with_a2l, h_v_with_l2a, h_v_with_ls, h_v_with_as])
            # Residual conection level 1 to 3 -------
#            h_vs_gmu += h_v_gmu
           # h_vs_gmu[-1] += h_v_gmu[-1]
            
            last_h_v = last_hs = h_vs_gmu[0] + h_vs_gmu[-1]
        
        #last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a)
 #       last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a, poster) #, h_early.squeeze(0))
        #last_hs, z = self.gmu(last_h_l1, last_h_l2, last_h_v1, last_h_v2, last_h_a1, last_h_a2, h_v_with_as[tok], h_a_with_vs[tok], h_v_with_ls[tok], h_l_with_vs[tok], h_a_with_ls[tok], h_l_with_as[tok], poster)
        if self.hybrid:
            last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a, poster, last_h_early)#last_hl_early, last_hv_early, last_ha_early)
        else:
            #print(last_h_l.size(), last_h_v.size(), last_h_a.size(), poster.size())
            last_hs, z = self.gmu([last_h_l, last_h_v, last_h_a, poster])
       # last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a, last_h_v2, last_h_a1, last_h_a2, poster)
        #print(last_hs.size())
        #last_hs = last_hs.squeeze(0)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        if output_gate:
            # Return attention from crossmodal transformers (token CLS)
            '''
            return self.out_layer(last_hs_proj), attention_CLS
            '''
            # Return GMU activation (Final GMU)
            #'''
            return self.out_layer(last_hs_proj), z
            #'''
            # Return GMU activation (Middle GMUs)
            '''
            which_token = 0 #-1
            return self.out_layer(last_hs_proj), torch.cat((z1_l[which_token], z2_l[which_token], z1_v[which_token], z2_v[which_token], z1_a[which_token], z2_a[which_token]), dim=1)
            '''
            
        else:
            return self.out_layer(last_hs_proj)
  
# El chingon pero para 3 clases usado para mosei
class MultiprojectionMMTransformer3DGMUClf(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model with a Transformer preprocessing for video and audio
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
        
        combined_dim = args.hidden_sz#768 # For GMU
        output_dim = args.n_classes        # This is actually not a hyperparameter :-)

        
        self.enc = BertEncoder(args)
#Comment following line to IMDb
        #self.audio_enc = AudioEncoder(args)
        
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
            self.gmu_early = TextShifting3Layer(combined_dim, combined_dim, combined_dim, combined_dim)
            #self.gmu_early = TextShifting3Layer(self.low_dim, self.low_dim, self.low_dim, combined_dim)
        
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
            
        # Trasformer preprocessing and prepend CLS Trainable Tokens
        '''
        ones_v = torch.ones(1, proj_x_v.size(1), proj_x_v.size(2)).cuda()
        ones_a = torch.ones(1, proj_x_a.size(1), proj_x_a.size(2)).cuda()
        
        proj_x_a = self.trans_a_ppro(torch.cat((proj_x_l[0].unsqueeze(0), proj_x_a), dim=0))
        proj_x_v = self.trans_v_ppro(torch.cat((proj_x_l[0].unsqueeze(0), proj_x_v), dim=0))
    #
        cls_a = self.CLS_a(proj_x_a)#.permute(2,1,0)).permute(2,1,0)
        cls_v = self.CLS_v(proj_x_v)#.permute(2,1,0)).permute(2,1,0)
    #
        cls_a = self.CLS_a(ones_a)
        cls_v = self.CLS_v(ones_v)
        
        proj_x_a = self.trans_a_ppro(torch.cat((cls_a[0].unsqueeze(0), proj_x_a), dim=0))
        proj_x_v = self.trans_v_ppro(torch.cat((cls_v[0].unsqueeze(0), proj_x_v), dim=0))
        '''

        # First Crossmodal Transformers
        # (V,A) --> L
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)    # Dimension (V, N, d_v)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)    # Dimension (A, N, d_a)
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
            
            t_h_a_with_vs = h_a_with_vs #self.transfm_a2l(h_a_with_vs.permute(2,1,0)).permute(2,1,0)#h_a_with_vs[tok_a]
            t_h_v_with_as = h_v_with_as #self.transfm_v2l(h_v_with_as.permute(2,1,0)).permute(2,1,0)#[tok_v])
            h_l_gmu, z1_l = self.gmu_l_m([t_h_v_with_as, t_h_a_with_vs])
            '''
            #t_h_a_with_vs = h_a_with_vs[0] + h_a_with_vs[-1]
            #t_h_v_with_as = h_v_with_as[0]  + h_v_with_as[-1]
            h_l_gmu, z1_l = self.gmu_l_m([h_a_with_vs[0], h_a_with_vs[-1], h_v_with_as[0], h_v_with_as[-1]])
            '''
            # Residual conection level 1 to 2 -------
            h_l_with_v2a_tot = h_l_with_v2a + t_h_a_with_vs #h_l_with_v2a[0] + h_l_with_v2a[-1] + t_h_a_with_vs[0] + t_h_a_with_vs[-1]
            #h_l_with_v2a_tot2 = h_l_with_v2a[-1] + h_a_with_vs[-1]
            h_l_with_a2v_tot = h_l_with_a2v + t_h_v_with_as #h_l_with_a2v[0] + h_l_with_a2v[-1] + t_h_v_with_as[0] + t_h_v_with_as[-1]
            #h_l_with_a2v_tot2 = h_l_with_a2v[-1] + h_v_with_as[-1]
            
            # Feature Fusion ---------
            h_ls_gmu, z2_l = self.gmu_l([h_l_with_a2v_tot, h_l_with_v2a_tot])
            #h_ls_gmu, z2_l = self.gmu_l([h_l_with_v2a_tot1, h_l_with_v2a_tot2, h_l_with_a2v_tot1, h_l_with_a2v_tot2])
            
            # Residual conection level 1 to 3 -------
            h_ls_gmu += h_l_gmu
            #h_ls_gmu[-1] += h_l_gmu[-1]

            last_h_l = last_hs = h_ls_gmu[0] + h_ls_gmu[-1]

        if self.aonly:
            # Biprojection ---------
            h_a_with_v2l = self.trans_a_with_v2l(proj_x_a, h_l_with_vs, h_l_with_vs)
            h_a_with_l2v = self.trans_a_with_l2v(proj_x_a, h_v_with_ls, h_v_with_ls)
            # GMU Middle --------
            
            t_h_l_with_vs = h_l_with_vs#self.transfm_l2a(h_l_with_vs.permute(2,1,0)).permute(2,1,0)#[tok_l])
            t_h_v_with_ls = h_v_with_ls #self.transfm_v2a(h_v_with_ls.permute(2,1,0)).permute(2,1,0)#[tok_v])
            h_a_gmu, z1_a = self.gmu_a_m([t_h_l_with_vs, t_h_v_with_ls])
            '''
            #t_h_l_with_vs = h_l_with_vs[0] + h_l_with_vs[-1]
            #t_h_v_with_ls = h_v_with_ls[0] + h_v_with_ls[-1]
            h_a_gmu, z1_a = self.gmu_a_m([h_l_with_vs[0], h_l_with_vs[-1], h_v_with_ls[0], h_v_with_ls[-1]])
            '''
            # Residual conection level 1 to 2 -------
            h_a_with_v2l_tot = h_a_with_v2l + t_h_l_with_vs #h_a_with_v2l[0] + h_a_with_v2l[-1] + t_h_l_with_vs #[tok_a] += t_h_l_with_vs
           # h_a_with_v2l_tot2 = h_a_with_v2l[-1] + h_l_with_vs[-1]
            h_a_with_l2v_tot = h_a_with_l2v + t_h_v_with_ls#h_a_with_l2v[0] + h_a_with_l2v[-1] + t_h_v_with_ls #[tok_a] += t_h_v_with_ls
         #   h_a_with_l2v_tot2 = h_a_with_l2v[-1] + h_v_with_ls[-1]
            # Feature Fusion ---------
            h_as_gmu, z2_a = self.gmu_a([h_a_with_v2l_tot, h_a_with_l2v_tot])
            # Residual conection level 1 to 3 -------
            h_as_gmu += h_a_gmu
            #h_as_gmu[-1] += h_a_gmu[-1]
            
            last_h_a = last_hs = h_as_gmu[0] + h_as_gmu[-1]

        if self.vonly:
            # Biprojection ---------
            h_v_with_a2l = self.trans_v_with_a2l(proj_x_v, h_l_with_as, h_l_with_as)
            h_v_with_l2a = self.trans_v_with_l2a(proj_x_v, h_a_with_ls, h_a_with_ls)
            # GMU Middle --------
            
            t_h_l_with_as = h_l_with_as#self.transfm_l2v(h_l_with_as.permute(2,1,0)).permute(2,1,0)#[tok_l])
            t_h_a_with_ls = h_a_with_ls #self.transfm_a2v(h_a_with_ls.permute(2,1,0)).permute(2,1,0)#[tok_a])
            h_v_gmu, z1_v = self.gmu_v_m([t_h_l_with_as, t_h_a_with_ls])
            '''
        #    t_h_l_with_as = h_l_with_as[0] + h_l_with_as[-1]
         #   t_h_a_with_ls = h_a_with_ls[0] + h_a_with_ls[-1]
            h_v_gmu, z1_v = self.gmu_v_m([h_l_with_as[0], h_l_with_as[-1], h_a_with_ls[0],h_a_with_ls[-1]])
            '''
            # Residual conection level 1 to 2 -------
            h_v_with_a2l_tot = h_v_with_a2l + t_h_l_with_as #h_v_with_a2l[0] + h_v_with_a2l[-1] + t_h_l_with_as #[tok_v] += t_h_l_with_as
        #    h_v_with_a2l_tot2 = h_v_with_a2l[-1] + h_l_with_as[-1]
            h_v_with_l2a_tot = h_v_with_l2a + t_h_a_with_ls #h_v_with_l2a[0] + h_v_with_l2a[-1] + t_h_a_with_ls #[tok_v] += t_h_a_with_ls
         #   h_v_with_l2a_tot2 = h_v_with_l2a[-1] + h_a_with_ls[-1]
            # Feature Fusion ---------
            h_vs_gmu, z2_v = self.gmu_v([h_v_with_a2l_tot, h_v_with_l2a_tot])
            # Residual conection level 1 to 3 -------
            h_vs_gmu += h_v_gmu
           # h_vs_gmu[-1] += h_v_gmu[-1]
            
            last_h_v = last_hs = h_vs_gmu[0] + h_vs_gmu[-1]
        
        #last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a)
 #       last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a, poster) #, h_early.squeeze(0))
        #last_hs, z = self.gmu(last_h_l1, last_h_l2, last_h_v1, last_h_v2, last_h_a1, last_h_a2, h_v_with_as[tok], h_a_with_vs[tok], h_v_with_ls[tok], h_l_with_vs[tok], h_a_with_ls[tok], h_l_with_as[tok], poster)
        if self.hybrid:
            last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a, last_h_early)#last_hl_early, last_hv_early, last_ha_early)
        else:
            #print(last_h_l.size(), last_h_v.size(), last_h_a.size(), poster.size())
            last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a)
       # last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a, last_h_v2, last_h_a1, last_h_a2, poster)
        #print(last_hs.size())
        #last_hs = last_hs.squeeze(0)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        if output_gate:
            # Return attention from crossmodal transformers (token CLS)
            '''
            return self.out_layer(last_hs_proj), attention_CLS
            '''
            # Return GMU activation (Final GMU)
            #'''
            return self.out_layer(last_hs_proj), z
            #'''
            # Return GMU activation (Middle GMUs)
            '''
            which_token = 0 #-1
            return self.out_layer(last_hs_proj), torch.cat((z1_l[which_token], z2_l[which_token], z1_v[which_token], z2_v[which_token], z1_a[which_token], z2_a[which_token]), dim=1)
            '''
            
        else:
            return self.out_layer(last_hs_proj)
            
class TranslatingMMTransformerGMUClf_TPrepro(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model with a Transformer preprocessing for video and audio
        """
        super(TranslatingMMTransformerGMUClf_TPrepro, self).__init__()
        self.args = args
        self.orig_d_l, self.orig_d_v, self.orig_d_a, args.orig_d_m = args.orig_d_l, args.orig_d_v, args.orig_d_a, 312
        self.d_l, self.d_a, self.d_v, self.d_m = 768, 768, 768, 768
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
        
        self.enc = BertEncoder(args)
        self.audio_enc = AudioEncoder(args)

        # 0. Project poster feature to 768 dim
 #       self.proj_poster  = nn.Linear(self.orig_d_v, self.d_v, bias=False)
        #self.proj2_poster = nn.Linear(self.orig_d_v, self.d_v, bias=False)
        # 0. Temporal Linear layers for Early fusion
        #------For the crossmodal layers sum
        #self.proj_v2a = nn.Linear(self.d_l, self.d_l)#, bias=False)
        #self.proj_a2v = nn.Linear(self.d_l, self.d_l)#, bias=False)
        #self.proj_v2l = nn.Linear(self.d_a, self.d_a)#, bias=False)
        #self.proj_l2v = nn.Linear(self.d_a, self.d_a)#, bias=False)
        #self.proj_l2a = nn.Linear(self.d_v, self.d_v)#, bias=False)
        #self.proj_a2l = nn.Linear(self.d_v, self.d_v)#, bias=False)
        #--- GMU instead of sum
        #------ GMU Middle
        '''self.gmu_l_m = GatedMultimodalLayerFeatures(self.d_v, self.d_a, self.d_l)
        self.gmu_v_m = GatedMultimodalLayerFeatures(self.d_l, self.d_a, self.d_v)
        self.gmu_a_m = GatedMultimodalLayerFeatures(self.d_l, self.d_v, self.d_a)
        #------ GMU Top
        self.gmu_l = GatedMultimodalLayerFeatures(self.d_v, self.d_a, self.d_l)
        self.gmu_v = GatedMultimodalLayerFeatures(self.d_l, self.d_a, self.d_v)
        self.gmu_a = GatedMultimodalLayerFeatures(self.d_l, self.d_v, self.d_a)
        '''
        #--------- GMU Top with sum
 #       self.gmu_l_m = GatedMultimodal3LayerFeatures(self.d_l, self.d_l, self.d_l, self.d_l)
  #      self.gmu_v_m = GatedMultimodal3LayerFeatures(self.d_v, self.d_v, self.d_v, self.d_v)
   #     self.gmu_a_m = GatedMultimodal3LayerFeatures(self.d_a, self.d_a, self.d_a, self.d_a)
        #------ GMU Top
 #       self.gmu_l = GatedMultimodal4LayerFeatures(self.d_l, self.d_l, self.d_l, self.d_l, self.d_l)
  #      self.gmu_v = GatedMultimodal4LayerFeatures(self.d_v, self.d_v, self.d_v, self.d_v, self.d_v)
   #     self.gmu_a = GatedMultimodal4LayerFeatures(self.d_a, self.d_a, self.d_a, self.d_a, self.d_a)
        #self.gmu_l = GatedMultimodal3LayerFeatures(self.d_l, self.d_l, self.d_l, self.d_l)
        #self.gmu_v = GatedMultimodal3LayerFeatures(self.d_v, self.d_v, self.d_v, self.d_l)
        #self.gmu_a = GatedMultimodal3LayerFeatures(self.d_a, self.d_a, self.d_a, self.d_l)
        #-------------
        #self.proj_va2l = nn.Linear(self.d_l * 2, self.d_l, bias=False)
        #self.proj_vl2a = nn.Linear(self.d_a * 2, self.d_a, bias=False)
        #self.proj_la2v = nn.Linear(self.d_v * 2, self.d_v, bias=False)

        combined_dim = 768 # For GMU
        
        output_dim = args.n_classes        # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.lonly:
    #        self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
            self.trans_l_with_v2a = self.get_network(self_type='lv2a')
            self.trans_l_with_a2v = self.get_network(self_type='la2v')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')
    #        self.trans_v_with_l2a = self.get_network(self_type='vl2a')
     #       self.trans_v_with_a2l = self.get_network(self_type='va2l')
        if self.aonly:
    #        self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
            self.trans_a_with_v2l = self.get_network(self_type='av2l')
            self.trans_a_with_l2v = self.get_network(self_type='al2v')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        #self.trans_p_ppro = self.get_network(self_type='p_mem', layers=3)
        #self.trans_v_ppro = self.get_network(self_type='v_mem', layers=6)
        #self.trans_a_ppro = self.get_network(self_type='a_mem', layers=6)
        self.trans_l1_mem = self.get_network(self_type='l_mem', layers=6)
   #     self.trans_l2_mem = self.get_network(self_type='l_mem', layers=6)
  ####      self.trans_v_mem = self.get_network(self_type='v_mem', layers=6)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=6)
 ##       self.trans_l1_mem = GatedMultimodalLayerFeatures(self.d_l, self.d_l, self.d_l)
  ##      self.trans_v_mem = GatedMultimodalLayerFeatures(self.d_v, self.d_v, self.d_v)
  ##      self.trans_a_mem = GatedMultimodalLayerFeatures(self.d_a, self.d_a, self.d_a)
        #self.CLS_v = nn.Linear(combined_dim, combined_dim)
        #self.CLS_a = nn.Linear(combined_dim, combined_dim)
        # MAG
        #beta_shift, dropout_prob = 1e-3, 0.5
        #self.MAG_v = MAG(self.d_v, beta_shift, dropout_prob)
        #self.MAG_a = MAG(self.d_a, beta_shift, dropout_prob)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim*4, combined_dim*4)
        self.proj2 = nn.Linear(combined_dim*4, combined_dim*4)
        self.out_layer = nn.Linear(combined_dim*4, output_dim)
        #self.proj12 = nn.Linear(combined_dim, combined_dim)
        #self.proj22 = nn.Linear(combined_dim, combined_dim)
        #self.out_layer2 = nn.Linear(combined_dim, output_dim)
        #self.out_layer_final = nn.Linear(output_dim, output_dim)
        
        # GMU layer for fusing text and image and audio information
   #     self.gmu = nn.Linear(self.d_l*2+self.d_v*2+self.d_a*2, combined_dim)
#        self.gmu = TextShifting4Layer(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l)
      ###  self.gmu = TextShifting4Layer(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l)
        #####self.gmu = TextShifting3Layer(self.d_l, self.d_l*2, self.d_l*2, self.d_l)
   #     self.gmu = GatedMultimodalLayer(self.d_l*2, self.d_l*2, self.d_l)
    #self.gmu = GatedMultimodal4LayerFusion(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l)
        #self.gmu = TextShifting5Layer(self.d_l*2, self.d_v*2, self.d_v*2, self.d_v, self.d_l, self.d_l)
        #self.gmuSimple = TextShifting4LayerSimple(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l)
        #self.gmuSimple = TextShifting3LayerSimple(self.d_l, self.d_v, self.d_a, self.d_l)
        
        #Transformation dimension layers
  #      self.transfm_a2l = nn.Linear(200, 512)
   #     self.transfm_v2l = nn.Linear(200, 512)
    #    self.transfm_l2a = nn.Linear(512, 200)
     #   self.transfm_l2v = nn.Linear(512, 200)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl', 'av2l', 'va2l']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va', 'lv2a', 'vl2a']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av', 'la2v', 'al2v']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
        # With GMU
            embed_dim, attn_dropout = self.d_l*2, self.attn_dropout
        # Without GMU (normal concat)
            #embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a*2, self.attn_dropout
            #embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = self.d_v*2, self.attn_dropout
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
        
            
    def forward(self, txt, mask, segment, img, audio, poster, output_gate=False):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        #txt = torch.cat((txt, txt), 0)
        #mask = torch.cat((mask, mask), 0)
        #segment = torch.cat((segment, segment), 0)
        #img = torch.cat((img, img), 0)
        #audio = torch.cat((audio, audio), 0)
        #poster = torch.cat((poster, poster), 0)
        x_l = self.enc(txt, mask, segment)
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v = img.transpose(1, 2)
        x_a = self.audio_enc(audio)
        

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        '''
        if proj_x_l.size(0) != 512:
            proj_x_l = self.transfm_2dim(proj_x_l, 0, 512)
        if proj_x_a.size(0) != 200:
            proj_x_a = self.transfm_2dim(proj_x_a, 0, 200)
        if proj_x_v.size(0) != 200:
            proj_x_v = self.transfm_2dim(proj_x_v, 0, 200)
        '''
        # Trasformer preprocessing and prepend BERT CLS Token
        #proj_x_a = self.trans_a_ppro(torch.cat((proj_x_l[0].unsqueeze(0), proj_x_a), dim=0))
        #proj_x_v = self.trans_a_ppro(torch.cat((proj_x_l[0].unsqueeze(0), proj_x_v), dim=0))
        #cls_a = self.CLS_a(proj_x_a.permute(2,1,0)).permute(2,1,0)
        #cls_v = self.CLS_v(proj_x_v.permute(2,1,0)).permute(2,1,0)
        '''cls_a = self.CLS_a(proj_x_l[0]).unsqueeze(0)
        cls_v = self.CLS_v(proj_x_l[0]).unsqueeze(0)
        t_cls_a = self.trans_a_ppro(torch.cat((cls_a, proj_x_a), dim=0))[0]
        t_cls_v = self.trans_v_ppro(torch.cat((cls_v, proj_x_v), dim=0))[0]
        '''
      #  poster   = self.proj_poster(poster)
        #self.trans_p_ppro(torch.cat((proj_x_l[0].unsqueeze(0), self.proj_poster(poster).unsqueeze(0)), dim=0))[0]

        # (V,A) --> L
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)    # Dimension (V, N, d_v)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)    # Dimension (A, N, d_a)
        # (L,V) --> A
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
        # (L,A) --> V
   #     h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
    #    h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
        
        tok = -1
        self.vonly= False
        if self.lonly:
            
            #attention_CLS = torch.cat((h_v_with_as[0], h_a_with_vs[0]), dim=1)
            
            # Feature Dimension Transformation
     #       t_h_a_with_vs = self.transfm_a2l(h_a_with_vs.permute(2, 1, 0)).permute(2, 1, 0)
      #      t_h_v_with_as = self.transfm_v2l(h_v_with_as.permute(2, 1, 0)).permute(2, 1, 0)
            # GMU Middle --------
      #      h_l_gmu, z1_l = self.gmu_l_m(t_h_v_with_as, t_h_a_with_vs)
            #h_ls = self.trans_l_with_v2a(proj_x_l, h_l_gmu, h_l_gmu)
            # GMU Top ---------
            h_l_with_v2a = self.trans_l_with_v2a(proj_x_l, h_a_with_vs, h_a_with_vs)# Dimension (L, N, d_l)
            h_l_with_a2v = self.trans_l_with_a2v(proj_x_l, h_v_with_as, h_v_with_as)# Dimension (L, N, d_l)
            #attention_CLS = torch.cat((attention_CLS, h_l_with_a2v[0], h_l_with_v2a[0]), dim=1)
            # Residual conection
       #     h_l_with_v2a += t_h_a_with_vs
        #    h_l_with_a2v += t_h_v_with_as
            # Option 1 ---------
      ###      h_l_gmu, z1_l = self.gmu_l_m(h_l_with_v2a, h_l_with_a2v, h_l_with_a2v*h_l_with_v2a)
            h_l1_gmu = self.trans_l1_mem(torch.cat((h_l_with_v2a, h_l_with_a2v), dim=-1))[tok]
      #      h_l1_gmu = self.trans_l1_mem(h_l_with_v2a[tok], h_l_with_a2v[tok])[0]
  ##########          h_l2_gmu = self.trans_l1_mem(torch.cat((h_l_with_vs, h_l_with_as), dim=-1))[tok]
            #h_ls_gmu, z2_l = self.gmu(proj_x_l[0], h_l1_gmu, h_l2_gmu)
       #####     h_ls_gmu, z2_l = self.gmu(h_l1_gmu, h_l2_gmu)
            #h_ls = h_ls_gmu + h_l_gmu
            
            last_h_l = last_hs = h_ls_gmu = h_l1_gmu
            #last_h_l = last_hs = h_l_with_a2v[-1]
            
        if self.aonly:
            
    #        attention_CLS = torch.cat((attention_CLS, h_v_with_ls[0], h_l_with_vs[0]), dim=1)
            
            # Feature Dimension Transformation
     #       t_h_l_with_vs = self.transfm_l2a(h_l_with_vs.permute(2, 1, 0)).permute(2, 1, 0)
            # GMU Middle --------
     #       h_a_gmu, z1_a = self.gmu_a_m(t_h_l_with_vs, h_v_with_ls)
            #h_as = self.trans_a_with_l2v(proj_x_a, h_a_gmu, h_a_gmu)
            # GMU Top --------
            h_a_with_v2l = self.trans_a_with_v2l(proj_x_a, h_l_with_vs, h_l_with_vs)
            h_a_with_l2v = self.trans_a_with_l2v(proj_x_a, h_v_with_ls, h_v_with_ls)
    #        attention_CLS = torch.cat((attention_CLS, h_a_with_l2v[0], h_a_with_v2l[0]), dim=1)
            
            # Residual conection
       #     h_a_with_v2l += t_h_l_with_vs
        #    h_a_with_l2v += h_v_with_ls
            # Option 1 ---------
  ######          h_a_gmu, z1_a = self.gmu_a_m(h_a_with_v2l, h_a_with_l2v, h_a_with_v2l*h_a_with_l2v)
  #####          h_as_gmu, z2_a = self.gmu_a(h_a_gmu, proj_x_a[tok], h_a_with_vs[tok], h_a_with_ls[tok])
            #h_as = h_as_gmu + h_a_gmu
            h_as_gmu = self.trans_a_mem(torch.cat((h_a_with_v2l, h_a_with_l2v), dim=-1))[tok]
            #h_as_gmu = self.trans_a_mem(h_a_with_v2l[tok], h_a_with_l2v[tok])[0]
            last_h_a = last_hs = h_as_gmu
            #print(last_h_a.size())

        if self.vonly:
            
   #         attention_CLS = torch.cat((attention_CLS, h_a_with_ls[0], h_l_with_as[0]), dim=1)
            
            # Feature Dimension Transformation
     #       t_h_l_with_as = self.transfm_l2v(h_l_with_as.permute(2, 1, 0)).permute(2, 1, 0)
            # GMU Middle --------
      #      h_v_gmu, z1_v = self.gmu_v_m(t_h_l_with_as, h_a_with_ls)
            #h_vs = self.trans_v_with_l2a(proj_x_v, h_v_gmu, h_v_gmu)
            # GMU Top --------
            h_v_with_a2l = self.trans_v_with_a2l(proj_x_v, h_l_with_as, h_l_with_as)
            h_v_with_l2a = self.trans_v_with_l2a(proj_x_v, h_a_with_ls, h_a_with_ls)
     #       attention_CLS = torch.cat((attention_CLS, h_v_with_l2a[0], h_v_with_a2l[0]), dim=1)
            
            # Residual conection
      #      h_v_with_a2l += t_h_l_with_as
       #     h_v_with_l2a += h_a_with_ls
            # Option 1 ---------
  ######          h_v_gmu, z1_v = self.gmu_v_m(h_v_with_a2l, h_v_with_l2a, h_v_with_a2l*h_v_with_l2a)
  ######          h_vs_gmu, z2_v = self.gmu_v(h_v_gmu, proj_x_v[tok], h_v_with_ls[tok], h_v_with_as[tok])
          #  h_vs = h_vs_gmu + h_v_gmu
            h_vs_gmu = self.trans_v_mem(torch.cat((h_v_with_a2l, h_v_with_l2a), dim=-1))[tok]
          #  h_vs_gmu = self.trans_v_mem(h_v_with_a2l[tok], h_v_with_l2a[tok])[0]
            last_h_v = last_hs = h_vs_gmu
            
        
        #last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a)
   #     last_hs, z = self.gmu(last_h_l, poster)
        z = last_hs = torch.cat((last_h_l, last_h_a), dim=-1)
    #    last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a, poster) #, h_early.squeeze(0))
   ####     z = torch.cat((last_h_l, last_h_v, last_h_a), dim=-1)
     #   last_hs = self.gmu(z)
        
        #print(last_hs.size())
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        if output_gate:
            # Return attention from crossmodal transformers (token CLS)
            '''
            return self.out_layer(last_hs_proj), attention_CLS
            '''
            # Return GMU activation (Final GMU)
            #'''
            return self.out_layer(last_hs_proj), z
            #'''
            # Return GMU activation (Middle GMUs)
            '''
            which_token = 0 #-1
            return self.out_layer(last_hs_proj), torch.cat((z1_l[which_token], z2_l[which_token], z1_v[which_token], z2_v[which_token], z1_a[which_token], z2_a[which_token]), dim=1)
            '''
            
        else:
            return self.out_layer(last_hs_proj)
   
class TranslatingMMTransformerGMUClf_residual_v4(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model for Text, Video frames and Audio spectrogram with GMU late fusion.
        """
        super(TranslatingMMTransformerGMUClf_residual_v4, self).__init__()
        self.args = args
        self.orig_d_l, self.orig_d_v, self.orig_d_a, args.orig_d_m = args.orig_d_l, args.orig_d_v, args.orig_d_a, 312
        self.d_l, self.d_a, self.d_v, self.d_m = 768, 768, 768, 768
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
        
        self.enc = BertEncoder(args)
        self.audio_enc = AudioEncoder(args)

        # 0. Project poster feature to 768 dim
        self.proj_poster  = nn.Linear(self.orig_d_v, self.d_v, bias=False)
        #self.proj2_poster = nn.Linear(self.orig_d_v, self.d_v, bias=False)
        # 0. Temporal Linear layers for Early fusion
        #------For the crossmodal layers sum
        #self.proj_v2a = nn.Linear(self.d_l, self.d_l)#, bias=False)
        #self.proj_a2v = nn.Linear(self.d_l, self.d_l)#, bias=False)
        #self.proj_v2l = nn.Linear(self.d_a, self.d_a)#, bias=False)
        #self.proj_l2v = nn.Linear(self.d_a, self.d_a)#, bias=False)
        #self.proj_l2a = nn.Linear(self.d_v, self.d_v)#, bias=False)
        #self.proj_a2l = nn.Linear(self.d_v, self.d_v)#, bias=False)
        #--- GMU instead of sum
        #------ GMU Middle
        self.gmu_l_m = GatedMultimodalLayerFeatures(self.d_v, self.d_a, self.d_l)
        self.gmu_v_m = GatedMultimodalLayerFeatures(self.d_l, self.d_a, self.d_v)
        self.gmu_a_m = GatedMultimodalLayerFeatures(self.d_l, self.d_v, self.d_a)
        #------ GMU Top
        self.gmu_l = GatedMultimodalLayerFeatures(self.d_v, self.d_a, self.d_l)
        self.gmu_v = GatedMultimodalLayerFeatures(self.d_l, self.d_a, self.d_v)
        self.gmu_a = GatedMultimodalLayerFeatures(self.d_l, self.d_v, self.d_a)
        #--------- GMU Top with sum
        #self.gmu_l = GatedMultimodal3LayerFeatures(self.d_l, self.d_l, self.d_l, self.d_l)
        #self.gmu_v = GatedMultimodal3LayerFeatures(self.d_v, self.d_v, self.d_v, self.d_l)
        #self.gmu_a = GatedMultimodal3LayerFeatures(self.d_a, self.d_a, self.d_a, self.d_l)
        #-------------
        #self.proj_va2l = nn.Linear(self.d_l * 2, self.d_l, bias=False)
        #self.proj_vl2a = nn.Linear(self.d_a * 2, self.d_a, bias=False)
        #self.proj_la2v = nn.Linear(self.d_v * 2, self.d_v, bias=False)

        combined_dim = 768 # For GMU
        
        output_dim = args.n_classes        # This is actually not a hyperparameter :-)

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
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        #self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        #self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
        #self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        #self.proj12 = nn.Linear(combined_dim, combined_dim)
        #self.proj22 = nn.Linear(combined_dim, combined_dim)
        #self.out_layer2 = nn.Linear(combined_dim, output_dim)
        #self.out_layer_final = nn.Linear(output_dim, output_dim)
        
        # GMU layer for fusing text and image and audio information
        #self.gmu = TextShifting4Layer(self.d_l*2, self.d_v*2, self.d_a*2, self.d_v, self.d_l)
        self.gmu = TextShifting4Layer(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l)
        #self.gmu = TextShifting5Layer(self.d_l*2, self.d_v*2, self.d_v*2, self.d_v, self.d_l, self.d_l)
        #self.gmuSimple = TextShifting4LayerSimple(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l)
        #self.gmuSimple = TextShifting3LayerSimple(self.d_l, self.d_v, self.d_a, self.d_l)
        
        #Transformation dimension layers
        self.transfm_a2l = nn.Linear(200, 512)
        self.transfm_v2l = nn.Linear(200, 512)
        self.transfm_l2a = nn.Linear(512, 200)
        self.transfm_l2v = nn.Linear(512, 200)

    def get_network(self, self_type='l', layers=-1):
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
        
            
    def forward(self, txt, mask, segment, img, audio, poster, output_gate=False):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        #txt = torch.cat((txt, txt), 0)
        #mask = torch.cat((mask, mask), 0)
        #segment = torch.cat((segment, segment), 0)
        #img = torch.cat((img, img), 0)
        #audio = torch.cat((audio, audio), 0)
        #poster = torch.cat((poster, poster), 0)
        x_l = self.enc(txt, mask, segment)
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v = img.transpose(1, 2)
        x_a = self.audio_enc(audio)
        
       # proj_x_l_e = self.proj_l_e(x_l.permute(2, 0, 1))
       # proj_x_a_e = self.proj_a_e(x_a.permute(2, 0, 1))
       # proj_x_v_e = self.proj_v_e(x_v.permute(2, 0, 1))
       # proj_x_p_e = self.proj2_poster(poster)
        
        #Early fusion
       # h_early, zx = self.gmuSimple(proj_x_l_e, proj_x_a_e, proj_x_v_e, proj_x_p_e)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        
        if proj_x_l.size(0) != 512:
            proj_x_l = self.transfm_2dim(proj_x_l, 0, 512)
        if proj_x_a.size(0) != 200:
            proj_x_a = self.transfm_2dim(proj_x_a, 0, 200)
        if proj_x_v.size(0) != 200:
            proj_x_v = self.transfm_2dim(proj_x_v, 0, 200)

        if self.lonly:
            # (V,A) --> L
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)    # Dimension (V, N, d_v)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)    # Dimension (A, N, d_a)
            attention_CLS = torch.cat((h_v_with_as[0], h_a_with_vs[0]), dim=1)
            
            # Feature Dimension Transformation
            t_h_a_with_vs = self.transfm_a2l(h_a_with_vs.permute(2, 1, 0)).permute(2, 1, 0)
            t_h_v_with_as = self.transfm_v2l(h_v_with_as.permute(2, 1, 0)).permute(2, 1, 0)
            # GMU Middle --------
            h_l_gmu, z1_l = self.gmu_l_m(t_h_v_with_as, t_h_a_with_vs)
            #h_ls = self.trans_l_with_v2a(proj_x_l, h_l_gmu, h_l_gmu)
            # GMU Top ---------
            h_l_with_v2a = self.trans_l_with_v2a(proj_x_l, h_a_with_vs, h_a_with_vs)    # Dimension (L, N, d_l)
            h_l_with_a2v = self.trans_l_with_a2v(proj_x_l, h_v_with_as, h_v_with_as)    # Dimension (L, N, d_l)
            attention_CLS = torch.cat((attention_CLS, h_l_with_a2v[0], h_l_with_v2a[0]), dim=1)
            # Residual conection
            h_l_with_v2a += t_h_a_with_vs
            h_l_with_a2v += t_h_v_with_as
            # Option 1 ---------
            h_ls_gmu, z2_l = self.gmu_l(h_l_with_a2v, h_l_with_v2a)
            h_ls = h_ls_gmu + h_l_gmu
            #h_ls = 0.6*F.normalize(h_ls_gmu) + 0.4*F.normalize(h_l_gmu)
            '''
            mean1 = torch.mean(z1_l[-1][:, :self.d_l], dim=1, keepdim=False)
            mean2 = torch.mean(z1_l[-1][:, self.d_l:], dim=1, keepdim=False)
            alpha1_l = 1 - (mean1 - mean2)
            mean1 = torch.mean(z2_l[-1][:, :self.d_l], dim=1, keepdim=False)
            mean2 = torch.mean(z2_l[-1][:, self.d_l:], dim=1, keepdim=False)
            alpha2_l = 1 - (mean1 - mean2)
            #print(h_ls_gmu.size(), h_l_gmu.size())
            h_ls = alpha2_l*F.normalize(h_ls_gmu.permute(0, 2, 1)) + alpha1_l*F.normalize(h_l_gmu.permute(0, 2, 1))
            h_ls = h_ls.permute(0, 2, 1)
           # '''
            # Option 2 ---------
            #sum_ls = self.proj_v2a(h_l_with_v2a) + self.proj_a2v(h_l_with_a2v)
            #h_ls = self.gmu_l(h_l_with_v2a, h_l_with_a2v, sum_ls)
            # Option 3 ---------
            #h_ls = self.proj_va2l(torch.cat([h_l_with_v2a, h_l_with_a2v], dim=2))
            #print(h_ls.size(), h_l_gmu.size())
            # Normal Option --------
            #h_ls = self.trans_l_mem(torch.cat((h_l_with_v2a, h_l_with_a2v), dim=2))
            #h_ls += h_ls_gmu
            #if type(h_ls) == tuple:
             #   h_ls = h_ls[0]
            #last_h_l = last_hs = h_ls[-1] #+ h_ls_gmu[-1]
            #------------
            '''
            # Residual conection
            #h_ls[:200,:,:] += h_l_gmu
            h_ls = self.trans_l_mem(h_ls_gmu)
            #h_ls += h_ls_gmu
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1] + h_ls_gmu[-1]  # Take the last output for prediction
            '''
            last_h_l = last_hs = h_ls[0]
            #last_h_l = last_hs = h_l_with_a2v[-1]

        if self.aonly:
            # (L,V) --> A
    #        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
    #        attention_CLS = torch.cat((attention_CLS, h_v_with_ls[0], h_l_with_vs[0]), dim=1)
            
            # Feature Dimension Transformation
            t_h_l_with_vs = self.transfm_l2a(h_l_with_vs.permute(2, 1, 0)).permute(2, 1, 0)
            # GMU Middle --------
    #        h_a_gmu, z1_a = self.gmu_a_m(t_h_l_with_vs, h_v_with_ls)
            #h_as = self.trans_a_with_l2v(proj_x_a, h_a_gmu, h_a_gmu)
            # GMU Top --------
            h_a_with_v2l = self.trans_a_with_v2l(proj_x_a, h_l_with_vs, h_l_with_vs)
   #         h_a_with_l2v = self.trans_a_with_l2v(proj_x_a, h_v_with_ls, h_v_with_ls)
    #        attention_CLS = torch.cat((attention_CLS, h_a_with_l2v[0], h_a_with_v2l[0]), dim=1)
            
            # Residual conection
            h_a_with_v2l += t_h_l_with_vs
   #         h_a_with_l2v += h_v_with_ls
            # Option 1 ---------
   #         h_as_gmu, z2_a = self.gmu_a(h_a_with_v2l, h_a_with_l2v)
    #        h_as = h_as_gmu + h_a_gmu
            '''
            mean1 = torch.mean(z1_a[-1][:, :self.d_a], dim=1, keepdim=False)
            mean2 = torch.mean(z1_a[-1][:, self.d_a:], dim=1, keepdim=False)
            alpha1_a = 1 - (mean1 - mean2)
            mean1 = torch.mean(z2_a[-1][:, :self.d_a], dim=1, keepdim=False)
            mean2 = torch.mean(z2_a[-1][:, self.d_a:], dim=1, keepdim=False)
            alpha2_a = 1 - (mean1 - mean2)
            h_as = alpha2_a*F.normalize(h_as_gmu.permute(0, 2, 1)) + alpha1_a*F.normalize(h_a_gmu.permute(0, 2, 1))
            h_as = h_as.permute(0, 2, 1)
            #'''
            # Option 2 ---------
            #sum_as = self.proj_l2v(h_a_with_l2v) + self.proj_v2l(h_a_with_v2l)
            #h_as = self.gmu_a(h_a_with_l2v, h_a_with_v2l, sum_as)
            # Option 3 ---------
            #h_as = self.proj_vl2a(torch.cat([h_a_with_l2v, h_a_with_v2l], dim=2))
            #print(h_as.size())
            # Normal Option --------
            #h_as = self.trans_a_mem(torch.cat((h_a_with_v2l, h_a_with_l2v), dim=2))
            #h_ls += h_ls_gmu
            #if type(h_as) == tuple:
             #   h_as = h_as[0]
            #last_h_a = last_hs = h_as[-1]
            #------------
            '''
            # Residual conection
            #h_as = h_a_gmu[:200,:,:] + h_as
            h_as = self.trans_a_mem(h_as_gmu)
            #h_as += h_as_gmu
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1] + h_as_gmu[-1]
            '''
            #last_h_a = last_hs = h_as[0]
            last_h_a = last_hs = h_a_with_v2l[0]
            #print(last_h_a.size())

        if self.vonly:
            # (L,A) --> V
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
   #         attention_CLS = torch.cat((attention_CLS, h_a_with_ls[0], h_l_with_as[0]), dim=1)
            
            # Feature Dimension Transformation
            t_h_l_with_as = self.transfm_l2v(h_l_with_as.permute(2, 1, 0)).permute(2, 1, 0)
            # GMU Middle --------
            h_v_gmu, z1_v = self.gmu_v_m(t_h_l_with_as, h_a_with_ls)
            #h_vs = self.trans_v_with_l2a(proj_x_v, h_v_gmu, h_v_gmu)
            # GMU Top --------
   #         h_v_with_a2l = self.trans_v_with_a2l(proj_x_v, h_l_with_as, h_l_with_as)
    #        h_v_with_l2a = self.trans_v_with_l2a(proj_x_v, h_a_with_ls, h_a_with_ls)
     #       attention_CLS = torch.cat((attention_CLS, h_v_with_l2a[0], h_v_with_a2l[0]), dim=1)
            
            # Residual conection
   #         h_v_with_a2l += t_h_l_with_as
    #        h_v_with_l2a += h_a_with_ls
            # Option 1 ---------
   #         h_vs_gmu, z2_v = self.gmu_v(h_v_with_a2l, h_v_with_l2a)
   #         h_vs = h_vs_gmu + h_v_gmu
            '''
            mean1 = torch.mean(z1_v[-1][:, :self.d_v], dim=1, keepdim=False)
            mean2 = torch.mean(z1_v[-1][:, self.d_v:], dim=1, keepdim=False)
            alpha1_v = 1 - (mean1 - mean2)
            mean1 = torch.mean(z2_v[-1][:, :self.d_v], dim=1, keepdim=False)
            mean2 = torch.mean(z2_v[-1][:, self.d_v:], dim=1, keepdim=False)
            alpha2_v = 1 - (mean1 - mean2)
            h_vs = alpha2_v*F.normalize(h_vs_gmu.permute(0, 2, 1)) + alpha1_v*F.normalize(h_v_gmu.permute(0, 2, 1))
            h_vs = h_vs.permute(0, 2, 1)
            #'''
            # Option 2 ---------
            #sum_vs = self.proj_l2a(h_v_with_l2a) + self.proj_a2l(h_v_with_a2l)
            #h_vs = self.gmu_v(h_v_with_l2a, h_v_with_a2l, sum_vs)
            # Option 3 ---------
            #h_vs = self.proj_la2v(torch.cat([h_v_with_l2a, h_v_with_a2l], dim=2))
            #print(h_vs.size())
            # Normal Option --------
            #h_vs = self.trans_v_mem(torch.cat((h_v_with_a2l, h_v_with_l2a), dim=2))
            #if type(h_vs) == tuple:
             #   h_vs = h_vs[0]
            #last_h_v = last_hs = h_vs[-1]
            #------------
            '''
            # Residual conection
            #h_vs = h_v_gmu[:200,:,:] + h_vs
            h_vs = self.trans_v_mem(h_vs_gmu)
            h_vs += h_vs_gmu
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1] + h_vs_gmu[-1]
            '''
            #last_h_v = last_hs = h_vs[0]
            last_h_v = last_hs = h_v_gmu[0]
        
        #last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a)
        last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a, self.proj_poster(poster)) #, h_early.squeeze(0))
        #print(last_hs.size())
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        if output_gate:
            # Return attention from crossmodal transformers (token CLS)
            '''
            return self.out_layer(last_hs_proj), attention_CLS
            '''
            # Return GMU activation (Final GMU)
            #'''
            return self.out_layer(last_hs_proj), z
            #'''
            # Return GMU activation (Middle GMUs)
            '''
            which_token = 0 #-1
            return self.out_layer(last_hs_proj), torch.cat((z1_l[which_token], z2_l[which_token], z1_v[which_token], z2_v[which_token], z1_a[which_token], z2_a[which_token]), dim=1)
            '''
            
        else:
            return self.out_layer(last_hs_proj)
            
class TranslatingMMTransformerGMUClf_residual_v3(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model for Text, Video frames and Audio spectrogram with GMU late fusion.
        """
        super(TranslatingMMTransformerGMUClf_residual_v3, self).__init__()
        self.args = args
        self.orig_d_l, self.orig_d_v, self.orig_d_a, args.orig_d_m = args.orig_d_l, args.orig_d_v, args.orig_d_a, 312
        self.d_l, self.d_a, self.d_v, self.d_m = 768, 768, 768, 768
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
        
        self.enc = BertEncoder(args)
        self.audio_enc = AudioEncoder(args)

        # 0. Project poster feature to 768 dim
        self.proj_poster  = nn.Linear(self.orig_d_v, self.d_v, bias=False)
        #self.proj2_poster = nn.Linear(self.orig_d_v, self.d_v, bias=False)
        # 0. Temporal Linear layers for Early fusion
        #------For the crossmodal layers sum
        #self.proj_v2a = nn.Linear(self.d_l, self.d_l)#, bias=False)
        #self.proj_a2v = nn.Linear(self.d_l, self.d_l)#, bias=False)
        #self.proj_v2l = nn.Linear(self.d_a, self.d_a)#, bias=False)
        #self.proj_l2v = nn.Linear(self.d_a, self.d_a)#, bias=False)
        #self.proj_l2a = nn.Linear(self.d_v, self.d_v)#, bias=False)
        #self.proj_a2l = nn.Linear(self.d_v, self.d_v)#, bias=False)
        #--- GMU instead of sum
        #------ GMU Middle
        self.gmu_l_m = GatedMultimodalLayerFeatures(self.d_v, self.d_a, self.d_l)
        self.gmu_v_m = GatedMultimodalLayerFeatures(self.d_l, self.d_a, self.d_v)
        self.gmu_a_m = GatedMultimodalLayerFeatures(self.d_l, self.d_v, self.d_a)
        #------ GMU Top
        self.gmu_l = GatedMultimodalLayerFeatures(self.d_v, self.d_a, self.d_l)
        self.gmu_v = GatedMultimodalLayerFeatures(self.d_l, self.d_a, self.d_v)
        self.gmu_a = GatedMultimodalLayerFeatures(self.d_l, self.d_v, self.d_a)
        #--------- GMU Top with sum
        #self.gmu_l = GatedMultimodal3LayerFeatures(self.d_l, self.d_l, self.d_l, self.d_l)
        #self.gmu_v = GatedMultimodal3LayerFeatures(self.d_v, self.d_v, self.d_v, self.d_l)
        #self.gmu_a = GatedMultimodal3LayerFeatures(self.d_a, self.d_a, self.d_a, self.d_l)
        #-------------
        #self.proj_va2l = nn.Linear(self.d_l * 2, self.d_l, bias=False)
        #self.proj_vl2a = nn.Linear(self.d_a * 2, self.d_a, bias=False)
        #self.proj_la2v = nn.Linear(self.d_v * 2, self.d_v, bias=False)

        combined_dim = 768 # For GMU
        
        output_dim = args.n_classes        # This is actually not a hyperparameter :-)

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
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        #self.proj12 = nn.Linear(combined_dim, combined_dim)
        #self.proj22 = nn.Linear(combined_dim, combined_dim)
        #self.out_layer2 = nn.Linear(combined_dim, output_dim)
        #self.out_layer_final = nn.Linear(output_dim, output_dim)
        
        # GMU layer for fusing text and image and audio information
        #self.gmu = TextShifting4Layer(self.d_l*2, self.d_v*2, self.d_a*2, self.d_v, self.d_l)
        self.gmu = TextShifting4Layer(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l)
        #self.gmu = TextShifting5Layer(self.d_l*2, self.d_v*2, self.d_v*2, self.d_v, self.d_l, self.d_l)
        #self.gmuSimple = TextShifting4LayerSimple(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l)
        #self.gmuSimple = TextShifting3LayerSimple(self.d_l, self.d_v, self.d_a, self.d_l)
        
        #Transformation dimension layers
        self.transfm_a2l = nn.Linear(200, 512)
        self.transfm_v2l = nn.Linear(200, 512)
        self.transfm_l2a = nn.Linear(512, 200)
        self.transfm_l2v = nn.Linear(512, 200)

    def get_network(self, self_type='l', layers=-1):
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
        
            
    def forward(self, txt, mask, segment, img, audio, poster, output_gate=False):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        #txt = torch.cat((txt, txt), 0)
        #mask = torch.cat((mask, mask), 0)
        #segment = torch.cat((segment, segment), 0)
        #img = torch.cat((img, img), 0)
        #audio = torch.cat((audio, audio), 0)
        #poster = torch.cat((poster, poster), 0)
        x_l = self.enc(txt, mask, segment)
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v = img.transpose(1, 2)
        x_a = self.audio_enc(audio)
        
       # proj_x_l_e = self.proj_l_e(x_l.permute(2, 0, 1))
       # proj_x_a_e = self.proj_a_e(x_a.permute(2, 0, 1))
       # proj_x_v_e = self.proj_v_e(x_v.permute(2, 0, 1))
       # proj_x_p_e = self.proj2_poster(poster)
        
        #Early fusion
       # h_early, zx = self.gmuSimple(proj_x_l_e, proj_x_a_e, proj_x_v_e, proj_x_p_e)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        
        if proj_x_l.size(0) != 512:
            proj_x_l = self.transfm_2dim(proj_x_l, 0, 512)
        if proj_x_a.size(0) != 200:
            proj_x_a = self.transfm_2dim(proj_x_a, 0, 200)
        if proj_x_v.size(0) != 200:
            proj_x_v = self.transfm_2dim(proj_x_v, 0, 200)

        if self.lonly:
            # (V,A) --> L
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)    # Dimension (V, N, d_l)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)    # Dimension (A, N, d_l)
            
            # Feature Dimension Transformation
            t_h_a_with_vs = self.transfm_a2l(h_a_with_vs.permute(2, 1, 0)).permute(2, 1, 0)
            t_h_v_with_as = self.transfm_v2l(h_v_with_as.permute(2, 1, 0)).permute(2, 1, 0)
            # GMU Middle --------
            h_l_gmu, z1_l = self.gmu_l_m(t_h_v_with_as, t_h_a_with_vs)
            #h_ls = self.trans_l_with_v2a(proj_x_l, h_l_gmu, h_l_gmu)
            # GMU Top ---------
            h_l_with_v2a = self.trans_l_with_v2a(proj_x_l, h_a_with_vs, h_a_with_vs)    # Dimension (L, N, d_l)
            h_l_with_a2v = self.trans_l_with_a2v(proj_x_l, h_v_with_as, h_v_with_as)    # Dimension (L, N, d_l)
            
            # Residual conection
      #      h_l_with_v2a += t_h_a_with_vs
       #     h_l_with_a2v += t_h_v_with_as
            # Option 1 ---------
            h_ls_gmu, z2_l = self.gmu_l(h_l_with_a2v, h_l_with_v2a)
            h_ls = h_ls_gmu + h_l_gmu
            '''
            mean1 = torch.mean(z1_l[-1][:, :self.d_l], dim=1, keepdim=False)
            mean2 = torch.mean(z1_l[-1][:, self.d_l:], dim=1, keepdim=False)
            alpha1_l = 1 - (mean1 - mean2)
            mean1 = torch.mean(z2_l[-1][:, :self.d_l], dim=1, keepdim=False)
            mean2 = torch.mean(z2_l[-1][:, self.d_l:], dim=1, keepdim=False)
            alpha2_l = 1 - (mean1 - mean2)
            #print(h_ls_gmu.size(), h_l_gmu.size())
            h_ls = alpha2_l*F.normalize(h_ls_gmu.permute(0, 2, 1)) + alpha1_l*F.normalize(h_l_gmu.permute(0, 2, 1))
            h_ls = h_ls.permute(0, 2, 1)
           # '''
            # Option 2 ---------
            #sum_ls = self.proj_v2a(h_l_with_v2a) + self.proj_a2v(h_l_with_a2v)
            #h_ls = self.gmu_l(h_l_with_v2a, h_l_with_a2v, sum_ls)
            # Option 3 ---------
            #h_ls = self.proj_va2l(torch.cat([h_l_with_v2a, h_l_with_a2v], dim=2))
            #print(h_ls.size(), h_l_gmu.size())
            # Normal Option --------
            #h_ls = self.trans_l_mem(torch.cat((h_l_with_v2a, h_l_with_a2v), dim=2))
            #h_ls += h_ls_gmu
            #if type(h_ls) == tuple:
             #   h_ls = h_ls[0]
            #last_h_l = last_hs = h_ls[-1] #+ h_ls_gmu[-1]
            #------------
            '''
            # Residual conection
            #h_ls[:200,:,:] += h_l_gmu
            h_ls = self.trans_l_mem(h_ls_gmu)
            #h_ls += h_ls_gmu
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1] + h_ls_gmu[-1]  # Take the last output for prediction
            '''
            last_h_l = last_hs = h_ls[-1]
            #last_h_l = last_hs = h_l_with_a2v[-1]

        if self.aonly:
            # (L,V) --> A
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
            
            # Feature Dimension Transformation
            t_h_l_with_vs = self.transfm_l2a(h_l_with_vs.permute(2, 1, 0)).permute(2, 1, 0)
            # GMU Middle --------
            h_a_gmu, z1_a = self.gmu_a_m(t_h_l_with_vs, h_v_with_ls)
            #h_as = self.trans_a_with_l2v(proj_x_a, h_a_gmu, h_a_gmu)
            # GMU Top --------
            h_a_with_v2l = self.trans_a_with_v2l(proj_x_a, h_l_with_vs, h_l_with_vs)
            h_a_with_l2v = self.trans_a_with_l2v(proj_x_a, h_v_with_ls, h_v_with_ls)
            
            # Residual conection
        #    h_a_with_v2l += t_h_l_with_vs
         #   h_a_with_l2v += h_v_with_ls
            # Option 1 ---------
            h_as_gmu, z2_a = self.gmu_a(h_a_with_v2l, h_a_with_l2v)
            h_as = h_as_gmu + h_a_gmu
            '''
            mean1 = torch.mean(z1_a[-1][:, :self.d_a], dim=1, keepdim=False)
            mean2 = torch.mean(z1_a[-1][:, self.d_a:], dim=1, keepdim=False)
            alpha1_a = 1 - (mean1 - mean2)
            mean1 = torch.mean(z2_a[-1][:, :self.d_a], dim=1, keepdim=False)
            mean2 = torch.mean(z2_a[-1][:, self.d_a:], dim=1, keepdim=False)
            alpha2_a = 1 - (mean1 - mean2)
            h_as = alpha2_a*F.normalize(h_as_gmu.permute(0, 2, 1)) + alpha1_a*F.normalize(h_a_gmu.permute(0, 2, 1))
            h_as = h_as.permute(0, 2, 1)
            #'''
            # Option 2 ---------
            #sum_as = self.proj_l2v(h_a_with_l2v) + self.proj_v2l(h_a_with_v2l)
            #h_as = self.gmu_a(h_a_with_l2v, h_a_with_v2l, sum_as)
            # Option 3 ---------
            #h_as = self.proj_vl2a(torch.cat([h_a_with_l2v, h_a_with_v2l], dim=2))
            #print(h_as.size())
            # Normal Option --------
            #h_as = self.trans_a_mem(torch.cat((h_a_with_v2l, h_a_with_l2v), dim=2))
            #h_ls += h_ls_gmu
            #if type(h_as) == tuple:
             #   h_as = h_as[0]
            #last_h_a = last_hs = h_as[-1]
            #------------
            '''
            # Residual conection
            #h_as = h_a_gmu[:200,:,:] + h_as
            h_as = self.trans_a_mem(h_as_gmu)
            #h_as += h_as_gmu
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1] + h_as_gmu[-1]
            '''
            last_h_a = last_hs = h_as[-1]
            #last_h_a = last_hs = h_a_with_v2l[-1]
            #print(last_h_a.size())

        if self.vonly:
            # (L,A) --> V
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
            
            # Feature Dimension Transformation
            t_h_l_with_as = self.transfm_l2v(h_l_with_as.permute(2, 1, 0)).permute(2, 1, 0)
            # GMU Middle --------
            h_v_gmu, z1_v = self.gmu_v_m(t_h_l_with_as, h_a_with_ls)
            #h_vs = self.trans_v_with_l2a(proj_x_v, h_v_gmu, h_v_gmu)
            # GMU Top --------
            h_v_with_a2l = self.trans_v_with_a2l(proj_x_v, h_l_with_as, h_l_with_as)
            h_v_with_l2a = self.trans_v_with_l2a(proj_x_v, h_a_with_ls, h_a_with_ls)
            # Residual conection
      #      h_v_with_a2l += t_h_l_with_as
       #     h_v_with_l2a += h_a_with_ls
            # Option 1 ---------
            h_vs_gmu, z2_v = self.gmu_v(h_v_with_a2l, h_v_with_l2a)
            h_vs = h_vs_gmu + h_v_gmu
            '''
            mean1 = torch.mean(z1_v[-1][:, :self.d_v], dim=1, keepdim=False)
            mean2 = torch.mean(z1_v[-1][:, self.d_v:], dim=1, keepdim=False)
            alpha1_v = 1 - (mean1 - mean2)
            mean1 = torch.mean(z2_v[-1][:, :self.d_v], dim=1, keepdim=False)
            mean2 = torch.mean(z2_v[-1][:, self.d_v:], dim=1, keepdim=False)
            alpha2_v = 1 - (mean1 - mean2)
            h_vs = alpha2_v*F.normalize(h_vs_gmu.permute(0, 2, 1)) + alpha1_v*F.normalize(h_v_gmu.permute(0, 2, 1))
            h_vs = h_vs.permute(0, 2, 1)
            #'''
            # Option 2 ---------
            #sum_vs = self.proj_l2a(h_v_with_l2a) + self.proj_a2l(h_v_with_a2l)
            #h_vs = self.gmu_v(h_v_with_l2a, h_v_with_a2l, sum_vs)
            # Option 3 ---------
            #h_vs = self.proj_la2v(torch.cat([h_v_with_l2a, h_v_with_a2l], dim=2))
            #print(h_vs.size())
            # Normal Option --------
            #h_vs = self.trans_v_mem(torch.cat((h_v_with_a2l, h_v_with_l2a), dim=2))
            #if type(h_vs) == tuple:
             #   h_vs = h_vs[0]
            #last_h_v = last_hs = h_vs[-1]
            #------------
            '''
            # Residual conection
            #h_vs = h_v_gmu[:200,:,:] + h_vs
            h_vs = self.trans_v_mem(h_vs_gmu)
            h_vs += h_vs_gmu
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1] + h_vs_gmu[-1]
            '''
            last_h_v = last_hs = h_vs[-1]
            #last_h_v = last_hs = h_v_with_l2a[-1]
        
        #last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a)
        last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a, self.proj_poster(poster)) #, h_early.squeeze(0))
        #print(last_hs.size())
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        if output_gate:
            return self.out_layer(last_hs_proj), z#torch.cat((z1_l[-1], z2_l[-1],
                                                   #         z1_v[-1], z2_v[-1],
                                                    #        z1_a[-1], z2_a[-1]), dim=1)
        else:
            return self.out_layer(last_hs_proj)
        
        
class TranslatingMMTransformerGMUClf_residual_v2(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model for Text, Video frames and Audio spectrogram with GMU late fusion.
        """
        super(TranslatingMMTransformerGMUClf_residual_v2, self).__init__()
        self.args = args
        self.orig_d_l, self.orig_d_v, self.orig_d_a, args.orig_d_m = args.orig_d_l, args.orig_d_v, args.orig_d_a, 312
        self.d_l, self.d_a, self.d_v, self.d_m = 768, 768, 768, 768
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
        
        self.enc = BertEncoder(args)
        self.audio_enc = AudioEncoder(args)

        # 0. Project poster feature to 768 dim
        self.proj_poster  = nn.Linear(self.orig_d_v, self.d_v, bias=False)
        #self.proj2_poster = nn.Linear(self.orig_d_v, self.d_v, bias=False)
        # 0. Temporal Linear layers for Early fusion
        #------For the crossmodal layers sum
        #self.proj_v2a = nn.Linear(self.d_l, self.d_l)#, bias=False)
        #self.proj_a2v = nn.Linear(self.d_l, self.d_l)#, bias=False)
        #self.proj_v2l = nn.Linear(self.d_a, self.d_a)#, bias=False)
        #self.proj_l2v = nn.Linear(self.d_a, self.d_a)#, bias=False)
        #self.proj_l2a = nn.Linear(self.d_v, self.d_v)#, bias=False)
        #self.proj_a2l = nn.Linear(self.d_v, self.d_v)#, bias=False)
        #--- GMU instead of sum
        #------ GMU Middle
        self.gmu_l_m = GatedMultimodalLayerFeatures(self.d_v, self.d_a, self.d_l)
        self.gmu_v_m = GatedMultimodalLayerFeatures(self.d_l, self.d_a, self.d_v)
        self.gmu_a_m = GatedMultimodalLayerFeatures(self.d_l, self.d_v, self.d_a)
        #------ GMU Top
        self.gmu_l = GatedMultimodalLayerFeatures(self.d_v, self.d_a, self.d_l)
        self.gmu_v = GatedMultimodalLayerFeatures(self.d_l, self.d_a, self.d_v)
        self.gmu_a = GatedMultimodalLayerFeatures(self.d_l, self.d_v, self.d_a)
        #--------- GMU Top with sum
        #self.gmu_l = GatedMultimodal3LayerFeatures(self.d_l, self.d_l, self.d_l, self.d_l)
        #self.gmu_v = GatedMultimodal3LayerFeatures(self.d_v, self.d_v, self.d_v, self.d_l)
        #self.gmu_a = GatedMultimodal3LayerFeatures(self.d_a, self.d_a, self.d_a, self.d_l)
        #-------------
        #self.proj_va2l = nn.Linear(self.d_l * 2, self.d_l, bias=False)
        #self.proj_vl2a = nn.Linear(self.d_a * 2, self.d_a, bias=False)
        #self.proj_la2v = nn.Linear(self.d_v * 2, self.d_v, bias=False)

        combined_dim = 768 # For GMU
        
        output_dim = args.n_classes        # This is actually not a hyperparameter :-)

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
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        #self.proj12 = nn.Linear(combined_dim, combined_dim)
        #self.proj22 = nn.Linear(combined_dim, combined_dim)
        #self.out_layer2 = nn.Linear(combined_dim, output_dim)
        #self.out_layer_final = nn.Linear(output_dim, output_dim)
        
        # GMU layer for fusing text and image and audio information
        #self.gmu = TextShifting3Layer(self.d_l*2, self.d_v*2, self.d_a*2, self.d_l)
        self.gmu = TextShifting4Layer(self.d_l, self.d_v, self.d_v, self.d_v, self.d_l)
        #self.gmu = TextShifting5Layer(self.d_l*2, self.d_v*2, self.d_v*2, self.d_v, self.d_l, self.d_l)
        #self.gmuSimple = TextShifting4LayerSimple(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l)
        #self.gmuSimple = TextShifting3LayerSimple(self.d_l, self.d_v, self.d_a, self.d_l)
        
        #Transformation dimension layers
        self.transfm_a2l = nn.Linear(200, 512)
        self.transfm_v2l = nn.Linear(200, 512)
        self.transfm_l2a = nn.Linear(512, 200)
        self.transfm_l2v = nn.Linear(512, 200)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl', 'av2l', 'va2l']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va', 'lv2a', 'vl2a']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av', 'la2v', 'al2v']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
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
        
            
    def forward(self, txt, mask, segment, img, audio, poster, output_gate=False):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        #txt = torch.cat((txt, txt), 0)
        #mask = torch.cat((mask, mask), 0)
        #segment = torch.cat((segment, segment), 0)
        #img = torch.cat((img, img), 0)
        #audio = torch.cat((audio, audio), 0)
        #poster = torch.cat((poster, poster), 0)
        x_l = self.enc(txt, mask, segment)
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v = img.transpose(1, 2)
        x_a = self.audio_enc(audio)
        
       # proj_x_l_e = self.proj_l_e(x_l.permute(2, 0, 1))
       # proj_x_a_e = self.proj_a_e(x_a.permute(2, 0, 1))
       # proj_x_v_e = self.proj_v_e(x_v.permute(2, 0, 1))
       # proj_x_p_e = self.proj2_poster(poster)
        
        #Early fusion
       # h_early, zx = self.gmuSimple(proj_x_l_e, proj_x_a_e, proj_x_v_e, proj_x_p_e)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        
        if proj_x_l.size(0) != 512:
            proj_x_l = self.transfm_2dim(proj_x_l, 0, 512)
        if proj_x_a.size(0) != 200:
            proj_x_a = self.transfm_2dim(proj_x_a, 0, 200)
        if proj_x_v.size(0) != 200:
            proj_x_v = self.transfm_2dim(proj_x_v, 0, 200)

        if self.lonly:
            # (V,A) --> L
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)    # Dimension (L, N, d_v)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)    # Dimension (L, N, d_a)
            
            # GMU Middle --------
            #h_l_gmu = self.gmu_l_m(h_v_with_as, h_a_with_vs)
            #h_ls = self.trans_l_with_v2a(proj_x_l, h_l_gmu, h_l_gmu)
            # GMU Top ---------
            h_l_with_v2a = self.trans_l_with_v2a(proj_x_l, h_a_with_vs, h_a_with_vs)    # Dimension (L, N, d_l)
            h_l_with_a2v = self.trans_l_with_a2v(proj_x_l, h_v_with_as, h_v_with_as)    # Dimension (L, N, d_l)
            # Residual conection
            h_l_with_v2a += self.transfm_a2l(h_a_with_vs.permute(2, 1, 0)).permute(2, 1, 0)
            h_l_with_a2v += self.transfm_v2l(h_v_with_as.permute(2, 1, 0)).permute(2, 1, 0)
            # Option 1 ---------
            h_ls_gmu = self.gmu_l(h_l_with_a2v, h_l_with_v2a)
            # Option 2 ---------
            #sum_ls = self.proj_v2a(h_l_with_v2a) + self.proj_a2v(h_l_with_a2v)
            #h_ls = self.gmu_l(h_l_with_v2a, h_l_with_a2v, sum_ls)
            # Option 3 ---------
            #h_ls = self.proj_va2l(torch.cat([h_l_with_v2a, h_l_with_a2v], dim=2))
            #print(h_ls.size(), h_l_gmu.size())
            #------------
            # Residual conection
            #h_ls[:200,:,:] += h_l_gmu
            h_ls = self.trans_l_mem(h_ls_gmu)
            h_ls += h_ls_gmu
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]   # Take the last output for prediction
            #print(last_h_l.size())

        if self.aonly:
            # (L,V) --> A
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
            
            # GMU Middle --------
            #h_a_gmu = self.gmu_a_m(h_l_with_vs, h_v_with_ls)
            #h_as = self.trans_a_with_l2v(proj_x_a, h_a_gmu, h_a_gmu)
            # GMU Top --------
            h_a_with_v2l = self.trans_a_with_v2l(proj_x_a, h_l_with_vs, h_l_with_vs)
            h_a_with_l2v = self.trans_a_with_l2v(proj_x_a, h_v_with_ls, h_v_with_ls)
            # Residual conection
            h_a_with_v2l += self.transfm_l2a(h_l_with_vs.permute(2, 1, 0)).permute(2, 1, 0)
            h_a_with_l2v += h_v_with_ls
            # Option 1 ---------
            h_as_gmu = self.gmu_a(h_a_with_v2l, h_a_with_l2v)
            # Option 2 ---------
            #sum_as = self.proj_l2v(h_a_with_l2v) + self.proj_v2l(h_a_with_v2l)
            #h_as = self.gmu_a(h_a_with_l2v, h_a_with_v2l, sum_as)
            # Option 3 ---------
            #h_as = self.proj_vl2a(torch.cat([h_a_with_l2v, h_a_with_v2l], dim=2))
            #print(h_as.size())
            #------------
            # Residual conection
            #h_as = h_a_gmu[:200,:,:] + h_as
            h_as = self.trans_a_mem(h_as_gmu)
            h_as += h_as_gmu
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]
            #print(last_h_a.size())

        if self.vonly:
            # (L,A) --> V
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
            
            # GMU Middle --------
            h_v_gmu = self.gmu_v_m(h_l_with_as, h_a_with_ls)
            #h_vs = self.trans_v_with_l2a(proj_x_v, h_v_gmu, h_v_gmu)
            # GMU Top --------
            h_v_with_a2l = self.trans_v_with_a2l(proj_x_v, h_l_with_as, h_l_with_as)
            h_v_with_l2a = self.trans_v_with_l2a(proj_x_v, h_a_with_ls, h_a_with_ls)
            # Residual conection
            h_v_with_a2l += self.transfm_l2v(h_l_with_as.permute(2, 1, 0)).permute(2, 1, 0)
            h_v_with_l2a += h_a_with_ls
            # Option 1 ---------
            h_vs_gmu = self.gmu_v(h_v_with_a2l, h_v_with_l2a)
            # Option 2 ---------
            #sum_vs = self.proj_l2a(h_v_with_l2a) + self.proj_a2l(h_v_with_a2l)
            #h_vs = self.gmu_v(h_v_with_l2a, h_v_with_a2l, sum_vs)
            # Option 3 ---------
            #h_vs = self.proj_la2v(torch.cat([h_v_with_l2a, h_v_with_a2l], dim=2))
            #print(h_vs.size())
            #------------
            # Residual conection
            #h_vs = h_v_gmu[:200,:,:] + h_vs
            h_vs = self.trans_v_mem(h_vs_gmu)
            h_vs += h_vs_gmu
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]
            #print(last_h_v.size())
        
        #last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a)
        last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a, self.proj_poster(poster)) #, h_early.squeeze(0))
        #print(last_hs.size())
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)
        return output
        # A residual block
        '''
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        last_hs_proj2 = self.proj22(F.dropout(F.relu(self.proj12(h_early)), p=self.out_dropout, training=self.training))
        last_hs_proj2 += h_early
        
        output = self.out_layer(last_hs_proj)
        output2 = self.out_layer2(last_hs_proj2)
        if output.size() != output2.size():
            output2 = output2.squeeze(0)
        output = self.out_layer_final(torch.cat((output, output2)))
        if output_gate:
            size_aux = output.size()[0]
            return output[:int(size_aux/2),:] + output[int(size_aux/2):,:], z #torch.sum(output, 0).unsqueeze(0), z
        else:
            size_aux = output.size()[0]
            return output[:int(size_aux/2),:] + output[int(size_aux/2):,:] #torch.sum(output, 0).unsqueeze(0)
        '''
        
class TranslatingMMTransformerGMUClf_Middle(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model for Text, Video frames and Audio spectrogram with GMU late fusion.
        """
        super(TranslatingMMTransformerGMUClf_Middle, self).__init__()
        self.args = args
        self.orig_d_l, self.orig_d_v, self.orig_d_a, args.orig_d_m = args.orig_d_l, args.orig_d_v, args.orig_d_a, 312
        self.d_l, self.d_a, self.d_v, self.d_m = 768, 768, 768, 768
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
        
        self.enc = BertEncoder(args)
        self.audio_enc = AudioEncoder(args)

        # 0. Project poster feature to 768 dim
        self.proj_poster  = nn.Linear(self.orig_d_v, self.d_v, bias=False)
        #self.proj2_poster = nn.Linear(self.orig_d_v, self.d_v, bias=False)
        # 0. Temporal Linear layers for Early fusion
        #------For the crossmodal layers sum
        #self.proj_v2a = nn.Linear(self.d_l, self.d_l)#, bias=False)
        #self.proj_a2v = nn.Linear(self.d_l, self.d_l)#, bias=False)
        #self.proj_v2l = nn.Linear(self.d_a, self.d_a)#, bias=False)
        #self.proj_l2v = nn.Linear(self.d_a, self.d_a)#, bias=False)
        #self.proj_l2a = nn.Linear(self.d_v, self.d_v)#, bias=False)
        #self.proj_a2l = nn.Linear(self.d_v, self.d_v)#, bias=False)
        #------GMU instead of sum
        self.gmu_l = GatedMultimodalLayerFeatures(self.d_v, self.d_a, self.d_l)
        self.gmu_v = GatedMultimodalLayerFeatures(self.d_l, self.d_a, self.d_v)
        self.gmu_a = GatedMultimodalLayerFeatures(self.d_l, self.d_v, self.d_a)
        #self.gmu_l = GatedMultimodal3LayerFeatures(self.d_l, self.d_l, self.d_l, self.d_l)
        #self.gmu_v = GatedMultimodal3LayerFeatures(self.d_v, self.d_v, self.d_v, self.d_l)
        #self.gmu_a = GatedMultimodal3LayerFeatures(self.d_a, self.d_a, self.d_a, self.d_l)
        #-------------
        #self.proj_va2l = nn.Linear(self.d_l * 2, self.d_l, bias=False)
        #self.proj_vl2a = nn.Linear(self.d_a * 2, self.d_a, bias=False)
        #self.proj_la2v = nn.Linear(self.d_v * 2, self.d_v, bias=False)

        combined_dim = 768 # For GMU
        
        output_dim = args.n_classes        # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
            self.trans_l_with_v2a = self.get_network(self_type='lv2a')
            #self.trans_l_with_a2v = self.get_network(self_type='la2v')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')
            self.trans_v_with_l2a = self.get_network(self_type='vl2a')
            #self.trans_v_with_a2l = self.get_network(self_type='va2l')
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
            #self.trans_a_with_v2l = self.get_network(self_type='av2l')
            self.trans_a_with_l2v = self.get_network(self_type='al2v')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        #self.proj12 = nn.Linear(combined_dim, combined_dim)
        #self.proj22 = nn.Linear(combined_dim, combined_dim)
        #self.out_layer2 = nn.Linear(combined_dim, output_dim)
        #self.out_layer_final = nn.Linear(output_dim, output_dim)
        
        # GMU layer for fusing text and image and audio information
        #self.gmu = TextShifting3Layer(self.d_l*2, self.d_v*2, self.d_a*2, self.d_l)
        self.gmu = TextShifting4Layer(self.d_l, self.d_v, self.d_v, self.d_v, self.d_l)
        #self.gmu = TextShifting5Layer(self.d_l*2, self.d_v*2, self.d_v*2, self.d_v, self.d_l, self.d_l)
        #self.gmuSimple = TextShifting4LayerSimple(self.d_l, self.d_v, self.d_a, self.d_v, self.d_l)
        #self.gmuSimple = TextShifting3LayerSimple(self.d_l, self.d_v, self.d_a, self.d_l)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl', 'av2l', 'va2l']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va', 'lv2a', 'vl2a']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av', 'la2v', 'al2v']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
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
            
    def forward(self, txt, mask, segment, img, audio, poster, output_gate=False):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        #txt = torch.cat((txt, txt), 0)
        #mask = torch.cat((mask, mask), 0)
        #segment = torch.cat((segment, segment), 0)
        #img = torch.cat((img, img), 0)
        #audio = torch.cat((audio, audio), 0)
        #poster = torch.cat((poster, poster), 0)
        x_l = self.enc(txt, mask, segment)
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v = img.transpose(1, 2)
        x_a = self.audio_enc(audio)
        
       # proj_x_l_e = self.proj_l_e(x_l.permute(2, 0, 1))
       # proj_x_a_e = self.proj_a_e(x_a.permute(2, 0, 1))
       # proj_x_v_e = self.proj_v_e(x_v.permute(2, 0, 1))
       # proj_x_p_e = self.proj2_poster(poster)
        
        #Early fusion
       # h_early, zx = self.gmuSimple(proj_x_l_e, proj_x_a_e, proj_x_v_e, proj_x_p_e)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)

        if self.lonly:
            # (V,A) --> L
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)    # Dimension (L, N, d_v)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)    # Dimension (L, N, d_a)
            
            h_l_gmu = self.gmu_l(h_v_with_as, h_a_with_vs)
            
            h_ls = self.trans_l_with_v2a(proj_x_l, h_l_gmu, h_l_gmu)
            #---------
            #h_l_with_v2a = self.trans_l_with_v2a(proj_x_l, h_a_with_vs, h_a_with_vs)    # Dimension (L, N, d_l)
            #h_l_with_a2v = self.trans_l_with_a2v(proj_x_l, h_v_with_as, h_v_with_as)    # Dimension (L, N, d_l)
            #print(h_l_with_v2a.size(), h_l_with_a2v.size())
            #----------
            #sum_ls = self.proj_v2a(h_l_with_v2a) + self.proj_a2v(h_l_with_a2v)
            #h_ls = self.gmu_l(h_l_with_v2a, h_l_with_a2v, sum_ls)
            #---------
            #h_ls = self.proj_va2l(torch.cat([h_l_with_v2a, h_l_with_a2v], dim=2))
            #print(h_ls.size())
            h_ls = self.trans_l_mem(h_ls)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]   # Take the last output for prediction
            #print(last_h_l.size())

        if self.aonly:
            # (L,V) --> A
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
            
            h_a_gmu = self.gmu_a(h_v_with_ls, h_l_with_vs)
            
            h_as = self.trans_a_with_l2v(proj_x_a, h_a_gmu, h_a_gmu)
            #---------
            #h_a_with_v2l = self.trans_a_with_v2l(proj_x_a, h_l_with_vs, h_l_with_vs)
            #h_a_with_l2v = self.trans_a_with_l2v(proj_x_a, h_v_with_ls, h_v_with_ls)
            #print(h_a_with_l2v.size(), h_a_with_v2l.size())
            #--------
            #sum_as = self.proj_l2v(h_a_with_l2v) + self.proj_v2l(h_a_with_v2l)
            #h_as = self.gmu_a(h_a_with_l2v, h_a_with_v2l, sum_as)
            #---------
            #h_as = self.proj_vl2a(torch.cat([h_a_with_l2v, h_a_with_v2l], dim=2))
            #print(h_as.size())
            h_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]
            #print(last_h_a.size())

        if self.vonly:
            # (L,A) --> V
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
            
            h_v_gmu = self.gmu_v(h_l_with_as, h_a_with_ls)
            
            h_vs = self.trans_v_with_l2a(proj_x_v, h_v_gmu, h_v_gmu)
            #---------
            #h_v_with_a2l = self.trans_v_with_a2l(proj_x_v, h_l_with_as, h_l_with_as)
            #h_v_with_l2a = self.trans_v_with_l2a(proj_x_v, h_a_with_ls, h_a_with_ls)
            #print(h_v_with_l2a.size(), h_v_with_a2l.size())
            #-------
            #sum_vs = self.proj_l2a(h_v_with_l2a) + self.proj_a2l(h_v_with_a2l)
            #h_vs = self.gmu_v(h_v_with_l2a, h_v_with_a2l, sum_vs)
            #---------
            #h_vs = self.proj_la2v(torch.cat([h_v_with_l2a, h_v_with_a2l], dim=2))
            #print(h_vs.size())
            h_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]
            #print(last_h_v.size())
        
        #last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a)
        last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a, self.proj_poster(poster)) #, h_early.squeeze(0))
        #print(last_hs.size())
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)
        return output
