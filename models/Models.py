import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import log, sqrt, log2, ceil, exp

class Model3(nn.Module):
    def __init__(self, vendors, cont_idx_hi=8, misc_idx_hi=12, ptag_idx_hi=55, vtag_idx_hi=123, d_fc=64):
        super(Model3, self).__init__()

        # for vendor lookup 
        self.vendor_lookup = nn.Embedding.from_pretrained(vendors)
        self.vendor_lookup.weight.requires_grad = False

        # indices for slicing inputs
        self.cont_idx_hi = cont_idx_hi
        self.misc_idx_hi = misc_idx_hi
        self.ptag_idx_hi = ptag_idx_hi
        self.vtag_idx_hi = vtag_idx_hi
        
        # dimensions of slices
        d_cont = cont_idx_hi
        d_misc = misc_idx_hi - cont_idx_hi
        d_ptag = ptag_idx_hi - misc_idx_hi
        d_vtag = vtag_idx_hi - ptag_idx_hi

        # primary_tags embeddings
        d_emb_ptag = int(ceil(log2(d_ptag)))
        self.emb_ptag = nn.Linear(d_ptag, d_emb_ptag)

        # vendor_tag embeddings
        d_emb_vtag = int(ceil(log2(d_vtag)))
        self.emb_vtag = nn.Linear(d_vtag, d_emb_vtag)

        # customer and vendor embeddings
        d_emb = int(ceil(log2(d_cont+d_misc+d_emb_ptag+d_emb_vtag)))
        self.c_emb = nn.Linear(d_cont+d_misc+d_emb_ptag+d_emb_vtag, d_emb)
        self.v_emb = nn.Linear(d_cont+d_misc+d_emb_ptag+d_emb_vtag, d_emb)

        # dense layers
        self.fc1 = nn.Linear(2 * d_emb, d_fc)
        self.fc1_bn = nn.BatchNorm1d(d_fc)
        self.fc2 = nn.Linear(d_fc, d_fc // 2)
        self.fc2_bn = nn.BatchNorm1d(d_fc // 2)
        self.fc3 = nn.Linear(d_fc // 2, d_fc // 4)
        self.fc3_bn = nn.BatchNorm1d(d_fc // 4)
        self.fc4 = nn.Linear(d_fc // 4, 1)

        self.dropout = nn.Dropout(p=0.5)


    def forward(self, c_seq, v_id):
        # lookup customer and vendor representations
        vendor = self.vendor_lookup(v_id)
        customer = torch.mean(self.vendor_lookup(c_seq), axis=1)     # correct axis?

        # split customer
        c_cont = customer[:, : self.cont_idx_hi]
        c_misc = customer[:, self.cont_idx_hi : self.misc_idx_hi]
        c_ptag = customer[:, self.misc_idx_hi : self.ptag_idx_hi]
        c_vtag = customer[:, self.ptag_idx_hi :]

        # split vendor
        v_cont = vendor[:, : self.cont_idx_hi]
        v_misc = vendor[:, self.cont_idx_hi : self.misc_idx_hi]
        v_ptag = vendor[:, self.misc_idx_hi : self.ptag_idx_hi]
        v_vtag = vendor[:, self.ptag_idx_hi :]

        # embed ptags
        c_ptag = self.emb_ptag(c_ptag.float())
        v_ptag = self.emb_ptag(v_ptag.float())

        # embed vtags
        c_vtag = self.emb_vtag(c_vtag.float())
        v_vtag = self.emb_vtag(v_vtag.float())

        # embed customer
        customer = torch.cat((c_cont, c_misc, c_ptag, c_vtag), axis=1)
        customer = self.c_emb(customer.float())

        # embed vendor
        vendor = torch.cat((v_cont, v_misc, v_ptag, v_vtag), axis=1)
        vendor = self.v_emb(vendor.float())

        # feed through classifier
        out = torch.cat((customer, vendor), axis=1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc1_bn(out)
        out = F.elu(out)

        out = self.fc2(out)
        out = self.dropout(out)
        out = self.fc2_bn(out)
        out = F.elu(out)

        out = self.fc3(out)
        out = self.dropout(out)
        out = self.fc3_bn(out)
        out = F.elu(out)

        out = self.fc4(out)     # output is raw
        return out


class Model2(nn.Module):
    def __init__(self, vendors, cont_idx_hi=8, misc_idx_hi=12, ptag_idx_hi=55, vtag_idx_hi=123, d_fc=64):
        super(Model2, self).__init__()

        # for vendor lookup 
        self.vendor_lookup = nn.Embedding.from_pretrained(vendors)
        self.vendor_lookup.weight.requires_grad = False

        # indices for slicing inputs
        self.cont_idx_hi = cont_idx_hi
        self.misc_idx_hi = misc_idx_hi
        self.ptag_idx_hi = ptag_idx_hi
        self.vtag_idx_hi = vtag_idx_hi
        
        # dimensions of slices
        d_cont = cont_idx_hi
        d_misc = misc_idx_hi - cont_idx_hi
        d_ptag = ptag_idx_hi - misc_idx_hi
        d_vtag = vtag_idx_hi - ptag_idx_hi

        # primary_tags embeddings
        d_emb_ptag = int(ceil(log2(d_ptag)))
        self.c_emb_ptag = nn.Linear(d_ptag, d_emb_ptag)
        self.c_emb_ptag_bn = nn.BatchNorm1d(d_emb_ptag)
        self.v_emb_ptag = nn.Linear(d_ptag, d_emb_ptag)
        self.v_emb_ptag_bn = nn.BatchNorm1d(d_emb_ptag)

        # vendor_tag embeddings
        d_emb_vtag = int(ceil(log2(d_vtag)))
        self.c_emb_vtag = nn.Linear(d_vtag, d_emb_vtag)
        self.c_emb_vtag_bn = nn.BatchNorm1d(d_emb_vtag)
        self.v_emb_vtag = nn.Linear(d_vtag, d_emb_vtag)
        self.v_emb_vtag_bn = nn.BatchNorm1d(d_emb_vtag)

        # customer and vendor embeddings
        d_emb = int(ceil(log2(d_cont+d_misc+d_emb_ptag+d_emb_vtag)))
        self.c_emb = nn.Linear(d_cont+d_misc+d_emb_ptag+d_emb_vtag, d_emb)
        self.c_emb_bn = nn.BatchNorm1d(d_emb)
        self.v_emb = nn.Linear(d_cont+d_misc+d_emb_ptag+d_emb_vtag, d_emb)
        self.v_emb_bn = nn.BatchNorm1d(d_emb)

        # dense layers
        self.fc1 = nn.Linear(2 * d_emb, d_fc)
        self.fc1_bn = nn.BatchNorm1d(d_fc)
        self.fc2 = nn.Linear(d_fc, d_fc // 2)
        self.fc2_bn = nn.BatchNorm1d(d_fc // 2)
        self.fc3 = nn.Linear(d_fc // 2, d_fc // 4)
        self.fc3_bn = nn.BatchNorm1d(d_fc // 4)
        self.fc4 = nn.Linear(d_fc // 4, 1)

        self.dropout = nn.Dropout(p=0.5)


    def forward(self, c_seq, v_id):
        # lookup customer and vendor representations
        vendor = self.vendor_lookup(v_id)
        customer = torch.sum(self.vendor_lookup(c_seq), axis=1)     # correct axis?

        # split customer
        c_cont = customer[:, : self.cont_idx_hi]
        c_misc = customer[:, self.cont_idx_hi : self.misc_idx_hi]
        c_ptag = customer[:, self.misc_idx_hi : self.ptag_idx_hi]
        c_vtag = customer[:, self.ptag_idx_hi :]

        # split vendor
        v_cont = vendor[:, : self.cont_idx_hi]
        v_misc = vendor[:, self.cont_idx_hi : self.misc_idx_hi]
        v_ptag = vendor[:, self.misc_idx_hi : self.ptag_idx_hi]
        v_vtag = vendor[:, self.ptag_idx_hi :]

        # embed ptags
        c_ptag = self.c_emb_ptag(c_ptag.float())
        c_ptag = self.dropout(c_ptag)
        c_ptag = self.c_emb_ptag_bn(c_ptag)
        c_ptag = F.elu(c_ptag)

        v_ptag = self.v_emb_ptag(v_ptag.float())
        v_ptag = self.dropout(v_ptag)
        v_ptag = self.v_emb_ptag_bn(v_ptag)
        v_ptag = F.elu(v_ptag)

        # embed vtags
        c_vtag = self.c_emb_vtag(c_vtag.float())
        c_vtag = self.dropout(c_vtag)
        c_vtag = self.c_emb_vtag_bn(c_vtag)
        c_vtag = F.elu(c_vtag)

        v_vtag = self.v_emb_vtag(v_vtag.float())
        v_vtag = self.dropout(v_vtag)
        v_vtag = self.v_emb_vtag_bn(v_vtag)
        v_vtag = F.elu(v_vtag)

        # embed customer
        customer = torch.cat((c_cont, c_misc, c_ptag, c_vtag), axis=1)
        customer = self.c_emb(customer.float())
        customer = self.dropout(customer)
        customer = self.c_emb_bn(customer)
        customer = F.elu(customer)

        # embed vendor
        vendor = torch.cat((v_cont, v_misc, v_ptag, v_vtag), axis=1)
        vendor = self.v_emb(vendor.float())
        vendor = self.dropout(vendor)
        vendor = self.v_emb_bn(vendor)
        vendor = F.elu(vendor)

        # feed through classifier
        out = torch.cat((customer, vendor), axis=1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc1_bn(out)
        out = F.elu(out)

        out = self.fc2(out)
        out = self.dropout(out)
        out = self.fc2_bn(out)
        out = F.elu(out)

        out = self.fc3(out)
        out = self.dropout(out)
        out = self.fc3_bn(out)
        out = F.elu(out)

        out = self.fc4(out)     # output is raw
        return out


class Model1(nn.Module):
    def __init__(self, vendors, cont_idx_hi=8, misc_idx_hi=12, ptag_idx_hi=55, vtag_idx_hi=123, d_fc=64):
        super(Model1, self).__init__()

        # for vendor lookup 
        self.vendor_lookup = nn.Embedding.from_pretrained(vendors)
        self.vendor_lookup.weight.requires_grad = False

        # indices for slicing inputs
        self.cont_idx_hi = cont_idx_hi
        self.misc_idx_hi = misc_idx_hi
        self.ptag_idx_hi = ptag_idx_hi
        self.vtag_idx_hi = vtag_idx_hi
        
        # dimensions of slices
        d_cont = cont_idx_hi
        d_misc = misc_idx_hi - cont_idx_hi
        d_ptag = ptag_idx_hi - misc_idx_hi
        d_vtag = vtag_idx_hi - ptag_idx_hi

        # primary_tags embeddings
        d_emb_ptag = int(ceil(log2(d_ptag)))
        self.c_emb_ptag = nn.Linear(d_ptag, d_emb_ptag)
        self.v_emb_ptag = nn.Linear(d_ptag, d_emb_ptag)

        # vendor_tag embeddings
        d_emb_vtag = int(ceil(log2(d_vtag)))
        self.c_emb_vtag = nn.Linear(d_vtag, d_emb_vtag)
        self.v_emb_vtag = nn.Linear(d_vtag, d_emb_vtag)

        # customer and vendor embeddings
        d_emb = int(ceil(log2(d_cont+d_misc+d_emb_ptag+d_emb_vtag)))
        self.c_emb = nn.Linear(d_cont+d_misc+d_emb_ptag+d_emb_vtag, d_emb)
        self.v_emb = nn.Linear(d_cont+d_misc+d_emb_ptag+d_emb_vtag, d_emb)

        # dense layers
        self.fc1 = nn.Linear(2 * d_emb, d_fc)
        self.fc2 = nn.Linear(d_fc, d_fc // 2)
        self.fc3 = nn.Linear(d_fc // 2, d_fc // 4)
        self.fc4 = nn.Linear(d_fc // 4, 1)


    def forward(self, c_seq, v_id):
        # lookup customer and vendor representations
        vendor = self.vendor_lookup(v_id)
        customer = torch.sum(self.vendor_lookup(c_seq), axis=1)     # correct axis?

        # split customer
        c_cont = customer[:, : self.cont_idx_hi]
        c_misc = customer[:, self.cont_idx_hi : self.misc_idx_hi]
        c_ptag = customer[:, self.misc_idx_hi : self.ptag_idx_hi]
        c_vtag = customer[:, self.ptag_idx_hi :]

        # split vendor
        v_cont = vendor[:, : self.cont_idx_hi]
        v_misc = vendor[:, self.cont_idx_hi : self.misc_idx_hi]
        v_ptag = vendor[:, self.misc_idx_hi : self.ptag_idx_hi]
        v_vtag = vendor[:, self.ptag_idx_hi :]

        # embed ptags
        c_ptag = self.c_emb_ptag(c_ptag.float())
        c_ptag = F.elu(c_ptag)

        v_ptag = self.v_emb_ptag(v_ptag.float())
        v_ptag = F.elu(v_ptag)

        # embed vtags
        c_vtag = self.c_emb_vtag(c_vtag.float())
        c_vtag = F.elu(c_vtag)

        v_vtag = self.v_emb_vtag(v_vtag.float())
        v_vtag = F.elu(v_vtag)

        # embed customer
        customer = torch.cat((c_cont, c_misc, c_ptag, c_vtag), axis=1)
        customer = self.c_emb(customer.float())
        customer = F.elu(customer)

        # embed vendor
        vendor = torch.cat((v_cont, v_misc, v_ptag, v_vtag), axis=1)
        vendor = self.v_emb(vendor.float())
        vendor = F.elu(vendor)

        # feed through classifier
        out = torch.cat((customer, vendor), axis=1)
        out = self.fc1(out)
        out = F.elu(out)

        out = self.fc2(out)
        out = F.elu(out)

        out = self.fc3(out)
        out = F.elu(out)

        out = self.fc4(out)     # output is raw
        return out