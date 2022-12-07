####################################model
import numpy as np
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from sklearn import preprocessing

def swish(x):
    return x.mul(torch.sigmoid(x))

def log_norm_pdf(x, mu, logvar):
    return -0.5*(logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


class CompositePrior(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, mixture_weights=[3/20, 3/4, 1/10]):
        super(CompositePrior, self).__init__()
        
        self.mixture_weights = mixture_weights
        
        self.mu_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.mu_prior.data.fill_(0)
        
        self.logvar_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_prior.data.fill_(0)
        
        self.logvar_uniform_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_uniform_prior.data.fill_(10)
        
        self.encoder_old = Encoder(hidden_dim, latent_dim, input_dim)
        self.encoder_old.requires_grad_(False)
        
    def forward(self, x, sub1, sub2, z):
        post_mu, post_logvar,_,_,_,_ = self.encoder_old(x, sub1, sub2, 0)
        
        stnd_prior = log_norm_pdf(z, self.mu_prior, self.logvar_prior)
        post_prior = log_norm_pdf(z, post_mu, post_logvar)
        unif_prior = log_norm_pdf(z, self.mu_prior, self.logvar_uniform_prior)
        
        gaussians = [stnd_prior, post_prior, unif_prior]
        gaussians = [g.add(np.log(w)) for g, w in zip(gaussians, self.mixture_weights)]
        
        density_per_gaussian = torch.stack(gaussians, dim=-1)
                
        return torch.logsumexp(density_per_gaussian, dim=-1)

    
class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1):
        super(Encoder, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x, sub1, sub2, dropout_rate):

        norm = x.pow(2).sum(dim=-1).sqrt()
        x = x / norm[:, None]
    
        x = F.dropout(x, p=dropout_rate, training=self.training)
        
        h1 = self.ln1(swish(self.fc1(x)))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3)) 
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        
        norm_1 = sub1.pow(2).sum(dim=-1).sqrt()
        sub1 = sub1 / norm_1[:, None] 
        h1_sub1 = self.ln1(swish(self.fc1(sub1)))
        h2_sub1 = self.ln2(swish(self.fc2(h1_sub1) + h1_sub1))
        h3_sub1 = self.ln3(swish(self.fc3(h2_sub1) + h1_sub1 + h2_sub1))
        h4_sub1 = self.ln4(swish(self.fc4(h3_sub1) + h1_sub1 + h2_sub1 + h3_sub1))
        h5_sub1 = self.ln5(swish(self.fc5(h4_sub1) + h1_sub1 + h2_sub1 + h3_sub1 + h4_sub1))
               
        norm_2 = sub2.pow(2).sum(dim=-1).sqrt()
        sub2 = sub2 / norm_2[:, None] 
        h1_sub2 = self.ln1(swish(self.fc1(sub2)))
        h2_sub2 = self.ln2(swish(self.fc2(h1_sub2) + h1_sub2))
        h3_sub2 = self.ln3(swish(self.fc3(h2_sub2) + h1_sub2 + h2_sub2))
        h4_sub2 = self.ln4(swish(self.fc4(h3_sub2) + h1_sub2 + h2_sub2 + h3_sub2))
        h5_sub2 = self.ln5(swish(self.fc5(h4_sub2) + h1_sub2 + h2_sub2 + h3_sub2 + h4_sub2))
        
        return self.fc_mu(h5), self.fc_logvar(h5), self.fc_mu(h5_sub1), self.fc_logvar(h5_sub1), self.fc_mu(h5_sub2), self.fc_logvar(h5_sub2)

class Dual_Encoder(nn.Module):
    def __init__(self, hidden_dim, input_dim, eps=1e-1):
        super(Dual_Encoder, self).__init__()        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=eps)   
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=eps)  
                           
    def forward(self, real, recon):
        norm = real.pow(2).sum(dim=-1).sqrt()
        real = real / norm[:, None]
        h1 = self.ln1(self.fc1(real))
        h2 = self.ln2(self.fc2(h1) + h1)
        h3 = real
        
        h4 = self.ln1(self.fc1(recon))
        h5 = self.ln2(self.fc2(h4) + h4)
        h6 = recon
        return h2, h3, h5, h6

 

               

class dual_ssl_loss (nn.Module): 
    def __init__(self, cont_temp, hidden_dim):
        super(dual_ssl_loss, self).__init__()
        self.cont_temp = cont_temp
        self.dim = hidden_dim
        
    def forward(self, d_q, d_k):
        #total_cont = torch.tensor(0.0).cuda()
        #d_q = dual_encoder(im_q, mode="cont")
        for l in range(2):
            q = F.normalize(d_q[l], dim=1)
           # print('q',q,q.shape)
            k = d_k[l]
            queue = torch.randn(self.dim[l], 150).cuda()#80是负样本个数  
            
            l_pos = torch.einsum("nc,nc->n", [k,q]).unsqueeze(-1)
            l_neg = torch.einsum('nc,ck->nk', [q, queue.detach()])
            logits = torch.cat([l_pos, l_neg], dim=1).cuda() / self.cont_temp#0.07
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            cont_loss = nn.CrossEntropyLoss()(logits, labels) * 0.5
            cont_loss += cont_loss
        return cont_loss


class calc_ssl_loss (nn.Module): 
    def __init__(self, ssl_temp, ssl_reg):
        super(calc_ssl_loss, self).__init__()
         
        self.ssl_temp = ssl_temp
        self.ssl_reg = ssl_reg       
        
    def forward(self,ua_embeddings_sub1, ua_embeddings_sub2, users):
        users = users.int()
        user_emb1 = ua_embeddings_sub1.index_select(0, users)
        user_emb2 = ua_embeddings_sub2.index_select(0, users)
        
        normalize_user_emb1 = torch.nn.functional.normalize(user_emb1)
        normalize_user_emb2 = torch.nn.functional.normalize(user_emb2)  
     
        normalize_all_user_emb2 = torch.nn.functional.normalize(ua_embeddings_sub2)
        
        pos_score_user = torch.sum(torch.mul(normalize_user_emb1, normalize_user_emb2), dim=1)
        ttl_score_user = torch.matmul(normalize_user_emb1, normalize_all_user_emb2.t())

        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.sum(torch.exp(ttl_score_user / self.ssl_temp), axis=1)
        
        ssl_loss_user = -torch.sum(torch.log(pos_score_user / ttl_score_user))
        
        ssl_loss = self.ssl_reg * ssl_loss_user
        return ssl_loss
        

class VAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, ssl_temp, ssl_reg, cont_temp):
        super(VAE, self).__init__()

        self.encoder = Encoder(hidden_dim, latent_dim, input_dim)
        self.prior = CompositePrior(hidden_dim, latent_dim, input_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        self.ssl_loss = calc_ssl_loss(ssl_temp, ssl_reg)
        self.dualencoder = Dual_Encoder(hidden_dim, input_dim)
        self.dual_loss = dual_ssl_loss(cont_temp, [hidden_dim, input_dim])
        
    def reparameterize(self, mu, logvar, mu_sub1, logvar_sub1, mu_sub2, logvar_sub2):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)           
            std_1 = torch.exp(0.5*logvar_sub1)           
            std_2 = torch.exp(0.5*logvar_sub2)
            
            return eps.mul(std).add_(mu), eps.mul(std_1).add_(mu_sub1), eps.mul(std_2).add_(mu_sub2)
        else:
            return mu, mu_sub1, mu_sub2
    
    def forward(self, user_ratings, sub1, sub2, n_users, beta=None, gamma=1, dropout_rate=0.5, calculate_loss=True):
        
        mu, logvar, mu_sub1, logvar_sub1, mu_sub2, logvar_sub2 = self.encoder(user_ratings, sub1, sub2, dropout_rate=dropout_rate)  
        z, z_sub1,z_sub2 = self.reparameterize(mu, logvar, mu_sub1, logvar_sub1, mu_sub2, logvar_sub2)
        x_pred = self.decoder(z)
        h2, h3, h5, h6 = self.dualencoder(user_ratings, x_pred)
        
        if calculate_loss:
            if gamma:
                norm = user_ratings.sum(dim=-1)
                kl_weight = gamma * norm
            elif beta:
                kl_weight = beta

            mll = (F.log_softmax(x_pred, dim=-1) * user_ratings).sum(dim=-1).mean()
            kld = (log_norm_pdf(z, mu, logvar) - self.prior(user_ratings, sub1, sub2, z)).sum(dim=-1).mul(kl_weight).mean()
            ssl = self.ssl_loss(z_sub1, z_sub2, n_users)
            dual_loss = self.dual_loss([h5, h6], [h2, h3])
            negative_elbo = -(mll - 1.*kld) + ssl + 1.42 * dual_loss#0.05  0.2
            
            return (mll, kld), negative_elbo
            
        else:
            return x_pred

    def update_prior(self):
        self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))