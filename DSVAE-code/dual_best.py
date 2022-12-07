import numpy as np

import torch
from torch import optim

import random
from copy import deepcopy
from mask import *
from utils import get_data, ndcg, recall
from model_dual import VAE
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='./douban',type=str)
parser.add_argument('--hidden-dim', type=int, default=600)
parser.add_argument('--latent-dim', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=128)#256)
parser.add_argument('--beta', type=float, default=None)
parser.add_argument('--gamma', type=float, default=0.005)
parser.add_argument('--lr', type=float, default=1e-3)#5e-4
parser.add_argument('--n-epochs', type=int, default=20)
parser.add_argument('--n-enc_epochs', type=int, default=3)
parser.add_argument('--n-dec_epochs', type=int, default=1)
parser.add_argument('--not-alternating', type=bool, default=False)
args = parser.parse_args()

seed = 123#1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cpu")#("cuda:")

data = get_data(args.dataset)
train_data, valid_in_data, valid_out_data, test_in_data, test_out_data = data

data_all = train_data.toarray()
data_sub1 = train_data.toarray()
data_sunb2 = train_data.toarray()
train_mask1, train_mask2 = mask_percent(data_all, data_sub1,data_sunb2, 0.5)



def generate(batch_size, device, data_in, sub1=None, sub2=None, data_out=None, shuffle=False, samples_perc_per_epoch=1):
    assert 0 < samples_perc_per_epoch <= 1
    
    total_samples = data_in.shape[0]
    samples_per_epoch = int(total_samples * samples_perc_per_epoch)
    
    if shuffle:
        idxlist = np.arange(total_samples)
        np.random.shuffle(idxlist)
        idxlist = idxlist[:samples_per_epoch]
    else:
        idxlist = np.arange(samples_per_epoch)
    
    for st_idx in range(0, samples_per_epoch, batch_size):
        end_idx = min(st_idx + batch_size, samples_per_epoch)
        idx = idxlist[st_idx:end_idx]

        yield Batch(device, idx, data_in, sub1, sub2, data_out)


class Batch:
    def __init__(self, device, idx, data_in, data_sub1=None, data_sub2=None, data_out=None):
        self._device = device
        self._idx = idx
        self._data_in = data_in
        self._data_out = data_out
        self._data_sub1 = data_sub1
        self._data_sub2 = data_sub2
        
    
    def get_idx(self):
        return self._idx
    
    def get_idx_to_dev(self):
        return torch.LongTensor(self.get_idx()).to(self._device)
        
    def get_ratings(self, is_out=False):
        data = self._data_out if is_out else self._data_in
        return data[self._idx]
    def get_sub(self):
        sub1 = self._data_sub1
        sub2 = self._data_sub2
        #print(self._idx,'self._idx',self._idx.shape[0])
        return torch.Tensor(sub1[self._idx]).to(self._device), torch.Tensor(sub2[self._idx]).to(self._device), torch.Tensor(np.arange(self._idx.shape[0])).to(self._device)  
  
    
    def get_ratings_to_dev(self, is_out=False):
        return torch.Tensor(
            self.get_ratings(is_out).toarray()
        ).to(self._device)


def evaluate(model, data_in, data_out, sub1, sub2, n_users, metrics, samples_perc_per_epoch=1, batch_size=1000):
    metrics = deepcopy(metrics)
    model.eval()
    
    for m in metrics:
        m['score'] = []
    
    for batch in generate(batch_size=batch_size,
                          device=device,
                          data_in=data_in,
                          sub1=sub1, 
                          sub2=sub2,
                          data_out=data_out,
                          samples_perc_per_epoch=samples_perc_per_epoch
                         ):
        
        ratings_in = batch.get_ratings_to_dev()
        sub1, sub2, n_users = batch.get_sub()
        ratings_out = batch.get_ratings(is_out=True)
    
        ratings_pred = model(ratings_in, sub1, sub2, n_users, calculate_loss=False).cpu().detach().numpy()
        
        if not (data_in is data_out):
            ratings_pred[batch.get_ratings().nonzero()] = -np.inf
            
        for m in metrics:
            m['score'].append(m['metric'](ratings_pred, ratings_out, k=m['k']))

    for m in metrics:
        m['score'] = np.concatenate(m['score'])#.mean()
        m['score'] = np.nanmean( m['score'])
        
    return [x['score'] for x in metrics]


def run(model, opts, train_data, data_sub1, data_sub2, batch_size, n_epochs, beta, gamma, dropout_rate):
    model.train()
    for epoch in range(n_epochs):
        for batch in generate(batch_size=batch_size, device=device, data_in=train_data, sub1=data_sub1, sub2=data_sub2, shuffle=True):
            ratings = batch.get_ratings_to_dev()
            sub1, sub2, n_users = batch.get_sub()
            for optimizer in opts:
                optimizer.zero_grad()
                
            _, loss = model(ratings, sub1, sub2, n_users, beta=beta, gamma=gamma, dropout_rate=dropout_rate)
            loss.backward()
            
            for optimizer in opts:
                optimizer.step()


model_kwargs = {
    'hidden_dim': args.hidden_dim,
    'latent_dim': args.latent_dim,
    'input_dim': train_data.shape[1],
    'ssl_reg': 0.01,#0.01
    'ssl_temp': 0.5,#0.2
    'cont_temp': 0.2#0.2
}
metrics = [{'metric': ndcg, 'k': 25}]

best_ndcg = -np.inf
train_scores, valid_scores = [], []

model = VAE(**model_kwargs).to(device)
model_best = VAE(**model_kwargs).to(device)
 
learning_kwargs = {
    'model': model,
    'train_data': train_data,
    'data_sub1':train_mask1, 
    'data_sub2':train_mask2,
    'batch_size': args.batch_size,
    'beta': args.beta,
    'gamma': args.gamma
}

decoder_params = set(model.decoder.parameters())
encoder_params = set(model.encoder.parameters())

optimizer_encoder = optim.Adam(encoder_params, lr=args.lr)
optimizer_decoder = optim.Adam(decoder_params, lr=args.lr)


for epoch in range(args.n_epochs):

    if args.not_alternating:
        run(opts=[optimizer_encoder, optimizer_decoder], n_epochs=1, dropout_rate=0.5, **learning_kwargs)
    else:
        run(opts=[optimizer_encoder], n_epochs=args.n_enc_epochs, dropout_rate=0.5, **learning_kwargs)
        model.update_prior()
        run(opts=[optimizer_decoder], n_epochs=args.n_dec_epochs, dropout_rate=0, **learning_kwargs)

    train_scores.append(
        evaluate(model, train_data, train_data, train_mask1, train_mask2, 1, metrics, 0.01)[0]
    )
    valid_scores.append(
        evaluate(model, valid_in_data, valid_out_data, train_mask1, train_mask2, 1, metrics, 1)[0]
    )
    
    if valid_scores[-1] > best_ndcg:
        best_ndcg = valid_scores[-1]
        model_best.load_state_dict(deepcopy(model.state_dict()))
        

    print(f'epoch {epoch} | valid ndcg@100: {valid_scores[-1]:.4f} | ' +
          f'best valid: {best_ndcg:.4f} | train ndcg@100: {train_scores[-1]:.4f}')


    
#test_metrics = [{'metric': ndcg, 'k': 100}, {'metric': recall, 'k': 20}, {'metric': recall, 'k': 50}]
test_metrics = [{'metric': ndcg, 'k': 25}, {'metric': ndcg, 'k': 50}, {'metric': recall, 'k': 25}, {'metric': recall, 'k': 50}]
final_scores = evaluate(model_best, test_in_data, test_out_data, train_mask1, train_mask2, 1, test_metrics)

for metric, score in zip(test_metrics, final_scores):
    print(f"{metric['metric'].__name__}@{metric['k']}:\t{score:.4f}")
print('ssl 0.01   mask 0.4 lr1e-3')








###################model
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
            queue = torch.randn(self.dim[l], 128)#.cuda()#128是负样本个数  256
            #print('shape k,q,queue',k.shape,q.shape,queue.shape)
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
            negative_elbo = -(mll - 1.*kld) + ssl + 0.1 * dual_loss
            
            return (mll, kld), negative_elbo
            
        else:
            return x_pred

    def update_prior(self):
        self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))
