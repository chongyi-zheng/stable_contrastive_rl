# class VQ_VAE(nn.Module): #vae bottleneck, then quantize
#     def __init__(
#         self,
#         embedding_dim=3,
#         input_channels=3,
#         num_hiddens=128,
#         num_residual_layers=3,
#         num_residual_hiddens=64,
#         num_embeddings=512,
#         commitment_cost=0.25,
#         decoder_output_activation=None, #IGNORED FOR NOW
#         architecture=None, #IGNORED FOR NOW
#         imsize=48,
#         decay=0.0):
#         super(VQ_VAE, self).__init__()
#         self.imsize = imsize
#         self.embedding_dim = embedding_dim
#         self.pixel_cnn = None
#         self.input_channels = input_channels
#         self.imlength = imsize * imsize * input_channels
#         self.num_embeddings = num_embeddings
#         self.log_min_variance = float(np.log(1e-3))
#         self._encoder = Encoder(input_channels, num_hiddens,
#                                 num_residual_layers,
#                                 num_residual_hiddens)
#         self.f_mu = nn.Conv2d(in_channels=num_hiddens,
#                                       out_channels=1,
#                                       kernel_size=1,
#                                       stride=1)
#         self.f_logvar = nn.Conv2d(in_channels=num_hiddens,
#                                       out_channels=1,
#                                       kernel_size=1,
#                                       stride=1)
#         self.f_dec = nn.Conv2d(in_channels=1,
#                                       out_channels=self.embedding_dim,
#                                       kernel_size=1,
#                                       stride=1)

#         if decay > 0.0:
#             self._vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
#                                               commitment_cost, decay)
#         else:
#             self._vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
#                                            commitment_cost)
#         self._decoder = Decoder(self.embedding_dim,
#                                 num_hiddens,
#                                 num_residual_layers,
#                                 num_residual_hiddens)
        
#         #Calculate latent sizes
#         if imsize == 32:
#             self.root_len = 8
#         elif imsize == 36:
#             self.root_len = 9
#         elif imsize == 48:
#             self.root_len = 12
#         elif imsize == 84:
#             self.root_len = 21
#         else:
#             raise ValueError(imsize)

#         self.discrete_size = self.root_len * self.root_len
#         self.representation_size = self.discrete_size * self.embedding_dim
#         #Calculate latent sizes
#         self.tanh = nn.Tanh()

#     def compute_loss(self, inputs):
#         inputs = inputs.view(-1,
#                             self.input_channels,
#                             self.imsize,
#                             self.imsize)

#         vq_loss, quantized, perplexity, encodings, kle = self.quantize_image(inputs)
#         recon = self.decode(quantized)
        
#         recon_error = F.mse_loss(recon, inputs, reduction='sum')
#         return vq_loss, recon, perplexity, recon_error, kle

#     def quantize_image(self, inputs):
#         inputs = inputs.view(-1,
#                             self.input_channels,
#                             self.imsize,
#                             self.imsize)

#         z = self._encoder(inputs)
#         mu = self.f_mu(z)
#         logvar = self.log_min_variance + torch.abs(self.f_logvar(z))
#         kle = self.kl_divergence(mu, logvar)

#         if self.training:
#           z_s = self.rsample(mu, logvar)
#         else:
#           z_s = mu

#         z = self.f_dec(z_s)
#         vq_loss, quantized, perplexity, encodings = self._vq_vae(z)
#         return vq_loss, quantized, perplexity, encodings, kle

#     def encode(self, inputs, cont=True):
#         _, quantized, _, encodings, _ = self.quantize_image(inputs)

#         if cont:
#             return quantized.reshape(-1, self.representation_size)
#         return encodings.reshape(-1, self.discrete_size)

#     def latent_to_square(self, latents):
#         return latents.reshape(-1, self.root_len, self.root_len)

#     def kl_divergence(self, mu, logvar):
#         mu = mu.reshape(-1, self.discrete_size)
#         logvar = logvar.reshape(-1, self.discrete_size)
#         return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

#     def rsample(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def discrete_to_cont(self, e_indices):
#         e_indices = self.latent_to_square(e_indices)
#         input_shape = e_indices.shape + (self.embedding_dim,)
#         e_indices = e_indices.reshape(-1).unsqueeze(1)

#         min_encodings = torch.zeros(e_indices.shape[0], self.num_embeddings, device=e_indices.device)
#         min_encodings.scatter_(1, e_indices, 1)

#         e_weights = self._vq_vae._embedding.weight
#         quantized = torch.matmul(
#             min_encodings, e_weights).view(input_shape)

#         z_q = torch.matmul(min_encodings, e_weights).view(input_shape)
#         z_q = z_q.permute(0, 3, 1, 2).contiguous()
#         return z_q

#     def set_pixel_cnn(self, pixel_cnn):
#         self.pixel_cnn = pixel_cnn

#     def decode(self, latents, cont=True):
#         if cont:
#             z_q = latents.reshape(-1, self.embedding_dim, self.root_len, self.root_len)
#         else:
#             z_q = self.discrete_to_cont(latents)

#         #z_q = self.tanh(z_q)

#         return self._decoder(z_q)

#     def sample_prior(self, batch_size):
#         z_s = ptu.randn(batch_size, 1, self.root_len, self.root_len)
#         z = self.f_dec(z_s)
#         vq_loss, quantized, perplexity, encodings = self._vq_vae(z)
#         return quantized.reshape(-1, self.representation_size)


# class VQ_VAE(nn.Module):
#     def __init__(
#         self,
#         embedding_dim=3,
#         input_channels=3,
#         num_hiddens=128,
#         num_residual_layers=3,
#         num_residual_hiddens=64,
#         num_embeddings=512,
#         commitment_cost=0.25,
#         decoder_output_activation=None, #IGNORED FOR NOW
#         architecture=None, #IGNORED FOR NOW
#         imsize=48,
#         decay=0.0):
#         super(VQ_VAE, self).__init__()
#         self.imsize = imsize
#         self.embedding_dim = embedding_dim
#         self.pixel_cnn = None
#         self.input_channels = input_channels
#         self.imlength = imsize * imsize * input_channels
#         self.log_min_variance = float(np.log(1e-3))
#         self.num_embeddings = num_embeddings
#         self._encoder = Encoder(2 * input_channels, num_hiddens,
#                                 num_residual_layers,
#                                 num_residual_hiddens)
#         self._cond_encoder = Encoder(2 * input_channels, num_hiddens,
#                                 num_residual_layers,
#                                 num_residual_hiddens)
#         self.f_mu = nn.Conv2d(in_channels=num_hiddens,
#                                       out_channels=self.embedding_dim,
#                                       kernel_size=1,
#                                       stride=1)
#         self.f_logvar = nn.Conv2d(in_channels=num_hiddens,
#                                       out_channels=self.embedding_dim,
#                                       kernel_size=1,
#                                       stride=1)

#         if decay > 0.0:
#             self._vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
#                                               commitment_cost, decay)
#         else:
#             self._vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
#                                            commitment_cost)
#         self._decoder = Decoder(self.embedding_dim,
#                                 num_hiddens,
#                                 num_residual_layers,
#                                 num_residual_hiddens)

        
#         #Calculate latent sizes
#         if imsize == 32:
#             self.root_len = 8
#         elif imsize == 36:
#             self.root_len = 9
#         elif imsize == 48:
#             self.root_len = 12
#         elif imsize == 84:
#             self.root_len = 21
#         else:
#             raise ValueError(imsize)

#         self.discrete_size = self.root_len * self.root_len
#         self.representation_size = self.discrete_size * self.embedding_dim
#         self.logvar = nn.Parameter(torch.randn(1))
#         #Calculate latent sizes
#         self.tanh = nn.Tanh()

#     def compute_loss(self, inputs):
#         inputs = inputs.view(-1,
#                             self.input_channels,
#                             self.imsize,
#                             self.imsize)

#         vq_loss, quantized, perplexity, _ = self.quantize_image(inputs)
#         kle = self.kl_divergence()
#         recon = self.decode(quantized)
        
#         recon_error = F.mse_loss(recon, inputs, reduction='sum')
#         return vq_loss, kle, recon, perplexity, recon_error

#     def quantize_image(self, inputs):
#         inputs = inputs.view(-1,
#                             self.input_channels,
#                             self.imsize,
#                             self.imsize)

#         z = self._encoder(inputs)
#         z = self.f_mu(z)
#         vq_loss, quantized, perplexity, encodings = self._vq_vae(z)
#         mu = self.tanh(quantized)

#         if self.training:
#           z_s = self.rsample(mu)
#         else:
#           z_s = mu

#         return vq_loss, z_s, perplexity, encodings

#     def kl_divergence(self):
#         # import math
#         logvar = self.log_min_variance + torch.abs(self.logvar)
#         # std = torch.exp(0.5 * logvar)
#         # return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(std)
#         return - 0.5 * (1 + logvar - logvar.exp())

#     def encode(self, inputs, cont=True):
#         _, quantized, _, encodings = self.quantize_image(inputs)

#         if cont:
#             return quantized.reshape(-1, self.representation_size)
#         return encodings.reshape(-1, self.discrete_size)

#     def latent_to_square(self, latents):
#         return latents.reshape(-1, self.root_len, self.root_len)

#     # def rsample(self, mu):
#     #     logvar = self.log_min_variance + torch.abs(self.logvar)
#     #     stds = (0.5 * logvar).exp()
#     #     stds = stds.repeat(mu.shape[0], 1).reshape(*mu.size())
#     #     epsilon = ptu.randn(*mu.size())
#     #     latents = epsilon * stds + mu
#     #     return latents

#     def rsample(self, mu):
#         logvar = self.log_min_variance + torch.abs(self.logvar)
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std


#     def discrete_to_cont(self, e_indices):
#         e_indices = self.latent_to_square(e_indices)
#         input_shape = e_indices.shape + (self.embedding_dim,)
#         e_indices = e_indices.reshape(-1).unsqueeze(1)

#         min_encodings = torch.zeros(e_indices.shape[0], self.num_embeddings, device=e_indices.device)
#         min_encodings.scatter_(1, e_indices, 1)

#         e_weights = self._vq_vae._embedding.weight
#         quantized = torch.matmul(
#             min_encodings, e_weights).view(input_shape)

#         z_q = torch.matmul(min_encodings, e_weights).view(input_shape)
#         z_q = z_q.permute(0, 3, 1, 2).contiguous()
#         return z_q

#     def set_pixel_cnn(self, pixel_cnn):
#         self.pixel_cnn = pixel_cnn

#     def decode(self, latents, cont=True):
#         if cont:
#             z_q = latents.reshape(-1, self.embedding_dim, self.root_len, self.root_len)
#         else:
#             z_q = self.discrete_to_cont(latents)

#         return self._decoder(z_q)

#     def encode_one_np(self, inputs, cont=True):
#         return ptu.get_numpy(self.encode(ptu.from_numpy(inputs), cont=cont))[0]

#     def encode_np(self, inputs, cont=True):
#         return ptu.get_numpy(self.encode(ptu.from_numpy(inputs), cont=cont))

#     def decode_one_np(self, inputs, cont=True):
#         return np.clip(ptu.get_numpy(self.decode(ptu.from_numpy(inputs).reshape(1, -1), cont=cont))[0], 0, 1)

#     def decode_np(self, inputs, cont=True):
#         return np.clip(ptu.get_numpy(self.decode(ptu.from_numpy(inputs), cont=cont)), 0, 1)


# class VQ_VAE(nn.Module):
#     def __init__(
#         self,
#         embedding_dim,
#         root_len=10,
#         input_channels=3,
#         num_hiddens=128,
#         num_residual_layers=3,
#         num_residual_hiddens=64,
#         num_embeddings=512,
#         decoder_output_activation=None, #IGNORED FOR NOW
#         architecture=None, #IGNORED FOR NOW
#         min_variance=1e-3,
#         commitment_cost=0.25,
#         imsize=48,
#         decay=0.0,
#         ):
#         super(VQ_VAE, self).__init__()
#         self.log_min_variance = float(np.log(min_variance))
#         self.imsize = imsize
#         self.embedding_dim = embedding_dim
#         self.input_channels = input_channels
#         self.imlength = imsize * imsize * input_channels
        
#         self._encoder = Encoder(input_channels, num_hiddens,
#                                 num_residual_layers,
#                                 num_residual_hiddens)
#         self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
#                                       out_channels=self.embedding_dim,
#                                       kernel_size=1,
#                                       stride=1)

#         self._decoder = Decoder(self.embedding_dim,
#                                 num_hiddens,
#                                 num_residual_layers,
#                                 num_residual_hiddens)

#         if decay > 0.0:
#             self._vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
#                                               commitment_cost, decay)
#         else:
#             self._vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
#                                            commitment_cost)
        
#         #Calculate latent sizes
#         if imsize == 48:
#             self.root_conv_size = 12
#         elif imsize == 84:
#             self.root_conv_size = 21
#         else:
#             raise ValueError(imsize)

#         self.conv_size = self.root_conv_size * self.root_conv_size * self.embedding_dim
#         self.root_len = root_len
#         self.discrete_size = root_len * root_len
#         self.representation_size = self.discrete_size * self.embedding_dim
#         #Calculate latent sizes

#         assert self.representation_size <= self.conv_size  # This is a bad idea (wrong bottleneck)

#         self.f_mu = nn.Linear(self.conv_size, self.representation_size)
#         self.f_logvar = nn.Linear(self.conv_size, self.representation_size)
#         self.f_dec = nn.Linear(self.representation_size, self.conv_size)

#         self.f_enc.weight.data.uniform_(-1e-3, 1e-3)
#         self.f_enc.bias.data.uniform_(-1e-3, 1e-3)
#         self.f_dec.weight.data.uniform_(-1e-3, 1e-3)
#         self.f_dec.bias.data.uniform_(-1e-3, 1e-3)

#     def compute_loss(self, obs):
#         obs = obs.view(-1,
#           self.input_channels,
#           self.imsize,
#           self.imsize)

#         vq_loss, quantized, perplexity, _ = self.encode_image(obs)

#         recon = self.decode(quantized)
#         recon_error = F.mse_loss(recon, obs)

#         return vq_loss, quantized, recon, perplexity, recon_error


#     def encode_image(self, obs):
#         obs = obs.view(-1,
#           self.input_channels,
#           self.imsize,
#           self.imsize)

#         z_conv = self._encoder(obs)
#         z_conv = self._pre_vq_conv(z_conv)

#         return self.compress(z_conv)

#     def compress(self, z_conv):
#         z_conv = z_conv.view(-1, self.conv_size)
#         z = self.f_enc(z_conv).view(-1, self.embedding_dim, self.root_len, self.root_len)
#         vq_loss, quantized, perplexity, encodings = self._vq_vae(z)
#         quantized = quantized.view(-1, self.representation_size)
        
#         return vq_loss, quantized, perplexity, encodings


#     def decompress(self, quantized):
#         z_conv = self.f_dec(quantized)
#         z_conv = z_conv.view(-1, self.embedding_dim, self.root_conv_size, self.root_conv_size)   
#         return z_conv


#     def encode(self, inputs, cont=True):
#         _, quantized, _, encodings = self.encode_image(inputs)

#         if cont:
#           return quantized.view(-1, self.representation_size)
#         return encodings.view(-1, self.discrete_size)

#     def sample_prior(self, batch_size):
#         z_s = ptu.randn(batch_size, self.representation_size)
#         return z_s

#     def decode(self, latents, cont=True):
#         if not cont:
#           latents = self.discrete_to_cont(latents)

#         z_conv = self.decompress(latents)

#         return self._decoder(z_conv)

#     def discrete_to_cont(self, e_indices):
#         e_indices = e_indices.reshape(-1, self.root_len, self.root_len)
#         input_shape = e_indices.shape + (self.embedding_dim,)
#         e_indices = e_indices.reshape(-1).unsqueeze(1)

#         min_encodings = torch.zeros(e_indices.shape[0], self.num_embeddings, device=e_indices.device)
#         min_encodings.scatter_(1, e_indices, 1)

#         e_weights = self._vq_vae._embedding.weight
#         quantized = torch.matmul(
#             min_encodings, e_weights).view(input_shape)

#         z_q = torch.matmul(min_encodings, e_weights).view(input_shape)
#         z_q = z_q.permute(0, 3, 1, 2).contiguous()
#         return z_q

class CVQVAE(nn.Module):
    def __init__(
        self,
        embedding_dim,
        root_len=10,
        input_channels=3,
        num_hiddens=128,
        num_residual_layers=3,
        num_residual_hiddens=64,
        num_embeddings=512,
        decoder_output_activation=None, #IGNORED FOR NOW
        architecture=None, #IGNORED FOR NOW
        min_variance=1e-3,
        commitment_cost=0.25,
        imsize=48,
        decay=0.0,
        ):
        super(CVQVAE, self).__init__()
        self.imsize = imsize
        self.embedding_dim = embedding_dim
        self.input_channels = input_channels
        self.imlength = imsize * imsize * input_channels
        
        self._encoder = Encoder(2 * input_channels, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self._cond_encoder = Encoder(input_channels, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=1,
                                      kernel_size=1,
                                      stride=1)
        self._cond_pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)

        self._decoder = Decoder(1 + self.embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                                           commitment_cost)
        
        #Calculate latent sizes
        if imsize == 48:
            self.root_conv_size = 12
        elif imsize == 84:
            self.root_conv_size = 21
        else:
            raise ValueError(imsize)

        self.root_len = root_len
        self.conv_size = self.root_conv_size * self.root_conv_size
        self.discrete_size = root_len * root_len
        self.latent_sizes = [self.discrete_size * self.embedding_dim, self.conv_size * self.embedding_dim]
        self.representation_size = sum(self.latent_sizes)
        #Calculate latent sizes

        assert self.latent_sizes[0] <= self.conv_size * self.embedding_dim  # This is a bad idea (wrong bottleneck)

        self.f_enc = nn.Linear(self.conv_size, self.latent_sizes[0])
        self.f_dec = nn.Linear(self.latent_sizes[0], self.conv_size)

        self.f_enc.weight.data.uniform_(-1e-3, 1e-3)
        #self.f_enc.bias.data.uniform_(-1e-3, 1e-3)
        self.f_dec.weight.data.uniform_(-1e-3, 1e-3)
        #self.f_dec.bias.data.uniform_(-1e-3, 1e-3)

    def compute_loss(self, x_delta, x_cond):
        x_delta = x_delta.view(-1,
          self.input_channels,
          self.imsize,
          self.imsize)

        vq_loss, quantized, perplexity, _ = self.encode_images(x_delta, x_cond)

        recon = self.decode(quantized)
        recon_error = F.mse_loss(recon, x_delta)

        return vq_loss, quantized, recon, perplexity, recon_error


    def encode_images(self, x_delta, x_cond):
        x_delta = x_delta.view(-1,
          self.input_channels,
          self.imsize,
          self.imsize)

        x_cond = x_cond.view(-1,
          self.input_channels,
          self.imsize,
          self.imsize)
        
        x_delta = torch.cat([x_delta, x_cond], dim=1)
        
        z_delta = self._encoder(x_delta)
        z_delta = self._pre_vq_conv(z_delta)

        z_cond = self._cond_encoder(x_cond)
        z_cond = self._cond_pre_vq_conv(z_cond)

        return self.compress(z_delta, z_cond)

    def compress(self, z_delta, z_cond):
        z_delta = z_delta.view(-1, self.conv_size)
        z_cond = z_cond.view(-1, self.conv_size * self.embedding_dim)
        
        z_delta = self.f_enc(z_delta)
        z_delta = z_delta.view(-1, self.embedding_dim, self.root_len, self.root_len)
        
        vq_loss, quantized, perplexity, encodings = self._vq_vae(z_delta)
        quantized = quantized.view(-1, self.latent_sizes[0])

        quantized = torch.cat([quantized, z_cond], dim=1)
        
        return vq_loss, quantized, perplexity, encodings

    def decompress(self, quantized):
        z_delta = quantized[:, :self.latent_sizes[0]]
        z_cond = quantized[:, self.latent_sizes[0]:]

        z_delta = self.f_dec(z_delta)
        z_delta = z_delta.view(-1, 1, self.root_conv_size, self.root_conv_size)
        z_cond = z_cond.view(-1, self.embedding_dim, self.root_conv_size, self.root_conv_size)

        z_conv = torch.cat([z_delta, z_cond], dim=1)
        return z_conv


    def encode(self, x_delta, x_cond, cont=True):
        batch_size = x_delta.shape[0]
        _, quantized, _, encodings = self.encode_images(x_delta, x_cond)

        if cont:
          return quantized.view(batch_size, -1)
        return encodings.view(batch_size, -1)

    def sample_prior(self, batch_size):
        z_s = ptu.randn(batch_size, self.latent_sizes[0])
        return z_s

    def decode(self, latents, cont=True):
        if not cont:
          return 1/0
          latents = self.discrete_to_cont(latents)

        z_conv = self.decompress(latents)

        return self._decoder(z_conv)

    def discrete_to_cont(self, e_indices):
        e_indices = e_indices.reshape(-1, self.root_len, self.root_len)
        input_shape = e_indices.shape + (self.embedding_dim,)
        e_indices = e_indices.reshape(-1).unsqueeze(1)

        min_encodings = torch.zeros(e_indices.shape[0], self.num_embeddings, device=e_indices.device)
        min_encodings.scatter_(1, e_indices, 1)

        e_weights = self._vq_vae._embedding.weight
        quantized = torch.matmul(
            min_encodings, e_weights).view(input_shape)

        z_q = torch.matmul(min_encodings, e_weights).view(input_shape)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q


class VQ_VAE1(nn.Module):
    def __init__(
        self,
        embedding_dim,
        input_channels=3,
        num_hiddens=128,
        num_residual_layers=3,
        num_residual_hiddens=64,
        num_embeddings=512,
        commitment_cost=0.25,
        decoder_output_activation=None, #IGNORED FOR NOW
        architecture=None, #IGNORED FOR NOW
        imsize=48,
        decay=0.0):
        super(VQ_VAE, self).__init__()
        self.imsize = imsize
        self.embedding_dim = embedding_dim
        self.pixel_cnn = None
        self.input_channels = input_channels
        self.imlength = imsize * imsize * input_channels
        self.num_embeddings = num_embeddings
        self._encoder = Encoder(input_channels, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                                           commitment_cost)
        self._decoder = Decoder(self.embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        
        #Calculate latent sizes
        if imsize == 48: self.root_len = 12
        elif imsize == 84: self.root_len = 21
        else: raise ValueError(imsize)

        self.discrete_size = self.root_len * self.root_len
        self.representation_size = self.discrete_size * self.embedding_dim
        #Calculate latent sizes

    def compute_loss(self, inputs):
        inputs = inputs.view(-1,
                            self.input_channels,
                            self.imsize,
                            self.imsize)
        z = self._encoder(inputs)
        z = self._pre_vq_conv(z)
        vq_loss, quantized, perplexity, _ = self._vq_vae(z)

        x_recon = self._decoder(quantized)
        recon_error = F.mse_loss(x_recon, inputs)
        return vq_loss, quantized, x_recon, perplexity, recon_error


    def latent_to_square(self, latents):
        #root_len = int(latents.shape[1] ** 0.5)
        return latents.reshape(-1, self.root_len, self.root_len)

    def encode(self, inputs, cont=True):
        inputs = inputs.view(-1,
                            self.input_channels,
                            self.imsize,
                            self.imsize)
        z = self._encoder(inputs)
        z = self._pre_vq_conv(z)
        _, quantized, _, encodings = self._vq_vae(z)

        if cont:
            return quantized.reshape(inputs.shape[0], -1)

        return encodings.reshape(inputs.shape[0], -1)


    def encode_one_np(self, inputs, cont=True):
        return ptu.get_numpy(self.encode(ptu.from_numpy(inputs), cont=cont))[0]


    def encode_np(self, inputs, cont=True):
        return ptu.get_numpy(self.encode(ptu.from_numpy(inputs), cont=cont))

    def decode_one_np(self, inputs, cont=True):
        return np.clip(ptu.get_numpy(self.decode(ptu.from_numpy(inputs).reshape(1, -1), cont=cont))[0], 0, 1)


    def decode_np(self, inputs, cont=True):
        return np.clip(ptu.get_numpy(self.decode(ptu.from_numpy(inputs), cont=cont)), 0, 1)


    def discrete_to_cont(self, e_indices):
        e_indices = self.latent_to_square(e_indices)
        input_shape = e_indices.shape + (self.embedding_dim,)
        e_indices = e_indices.reshape(-1).unsqueeze(1)

        min_encodings = torch.zeros(e_indices.shape[0], self.num_embeddings, device=e_indices.device)
        min_encodings.scatter_(1, e_indices, 1)

        e_weights = self._vq_vae._embedding.weight
        quantized = torch.matmul(
            min_encodings, e_weights).view(input_shape)

        z_q = torch.matmul(min_encodings, e_weights).view(input_shape)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

    def set_pixel_cnn(self, pixel_cnn):
        self.pixel_cnn = pixel_cnn


    def decode(self, latents, cont=True):
        z_q = None
        if cont:
            z_q = latents.reshape(-1, self.embedding_dim, self.root_len, self.root_len)
        else:
            z_q = self.discrete_to_cont(latents)

        return self._decoder(z_q)


    def get_distance(self, s_indices, g_indices):
        assert s_indices.shape == g_indices.shape
        batch_size = s_indices.shape[0]
        s_q = self.discrete_to_cont(s_indices).reshape(batch_size, -1)
        g_q = self.discrete_to_cont(g_indices).reshape(batch_size, -1)
        return ptu.get_numpy(torch.norm(s_q - g_q, dim=1))

    def sample_prior(self, batch_size, cont=True):
        e_indices = self.pixel_cnn.generate(shape=(self.root_len, elf.root_len), batch_size=batch_size)
        e_indices = e_indices.reshape(batch_size, -1)
        if cont:
            return self.discrete_to_cont(e_indices)
        return e_indices

    def logprob(self, images, cont=True):
        batch_size = images.shape[0]
        root_len = int((self.representation_size // self.embedding_dim)**0.5)
        e_indices = self.encode(images, cont=False)
        e_indices = e_indices.reshape(batch_size, root_len, root_len)
        cond = ptu.from_numpy(np.ones((images.shape[0], 1)))
        logits = self.pixel_cnn(e_indices, cond)
        logits = logits.permute(0, 2, 3, 1).contiguous()
        criterion = nn.CrossEntropyLoss(reduction='none')#.cuda()

        logprob = - criterion(
            logits.view(-1, self.num_embeddings),
            e_indices.contiguous().view(-1))

        logprob = logprob.reshape(batch_size, -1).mean(dim=1)

        return logprob


class CVQVAENormal(nn.Module):
    def __init__(
        self,
        embedding_dim,
        input_channels=3,
        num_hiddens=128,
        num_residual_layers=3,
        num_residual_hiddens=64,
        num_embeddings=512,
        commitment_cost=0.25,
        decoder_output_activation=None, #IGNORED FOR NOW
        architecture=None, #IGNORED FOR NOW
        imsize=48,
        decay=0):
        super(CVQVAE, self).__init__()
        self.imsize = imsize
        self.embedding_dim = embedding_dim
        self.pixel_cnn = None
        self.input_channels = input_channels
        self.imlength = imsize * imsize * input_channels
        self.num_embeddings = num_embeddings

        #Calculate latent sizes
        if imsize == 48: self.root_len = 12
        elif imsize == 84: self.root_len = 21
        else: raise ValueError(imsize)

        self.discrete_size = self.root_len * self.root_len
        self.latent_sizes = [self.discrete_size * self.embedding_dim, self.discrete_size * self.embedding_dim]
        self.representation_size = sum(self.latent_sizes)
        #Calculate latent sizes

        self._encoder = Encoder(input_channels * 2, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self.cond_encoder = Encoder(input_channels, num_hiddens,
                                    num_residual_layers,
                                    num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)

        self.cond_pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
                                              commitment_cost, decay)
            self.cond_vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                                           commitment_cost)

            self.cond_vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                                              commitment_cost)

        self._decoder = Decoder(self.embedding_dim * 2,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self.cond_decoder = Decoder(self.embedding_dim,
                                    num_hiddens,
                                    num_residual_layers,
                                    num_residual_hiddens)

    def compute_loss(self, obs, cond):
        obs = obs.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        cond = cond.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        inputs = torch.cat([obs, cond], dim=1)
        
        z = self._encoder(inputs)
        z = self._pre_vq_conv(z)
        z_cond = self.cond_encoder(cond)
        z_cond = self.cond_pre_vq_conv(z_cond)

        vq_loss, quantized, perplexity, _ = self._vq_vae(z)
        cond_vq_loss, cond_quantized, cond_perplexity, _ = self.cond_vq_vae(z_cond)
        cat_quantized = torch.cat([quantized, cond_quantized], dim=1)
        
        cond_recon = self.cond_decoder(cond_quantized)
        x_recon = self._decoder(cat_quantized)
        vq_losses = [vq_loss, cond_vq_loss]
        perplexities = [perplexity, cond_perplexity]
        recons = [x_recon, cond_recon]
        errors = [F.mse_loss(x_recon, obs), F.mse_loss(cond_recon, cond)]
        #errors = [F.mse_loss(x_recon, obs), F.mse_loss(cond_recon, cond)]
        return vq_losses, perplexities, recons, errors


    def latent_to_square(self, latents):
        latents = latents.reshape(-1, 2, self.root_len, self.root_len)
        return latents[:, 0], latents[:, 1]

    def encode(self, obs, cond, cont=True):
        obs = obs.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        cond = cond.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        inputs = torch.cat([obs, cond], dim=1)
        z = self._encoder(inputs)
        z = self._pre_vq_conv(z)

        z_cond = self.cond_encoder(cond)
        z_cond = self.cond_pre_vq_conv(z_cond)

        _, quantized, _, encodings = self._vq_vae(z)
        _, cond_quantized, _, cond_encodings = self.cond_vq_vae(z_cond)

        if cont:
            z, z_c = quantized, cond_quantized
        else:
            z, z_c = encodings, cond_encodings
        

        z = z.reshape(obs.shape[0], -1)
        z_c = z_c.reshape(cond.shape[0], -1)
        z_cond = torch.cat([z, z_c], dim=1)
        return z_cond


    def conditioned_discrete_to_cont(self, e_indices):
        z_ind, cond_ind = self.latent_to_square(e_indices)
        z = self.discrete_to_cont(z_ind, self._vq_vae._embedding.weight)
        z_cond = self.discrete_to_cont(cond_ind, self.cond_vq_vae._embedding.weight)
        cat_quantized = torch.cat([z, z_cond], dim=1)
        return cat_quantized


    def discrete_to_cont(self, e_indices, e_weights):
        input_shape = e_indices.shape + (self.embedding_dim,)
        e_indices = e_indices.reshape(-1).unsqueeze(1)
        
        min_encodings = torch.zeros(e_indices.shape[0], self.num_embeddings, device=e_indices.device)
        min_encodings.scatter_(1, e_indices, 1)
        
        quantized = torch.matmul(
            min_encodings, e_weights).view(input_shape)
        
        z_q = torch.matmul(min_encodings, e_weights).view(input_shape) 
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

    def set_pixel_cnn(self, pixel_cnn):
        self.pixel_cnn = pixel_cnn

    def decode(self, latents, cont=True):
        if cont:
            z = latents.reshape(-1, 2 * self.embedding_dim, self.root_len, self.root_len)
        else:
            z = self.conditioned_discrete_to_cont(latents)
        return self._decoder(z)


    # def sample_prior(self, batch_size, cont=True):
    #     size = self.latent_sizes[0]**0.5
    #     e_indices = self.pixel_cnn.generate(shape=(size, size), batch_size=batch_size)
    #     e_indices = e_indices.reshape(batch_size, -1)
    #     if cont:
    #         return self.discrete_to_cont(e_indices)
    #     return e_indices


class CVQVAE1(nn.Module):
    def __init__(
        self,
        embedding_dim,
        input_channels=3,
        num_hiddens=128,
        num_residual_layers=3,
        num_residual_hiddens=64,
        num_embeddings=512,
        commitment_cost=0.25,
        decoder_output_activation=None, #IGNORED FOR NOW
        architecture=None, #IGNORED FOR NOW
        imsize=48,
        decay=0):
        super(CVQVAE, self).__init__()
        self.imsize = imsize
        self.embedding_dim = embedding_dim
        self.pixel_cnn = None
        self.input_channels = input_channels
        self.imlength = imsize * imsize * input_channels
        self.num_embeddings = num_embeddings
        self._encoder = Encoder(input_channels * 2, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self.cond_encoder = Encoder(input_channels, num_hiddens,
                                    num_residual_layers,
                                    num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)

        self.cond_pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
                                              commitment_cost, decay)
            self.cond_vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                                           commitment_cost)

            self.cond_vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                                              commitment_cost)

        self._decoder = Decoder(self.embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self.representation_size = 0
        self.latent_sizes = []
        self.discrete_size = 0
        self.root_len = 0

    def compute_loss(self, obs, cond):
        obs = obs.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        cond = cond.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        inputs = torch.cat([obs, cond], dim=1)
        
        z = self._encoder(inputs)
        z = self._pre_vq_conv(z)
        z_cond = self.cond_encoder(cond)
        z_cond = self.cond_pre_vq_conv(z_cond)

        vq_loss, quantized, perplexity, _ = self._vq_vae(z)
        cond_vq_loss, cond_quantized, cond_perplexity, _ = self.cond_vq_vae(z_cond)
        comb_quantized = quantized + cond_quantized
        #cat_quantized = torch.cat([quantized, cond_quantized], dim=1)

        if self.representation_size == 0:
            z_size = quantized[0].flatten().shape[0]
            z_cond_size = cond_quantized[0].flatten().shape[0]
            self.latent_sizes = [z_size, z_cond_size]
            self.representation_size = z_size + z_cond_size
            self.discrete_size = self.representation_size // self.embedding_dim
            self.root_len = int((self.discrete_size // 2) ** 0.5)
        
        cond_recon = self._decoder(cond_quantized)
        x_recon = self._decoder(comb_quantized)
        vq_losses = [vq_loss, cond_vq_loss]
        perplexities = [perplexity, cond_perplexity]
        recons = [x_recon, cond_recon]
        errors = [F.mse_loss(x_recon, obs), F.mse_loss(cond_recon, cond)]
        return vq_losses, perplexities, recons, errors


    def latent_to_square(self, latents):
        latents = latents.reshape(-1, 2, self.root_len, self.root_len)
        return latents[:, 0], latents[:, 1]

    def encode(self, obs, cond, cont=True):
        obs = obs.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        cond = cond.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        inputs = torch.cat([obs, cond], dim=1)
        z = self._encoder(inputs)
        z = self._pre_vq_conv(z)

        z_cond = self.cond_encoder(cond)
        z_cond = self.cond_pre_vq_conv(z_cond)

        _, quantized, _, encodings = self._vq_vae(z)
        _, cond_quantized, _, cond_encodings = self.cond_vq_vae(z_cond)

        if cont:
            z, z_c = quantized, cond_quantized
        else:
            z, z_c = encodings, cond_encodings
        

        z = z.reshape(obs.shape[0], -1)
        z_c = z_c.reshape(cond.shape[0], -1)
        z_cond = torch.cat([z, z_c], dim=1)
        return z_cond


    def conditioned_discrete_to_cont(self, e_indices):
        z_ind, cond_ind = self.latent_to_square(e_indices)
        z = self.discrete_to_cont(z_ind, self._vq_vae._embedding.weight)
        z_cond = self.discrete_to_cont(cond_ind, self.cond_vq_vae._embedding.weight)
        cat_quantized = torch.cat([z, z_cond], dim=1)
        return cat_quantized


    def discrete_to_cont(self, e_indices, e_weights):
        input_shape = e_indices.shape + (self.embedding_dim,)
        e_indices = e_indices.reshape(-1).unsqueeze(1)
        
        min_encodings = torch.zeros(e_indices.shape[0], self.num_embeddings, device=e_indices.device)
        min_encodings.scatter_(1, e_indices, 1)
        
        quantized = torch.matmul(
            min_encodings, e_weights).view(input_shape)
        
        z_q = torch.matmul(min_encodings, e_weights).view(input_shape) 
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

    def set_pixel_cnn(self, pixel_cnn):
        self.pixel_cnn = pixel_cnn

    def decode(self, latents, cont=True):
        if cont:
            z = latents.reshape(-1, 2 * self.embedding_dim, self.root_len, self.root_len)
        else:
            z_comb = self.conditioned_discrete_to_cont(latents)
            z_pos = z_comb[:, :self.embedding_dim]
            z_obj = z_comb[:, self.embedding_dim:]
            z = z_pos + z_obj
        return self._decoder(z)



class CVQVAE1(nn.Module):
    def __init__(
        self,
        embedding_dim,
        input_channels=3,
        num_hiddens=128,
        num_residual_layers=3,
        num_residual_hiddens=64,
        num_embeddings=512,
        commitment_cost=0.25,
        decoder_output_activation=None, #IGNORED FOR NOW
        architecture=None, #IGNORED FOR NOW
        min_variance=1e-3,
        imsize=48,
        decay=0):
        super(CVQVAE, self).__init__()
        self.imsize = imsize
        self.embedding_dim = embedding_dim
        self.pixel_cnn = None
        self.input_channels = input_channels
        self.imlength = imsize * imsize * input_channels
        self.num_embeddings = num_embeddings
        self.log_min_variance = float(np.log(min_variance))

        #Calculate latent sizes
        if imsize == 48: self.root_len = 12
        elif imsize == 84: self.root_len = 21
        else: raise ValueError(imsize)

        self.discrete_size = self.root_len * self.root_len
        self.latent_sizes = [self.discrete_size, self.discrete_size * self.embedding_dim]
        self.representation_size = sum(self.latent_sizes)
        #Calculate latent sizes

        self._encoder = Encoder(input_channels * 2, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self.cond_encoder = Encoder(input_channels, num_hiddens,
                                    num_residual_layers,
                                    num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=1,
                                      #out_channels=self.embedding_dim
                                      kernel_size=1,
                                      stride=1)

        self.cond_pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
                                              commitment_cost, decay)
            self.cond_vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                                           commitment_cost, gaussion_prior=True)

            self.cond_vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                                              commitment_cost)

        self._decoder = Decoder(self.embedding_dim + 1,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self.cond_decoder = Decoder(self.embedding_dim,
                                    num_hiddens,
                                    num_residual_layers,
                                    num_residual_hiddens)

        in_layers = 2
        out_layers = 2
        self.fc_in = nn.ModuleList([nn.Linear(self.discrete_size, self.discrete_size) for i in range(in_layers)])
        self.bn_in = nn.ModuleList([nn.BatchNorm1d(self.discrete_size) for i in range(in_layers)])
        self.fc_out = nn.ModuleList([nn.Linear(self.representation_size, self.representation_size) for i in range(out_layers)])
        self.bn_out = nn.ModuleList([nn.BatchNorm1d(self.representation_size) for i in range(out_layers)])
        self.dropout = nn.Dropout(0.5)
        for f in self.fc_in:
            f.weight.data.uniform_(-1e-3, 1e-3)
            f.bias.data.uniform_(-1e-3, 1e-3)
        for f in self.fc_out:
            f.weight.data.uniform_(-1e-3, 1e-3)
            f.bias.data.uniform_(-1e-3, 1e-3)
        for f in self.bn_in:
            f.weight.data.uniform_(-1e-3, 1e-3)
            f.bias.data.uniform_(-1e-3, 1e-3)
        for f in self.bn_out:
            f.weight.data.uniform_(-1e-3, 1e-3)
            f.bias.data.uniform_(-1e-3, 1e-3)

        self.logvar = nn.Parameter(torch.randn(144))

    def compute_loss(self, obs, cond):
        obs = obs.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        cond = cond.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        inputs = torch.cat([obs, cond], dim=1)
        
        z = self._encoder(inputs)
        z = self._pre_vq_conv(z).reshape(-1, self.discrete_size)
        for i in range(len(self.fc_in)):
            z = self.fc_in[i](self.dropout(self.bn_in[i](F.relu(z))))
        z = z.reshape(-1, 1, self.root_len, self.root_len)
        
        z_cond = self.cond_encoder(cond)
        z_cond = self.cond_pre_vq_conv(z_cond)

        vq_loss, quantized, perplexity, _ = self._vq_vae(z.detach()) 
        #NOTE DETACHED ABOVE
        
        #quantized = self.reparameterize(quantized)
        quantized = self.reparameterize(z)
        
        cond_vq_loss, cond_quantized, cond_perplexity, _ = self.cond_vq_vae(z_cond)
        cat_quantized = torch.cat([quantized, cond_quantized], dim=1).reshape(-1, self.representation_size)
        
        for i in range(len(self.fc_out)):
            cat_quantized = self.fc_out[i](self.dropout(self.bn_out[i]((F.relu(cat_quantized)))))
        cat_quantized = cat_quantized.reshape(-1, self.embedding_dim + 1, self.root_len, self.root_len)

        kle = self.kl_divergence(z)
        cond_recon = self.cond_decoder(cond_quantized)
        x_recon = self._decoder(cat_quantized)
        vq_losses = [vq_loss, cond_vq_loss]
        perplexities = [perplexity, cond_perplexity]
        recons = [x_recon, cond_recon]
        #errors = [F.mse_loss(x_recon, obs), F.mse_loss(cond_recon, cond)]
        errors = [F.mse_loss(x_recon, obs, reduction='sum'), F.mse_loss(cond_recon, cond, reduction='sum')]
        return vq_losses, perplexities, recons, errors, kle


    def latent_to_square(self, latents):
        latents = latents.reshape(-1, 2, self.root_len, self.root_len)
        return latents[:, 0], latents[:, 1]

    def encode(self, obs, cond, cont=True):
        obs = obs.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        cond = cond.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        inputs = torch.cat([obs, cond], dim=1)
        z = self._encoder(inputs)
        z = self._pre_vq_conv(z).reshape(-1, self.discrete_size)        
        for i in range(len(self.fc_in)):
            z = self.fc_in[i](self.dropout(self.bn_in[i](F.relu(z))))
        z = z.reshape(-1, 1, self.root_len, self.root_len)

        z_cond = self.cond_encoder(cond)
        z_cond = self.cond_pre_vq_conv(z_cond)

        _, quantized, _, encodings = self._vq_vae(z)
        _, cond_quantized, _, cond_encodings = self.cond_vq_vae(z_cond)
        quantized = self.reparameterize(z)

        if cont:
            #z, z_c = z, cond_quantized
            z, z_c = quantized, cond_quantized
        else:
            z, z_c = encodings, cond_encodings
        

        z = z.reshape(obs.shape[0], -1)
        z_c = z_c.reshape(cond.shape[0], -1)
        z_cond = torch.cat([z, z_c], dim=1)
        return z_cond

    def reparameterize(self, quantized):
        if self.training:
            return self.rsample(quantized)
        return quantized

    def kl_divergence(self, quantized):
        logvar = self.log_min_variance + torch.abs(self.logvar)
        mu = quantized.reshape(-1, self.latent_sizes[0])
        return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def rsample(self, mu):
        logvar = self.log_min_variance + torch.abs(self.logvar)
        stds = (0.5 * logvar).exp()
        stds = stds.repeat(mu.shape[0], 1).reshape(*mu.size())
        epsilon = ptu.randn(*mu.size())
        latents = epsilon * stds + mu
        return latents
    
    # def rsample(self, mu):
    #     logvar = self.log_min_variance + torch.abs(self.logvar)
    #     stds = (0.5 * logvar).exp()
    #     epsilon = ptu.randn(*mu.size())
    #     latents = epsilon * stds + mu
    #     return latents

    def conditioned_discrete_to_cont(self, e_indices):
        z_ind, cond_ind = self.latent_to_square(e_indices)
        z = self.discrete_to_cont(z_ind, self._vq_vae._embedding.weight)
        z_cond = self.discrete_to_cont(cond_ind, self.cond_vq_vae._embedding.weight)
        cat_quantized = torch.cat([z, z_cond], dim=1)
        return cat_quantized


    def discrete_to_cont(self, e_indices, e_weights):
        input_shape = e_indices.shape + (self.embedding_dim,)
        e_indices = e_indices.reshape(-1).unsqueeze(1)
        
        min_encodings = torch.zeros(e_indices.shape[0], self.num_embeddings, device=e_indices.device)
        min_encodings.scatter_(1, e_indices, 1)
        
        quantized = torch.matmul(
            min_encodings, e_weights).view(input_shape)
        
        z_q = torch.matmul(min_encodings, e_weights).view(input_shape) 
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

    def set_pixel_cnn(self, pixel_cnn):
        self.pixel_cnn = pixel_cnn

    def decode(self, latents, cont=True):
        if cont:
            for i in range(len(self.fc_out)):
                latents = self.fc_out[i](self.dropout(self.bn_out[i](F.relu(latents))))
            z = latents.reshape(-1, self.embedding_dim + 1, self.root_len, self.root_len)
            #z = latents.reshape(-1, 2 * self.embedding_dim, self.root_len, self.root_len)
        else:
            z = self.conditioned_discrete_to_cont(latents)
        return self._decoder(z)

    def sample_prior(self, batch_size, cond, cont=True):
        if cond.shape[0] == 1:
            cond = cond.repeat(batch_size, 1)

        cond = cond.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)
        z_cond = self.cond_encoder(cond)
        z_cond = self.cond_pre_vq_conv(z_cond)
        _, cond_quantized, _, cond_encodings = self.cond_vq_vae(z_cond)

        z = ptu.randn(batch_size, 1, self.root_len, self.root_len)
        _, quantized, _, encodings = self._vq_vae(z)

        if cont:
            z, z_c = z, cond_quantized
            #z, z_c = quantized, cond_quantized
        else:
            z, z_c = encodings, cond_encodings

        z = z.reshape(batch_size, -1)
        z_c = z_c.reshape(batch_size, -1)
        z_cond = torch.cat([z, z_c], dim=1)
        return z_cond



class CVQVAEQuantize(nn.Module):
    def __init__(
        self,
        embedding_dim,
        input_channels=3,
        num_hiddens=128,
        num_residual_layers=3,
        num_residual_hiddens=64,
        num_embeddings=512,
        commitment_cost=0.25,
        decoder_output_activation=None, #IGNORED FOR NOW
        architecture=None, #IGNORED FOR NOW
        min_variance=1e-3,
        imsize=48,
        decay=0.0):
        super(CVQVAE, self).__init__()
        self.imsize = imsize
        self.embedding_dim = embedding_dim
        self.pixel_cnn = None
        self.input_channels = input_channels
        self.imlength = imsize * imsize * input_channels
        self.num_embeddings = num_embeddings
        self.log_min_variance = float(np.log(min_variance))

        #Calculate latent sizes
        if imsize == 48: self.root_len = 12
        elif imsize == 84: self.root_len = 21
        else: raise ValueError(imsize)

        self.discrete_size = self.root_len * self.root_len
        self.latent_sizes = [self.discrete_size * self.embedding_dim, self.discrete_size * self.embedding_dim]
        self.representation_size = sum(self.latent_sizes)
        #Calculate latent sizes

        self._encoder = Encoder(input_channels * 2, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self.cond_encoder = Encoder(input_channels, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self.f_mu = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        self.f_logvar = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        self.cond_pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
                                              commitment_cost, decay)
            self.cond_vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                                           commitment_cost)

            self.cond_vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                                              commitment_cost)

        self._decoder = Decoder(self.embedding_dim * 2,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self.cond_decoder = Decoder(self.embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)


    def compute_loss(self, obs, cond):
        obs = obs.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        cond = cond.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        inputs = torch.cat([obs, cond], dim=1)
        z_delta = self._encoder(inputs)
        z_cond = self.cond_encoder(cond)
        z_cond = self.cond_pre_vq_conv(z_cond)
        z_s, kle, vq_loss, z_quant, perplexity, _ = self.reparameterize(z_delta, return_loss=True)
        cond_vq_loss, cond_quant, cond_perplexity, _ = self.cond_vq_vae(z_cond)
        
        cat_quantized = torch.cat([z_s, cond_quant], dim=1)
        x_recon = self._decoder(cat_quantized)
        cond_recon = self.cond_decoder(cond_quant)
        vq_losses = [vq_loss, cond_vq_loss]
        perplexities = [perplexity, cond_perplexity]
        recons = [x_recon, cond_recon]
        errors = [F.mse_loss(x_recon, obs, reduction='sum'), F.mse_loss(cond_recon, cond, reduction='sum')]
        return vq_losses, perplexities, recons, errors, kle


    def latent_to_square(self, latents):
        latents = latents.reshape(-1, 2, self.root_len, self.root_len)
        return latents[:, 0], latents[:, 1]

    def encode(self, obs, cond, cont=True):
        obs = obs.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        cond = cond.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        inputs = torch.cat([obs, cond], dim=1)

        z_delta = self._encoder(inputs)
        z_cond = self.cond_encoder(cond)
        z_cond = self.cond_pre_vq_conv(z_cond)
        z_s = self.reparameterize(z_delta)
        cond_vq_loss, cond_quant, cond_perplexity, _ = self.cond_vq_vae(z_cond)

        cat_quantized = torch.cat([z_s, cond_quant], dim=1)

        return cat_quantized.reshape(-1, self.representation_size)

    def reparameterize(self, latent, return_loss=False):
        mu = self.f_mu(latent)
        logvar = self.log_min_variance + torch.abs(self.f_logvar(latent))

        vq_loss, z_quant, perplexity, e_indices = self._vq_vae(mu)
        z_quant = z_quant.reshape(-1, self.latent_sizes[0])
        mu = mu.reshape(-1, self.latent_sizes[0])
        logvar = logvar.reshape(-1, self.latent_sizes[0])
        
        # if self.training: z_s = self.rsample(mu, logvar)
        # else: z_s = mu
        if self.training: z_s = self.rsample(z_quant, logvar)
        else: z_s = z_quant
        
        z_s = z_s.reshape(-1, self.embedding_dim, self.root_len, self.root_len)
        
        if return_loss:
            kle = self.kl_divergence(mu, logvar)
            return z_s, kle, vq_loss, z_quant, perplexity, e_indices

        return z_s

    def kl_divergence(self, mu, logvar):
        return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def rsample(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = ptu.randn(*mu.size())
        latents = epsilon * stds + mu
        return latents

    def conditioned_discrete_to_cont(self, e_indices):
        z_ind, cond_ind = self.latent_to_square(e_indices)
        z = self.discrete_to_cont(z_ind, self._vq_vae._embedding.weight)
        z_cond = self.discrete_to_cont(cond_ind, self.cond_vq_vae._embedding.weight)
        cat_quantized = torch.cat([z, z_cond], dim=1)
        return cat_quantized


    def discrete_to_cont(self, e_indices, e_weights):
        input_shape = e_indices.shape + (self.embedding_dim,)
        e_indices = e_indices.reshape(-1).unsqueeze(1)
        
        min_encodings = torch.zeros(e_indices.shape[0], self.num_embeddings, device=e_indices.device)
        min_encodings.scatter_(1, e_indices, 1)
        
        quantized = torch.matmul(
            min_encodings, e_weights).view(input_shape)
        
        z_q = torch.matmul(min_encodings, e_weights).view(input_shape) 
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

    def set_pixel_cnn(self, pixel_cnn):
        self.pixel_cnn = pixel_cnn

    def decode(self, latents, cont=True):
        if cont:
            # for i in range(len(self.fc_out)):
            #     latents = self.fc_out[i](self.dropout(self.bn_out[i](F.relu(latents))))
            z = latents.reshape(-1, self.embedding_dim * 2, self.root_len, self.root_len)
        else:
            z = self.conditioned_discrete_to_cont(latents)
        return self._decoder(z)

    def sample_prior(self, batch_size, cond, cont=True):
        if cond.shape[0] == 1:
            cond = cond.repeat(batch_size, 1)

        cond = cond.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)
        z_cond = self.cond_encoder(cond)
        z_cond = self.cond_pre_vq_conv(z_cond)
        z = ptu.randn(batch_size, self.embedding_dim, self.root_len, self.root_len)
        
        _, cond_quant, _, cond_encodings = self.cond_vq_vae(z_cond)
        _, z_quant, _, encodings = self._vq_vae(z)

        if cont:
            z, z_c = z_quant, cond_quant
        else:
            z, z_c = encodings, cond_encodings

        z = z.reshape(batch_size, -1)
        z_c = z_c.reshape(batch_size, -1)
        z_cond = torch.cat([z, z_c], dim=1)
        return z_cond


class VAE(nn.Module):
    def __init__(
        self,
        representation_size,
        embedding_dim=3,
        input_channels=3,
        num_hiddens=128,
        num_residual_layers=3,
        num_residual_hiddens=64,
        decoder_output_activation=None, #IGNORED FOR NOW
        architecture=None, #IGNORED FOR NOW
        min_variance=1e-3,
        imsize=48,
        ):
        super(VAE, self).__init__()
        self.log_min_variance = float(np.log(min_variance))
        self.imsize = imsize
        self.embedding_dim = embedding_dim
        self.input_channels = input_channels
        self.imlength = imsize * imsize * input_channels
        
        self._encoder = Encoder(input_channels, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self._decoder = Decoder(1,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        
        #Calculate latent sizes
        if imsize == 48:
            self.root_deconv_conv_size = 12
        elif imsize == 84:
            self.root_deconv_conv_size = 21
        else:
            raise ValueError(imsize)

        self.conv_size = num_hiddens * 2
        self.representation_size = representation_size
        #Calculate latent sizes

        assert representation_size < self.conv_size  # This is a bad idea (wrong bottleneck)

        self.f_mu = nn.Linear(self.conv_size, self.representation_size)
        self.f_logvar = nn.Linear(self.conv_size, self.representation_size)
        self.f_dec = nn.Linear(self.representation_size, int(self.root_deconv_conv_size ** 2))

        #self.dropout = nn.Dropout(0.5)
        #self.bn = nn.BatchNorm1d(self.conv_size)

        self.f_mu.weight.data.uniform_(-1e-3, 1e-3)
        self.f_mu.bias.data.uniform_(-1e-3, 1e-3)
        self.f_logvar.weight.data.uniform_(-1e-3, 1e-3)
        self.f_logvar.bias.data.uniform_(-1e-3, 1e-3)
        self.f_dec.weight.data.uniform_(-1e-3, 1e-3)
        self.f_dec.bias.data.uniform_(-1e-3, 1e-3)

    def compute_loss(self, obs):
        obs = obs.view(-1,
          self.input_channels,
          self.imsize,
          self.imsize)

        z_s, kle = self.encode_image(obs)
        recon = self.decode(z_s)

        # log_sigma = ((obs - recon) ** 2).mean([0,1,2,3], keepdim=True).sqrt().log()
        # log_sigma = self.softclip(log_sigma, -6)

        recon_error = F.mse_loss(recon, obs, reduction='sum')
        #recon_error = self.gaussian_nll(recon, log_sigma, obs).sum()

        return recon, recon_error, kle

    # def gaussian_nll(self, mu, log_sigma, x):
    #     return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)

    # def softclip(self, tensor, min):
    #     """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    #     result_tensor = min + F.softplus(tensor - min)
    #     return result_tensor


    def encode_image(self, obs):
        obs = obs.view(-1,
          self.input_channels,
          self.imsize,
          self.imsize)

        z_conv = self._encoder(obs)
        z_conv = self.spatial_encoder(z_conv)

        return self.compress(z_conv)

    def compress(self, z_conv):
        z_conv = z_conv.view(-1, self.conv_size)

        mu = self.f_mu(z_conv)
        logvar = self.log_min_variance + torch.abs(self.f_logvar(z_conv))
        if self.training:
          z_s = self.rsample(mu, logvar)
        else:
          z_s = mu

        kle = self.kl_divergence(mu, logvar)
        
        return z_s, kle

    def decompress(self, z_s):
        z_conv = self.f_dec(z_s)
        z_conv = z_conv.view(-1, 1, self.root_deconv_conv_size, self.root_deconv_conv_size)   
        return z_conv

    def rsample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_divergence(self, mu, logvar):
        return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def encode(self, inputs):
        z_s, _ = self.encode_image(inputs)
        return z_s

    def sample_prior(self, batch_size):
        z_s = ptu.randn(batch_size, self.representation_size)
        return z_s

    def decode(self, latents):
        z_conv = self.decompress(latents)
        img_recon = self._decoder(z_conv)
        return img_recon

    def spatial_encoder(self, latent):
        latent = self.softmax(latent)

        maps_x = torch.sum(latent, 2)
        maps_y = torch.sum(latent, 3)

        weights = ptu.from_numpy(np.arange(maps_x.shape[-1]) / maps_x.shape[-1])
        fp_x = torch.sum(maps_x * weights, 2)
        fp_y = torch.sum(maps_y * weights, 2)
        latent = torch.cat([fp_x, fp_y], 1)

        return latent


class CVAE(nn.Module):
    def __init__(
        self,
        representation_size, #ignored currently
        embedding_dim=1,
        input_channels=3,
        num_hiddens=128,
        num_residual_layers=3,
        num_residual_hiddens=64,
        decoder_output_activation=None, #IGNORED FOR NOW
        architecture=None, #IGNORED FOR NOW
        min_variance=1e-3,
        imsize=48,
        ):
        super(CVAE, self).__init__()
        self.log_min_variance = float(np.log(min_variance))
        self.imsize = imsize
        self.embedding_dim = embedding_dim
        self.input_channels = input_channels
        self.imlength = imsize * imsize * input_channels
        #self.softmax = nn.Softmax2d()
        
        self._encoder = Encoder(2 * input_channels, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self._cond_encoder = Encoder(input_channels, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self.f_mu = nn.Conv2d(in_channels=num_hiddens,
                                    out_channels=self.embedding_dim,
                                    kernel_size=1,
                                    stride=1)

        self.f_logvar = nn.Conv2d(in_channels=num_hiddens,
                                    out_channels=self.embedding_dim,
                                    kernel_size=1,
                                    stride=1)

        self._conv_cond = nn.Conv2d(in_channels=num_hiddens,
                                    out_channels=self.embedding_dim,
                                    kernel_size=1,
                                    stride=1)

        self._decoder = Decoder(2 * self.embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens,
                                out_channels=2 * input_channels)
        
        #Calculate latent sizes
        if imsize == 48:
            self.root_len = 12
        elif imsize == 84:
            self.root_len = 21
        else:
            raise ValueError(imsize)

        self.representation_size = self.root_len * self.root_len * self.embedding_dim

        # self.conv_in_size = num_hiddens * 2
        # self.conv_out_size = int(self.root_deconv_conv_size ** 2)
        # self.representation_size = representation_size
        #Calculate latent sizes

        #assert representation_size < self.conv_in_size  # This is a bad idea (wrong bottleneck)

        # self.f_mu = nn.Linear(self.conv_in_size, self.representation_size)
        # self.f_logvar = nn.Linear(self.conv_in_size, self.representation_size)
        # self.f_dec = nn.Linear(self.representation_size, self.conv_out_size)


        # self.f_mu.weight.data.uniform_(-1e-3, 1e-3)
        # self.f_mu.bias.data.uniform_(-1e-3, 1e-3)
        # self.f_logvar.weight.data.uniform_(-1e-3, 1e-3)
        # self.f_logvar.bias.data.uniform_(-1e-3, 1e-3)
        # self.f_dec.weight.data.uniform_(-1e-3, 1e-3)
        # self.f_dec.bias.data.uniform_(-1e-3, 1e-3)

    def compute_loss(self, x_delta, x_cond):
        x_delta = x_delta.view(-1,
          self.input_channels,
          self.imsize,
          self.imsize)

        x_cond = x_cond.view(-1,
          self.input_channels,
          self.imsize,
          self.imsize)

        z_cat, kle = self.encode_image(x_delta, x_cond)
        recons = self._decoder(z_cat)

        delta_recon = recons[:,:self.input_channels]
        cond_recon = recons[:,self.input_channels:]

        # log_sigma = ((obs - recon) ** 2).mean([0,1,2,3], keepdim=True).sqrt().log()
        # log_sigma = self.softclip(log_sigma, -6)
        x_recon_error = F.mse_loss(delta_recon, x_delta, reduction='sum')
        c_recon_error = F.mse_loss(cond_recon, x_cond, reduction='sum')
        #recon_error = self.gaussian_nll(recon, log_sigma, obs).sum()

        return delta_recon, x_recon_error, c_recon_error, kle

    # def gaussian_nll(self, mu, log_sigma, x):
    #     return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)

    # def softclip(self, tensor, min):
    #     """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    #     result_tensor = min + F.softplus(tensor - min)
    #     return result_tensor


    def encode_image(self, x_delta, x_cond):
        x_delta = x_delta.view(-1,
          self.input_channels,
          self.imsize,
          self.imsize)
        x_cond = x_cond.view(-1,
          self.input_channels,
          self.imsize,
          self.imsize)

        obs = torch.cat([x_delta, x_cond], dim=1)
        z_delta = self._encoder(obs)

        mu = self.f_mu(z_delta)
        logvar = self.log_min_variance + torch.abs(self.f_logvar(z_delta))
        kle = self.kl_divergence(mu, logvar)

        if self.training:
          z_s = self.rsample(mu, logvar)
        else:
          z_s = mu

        z_cond = self._cond_encoder(x_cond)
        z_cond = self._conv_cond(z_cond)

        z_cat = torch.cat([z_s, z_cond], dim=1)

        return z_cat, kle

    # def compress(self, z_delta, z_cond):
    #     z_delta = z_delta.view(-1, self.conv_in_size)
    #     z_cond = z_cond.view(-1, self.conv_out_size)

    #     mu = self.f_mu(z_delta)
    #     logvar = self.log_min_variance + torch.abs(self.f_logvar(z_delta))
    #     kle = self.kl_divergence(mu, logvar)

    #     if self.training:
    #       z_s = self.rsample(mu, logvar)
    #     else:
    #       z_s = mu

    #     z_cat = torch.cat([z_s, z_cond], dim=1)
        
    #     return z_cat, kle

    def decompress(self, z_cat):
        z_s = z_cat[:, :self.representation_size]
        z_cond = z_cat[:, self.representation_size:]
        
        z_delta = self.f_dec(z_s)
        z_delta = z_delta.view(-1, 1, self.root_deconv_conv_size, self.root_deconv_conv_size)
        z_cond = z_cond.view(-1, 1, self.root_deconv_conv_size, self.root_deconv_conv_size)
        
        z_conv = torch.cat([z_delta,z_cond], dim=1)
        return z_conv

    def rsample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_divergence(self, mu, logvar):
        mu = mu.reshape(-1, self.representation_size)
        logvar = logvar.reshape(-1, self.representation_size)
        return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def encode(self, inputs):
        z_s, _ = self.encode_image(inputs)
        return z_s

    def sample_prior(self, batch_size, cond):
        if cond.shape[0] == 1:
            cond = cond.repeat(batch_size, 1)

        cond = cond.view(batch_size,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        z_cond = self._cond_encoder(cond)
        z_cond = self._conv_cond(z_cond)
        #z_cond = z_cond.view(-1, self.conv_out_size)

        z_delta = ptu.randn(batch_size, self.embedding_dim, self.root_len, self.root_len)
        z_cat = torch.cat([z_delta, z_cond], dim=1)
        return z_cat.view(-1, self.representation_size)

    def decode(self, latents):
        #z_conv = self.decompress(latents)
        latents = latents.view(-1, self.embedding_dim * 2, self.root_len, self.root_len)
        img_recon = self._decoder(latents)[:,:self.input_channels]
        return img_recon

    def spatial_encoder(self, latent):
        latent = self.softmax(latent)

        maps_x = torch.sum(latent, 2)
        maps_y = torch.sum(latent, 3)

        weights = ptu.from_numpy(np.arange(maps_x.shape[-1]) / maps_x.shape[-1])
        fp_x = torch.sum(maps_x * weights, 2)
        fp_y = torch.sum(maps_y * weights, 2)
        latent = torch.cat([fp_x, fp_y], 1)

        return latent

# class CVAE(nn.Module):
#     def __init__(
#         self,
#         embedding_dim,
#         root_len=21,
#         input_channels=3,
#         num_hiddens=128,
#         num_residual_layers=3,
#         num_residual_hiddens=64,
#         num_embeddings=512,
#         commitment_cost=0.25,
#         decoder_output_activation=None, #IGNORED FOR NOW
#         architecture=None, #IGNORED FOR NOW
#         imsize=48,
#         decay=0.0):
#         super(CVAE, self).__init__()
#         self.imsize = imsize
#         self.embedding_dim = embedding_dim
#         self.pixel_cnn = None
#         self.input_channels = input_channels
#         self.imlength = imsize * imsize * input_channels
#         self.num_embeddings = num_embeddings
        
#         self._encoder = Encoder(2 * input_channels, num_hiddens,
#                                 num_residual_layers,
#                                 num_residual_hiddens)
#         self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
#                                       out_channels=self.embedding_dim,
#                                       kernel_size=1,
#                                       stride=1)


#         self._cond_encoder = Encoder(input_channels, num_hiddens,
#                                 num_residual_layers,
#                                 num_residual_hiddens)
        
#         self._cond_pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
#                                       out_channels=self.embedding_dim,
#                                       kernel_size=1,
#                                       stride=1)

#         #FINISH THIS!!!

#         self._decoder = Decoder(2 * self.embedding_dim,
#                                 num_hiddens,
#                                 num_residual_layers,
#                                 num_residual_hiddens)

#         # self._cond_decoder = Decoder(self.embedding_dim,
#         #                         num_hiddens,
#         #                         num_residual_layers,
#         #                         num_residual_hiddens)
        
#         #Calculate latent sizes
#         if imsize == 48:
#             self.root_conv_size = 12
#         elif imsize == 84:
#             self.root_conv_size = 21
#         else:
#             raise ValueError(imsize)

#         #assert root_len < self.root_conv_size  # This is a bad idea (wrong bottleneck)
#         self.root_len = root_len
#         self.discrete_size = root_len * root_len
#         self.conv_size = self.root_conv_size * self.root_conv_size * self.embedding_dim
#         self.representation_size = self.discrete_size * self.embedding_dim
#         #Calculate latent sizes

#         self.f_mu = nn.Linear(self.conv_size, self.representation_size)
#         self.f_logvar = nn.Linear(self.conv_size, self.representation_size)
#         #self.f_c = nn.Linear(self.conv_size, self.representation_size)
#         self.f_dec = nn.Linear(self.representation_size, self.conv_size)

#         self.bn_c = nn.BatchNorm1d(self.conv_size)


#     def compute_loss(self, x_delta, x_cond):
#         x_delta = x_delta.view(-1,
#                             self.input_channels,
#                             self.imsize,
#                             self.imsize)

#         z_cat, kle = self.encode_images(x_delta, x_cond)

#         x_recon = self.decode(z_cat)
#         recon_error = F.mse_loss(x_recon, x_delta, reduction='sum')

#         return x_recon, recon_error, kle


#     def encode_images(self, x_delta, x_cond):
#         x_delta = x_delta.view(-1,
#                             self.input_channels,
#                             self.imsize,
#                             self.imsize)
#         x_cond = x_cond.view(-1,
#                             self.input_channels,
#                             self.imsize,
#                             self.imsize)

#         x_delta = torch.cat([x_delta, x_cond], dim=1)

#         z_delta = self._encoder(x_delta)
#         z_delta = self._pre_vq_conv(z_delta)

#         z_cond = self._cond_encoder(x_cond)
#         z_cond = self._cond_pre_vq_conv(z_cond)

#         return self.compress(z_delta, z_cond)

#     def compress(self, z_delta, z_cond):
#         z_delta = z_delta.view(-1, self.conv_size)
#         mu = self.f_mu(z_delta)
#         logvar = self.f_logvar(z_delta)

#         z_cond = z_cond.view(-1, self.conv_size)
#         #z_cond = self.bn_c(z_cond)

#         if self.training: z_s = self.rsample(mu, logvar)
#         else: z_s = mu

#         z_cat = torch.cat([z_s, z_cond], dim=1)
#         kle = self.kl_divergence(mu, logvar)
        
#         return z_cat, kle

#     def decompress(self, latents):
#         z_delta = self.f_dec(latents[:, :self.representation_size])
#         z_cond = latents[:, self.representation_size:]

#         z_delta = z_delta.view(-1, self.embedding_dim, self.root_conv_size, self.root_conv_size)
#         z_cond = z_cond.view(-1, self.embedding_dim, self.root_conv_size, self.root_conv_size)

#         z_cat = torch.cat([z_delta, z_cond], dim=1)    
#         return z_cat

#     def reparameterize(self, mu, logvar):
#         mu = self.f_mu(latent).reshape(-1, self.latent_sizes[0])
#         logvar = (self.log_min_variance + self.f_logvar(latent)).reshape(-1, self.latent_sizes[0])
        
#         if self.training: z_s = self.rsample(mu, logvar)
#         else: z_s = mu
        
#         z_s = z_s.reshape(-1, self.embedding_dim, self.root_len, self.root_len)
#         kle = self.kl_divergence(mu, logvar)
#         return z_s, kle

#     def rsample(self, mu, logvar):
#         stds = (0.5 * logvar).exp()
#         epsilon = ptu.randn(*mu.size())
#         latents = epsilon * stds + mu
#         return latents

#     def kl_divergence(self, mu, logvar):
#         return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

#     def encode(self, inputs, cont=True):
#         z_cat, _ = self.encode_images(inputs)
#         return z_cat

#     def sample_prior(self, batch_size, cond):
#         if cond.shape[0] == 1:
#             cond = cond.repeat(batch_size, 1)

#         cond = cond.view(batch_size,
#                         self.input_channels,
#                         self.imsize,
#                         self.imsize)

#         z_cond = self._cond_encoder(cond)
#         z_cond = self._cond_pre_vq_conv(z_cond)
#         z_cond = z_cond.view(-1, self.conv_size)
#         #z_cond = self.bn_c(z_cond)

#         z_delta = ptu.randn(batch_size, self.representation_size)
#         z_cat = torch.cat([z_delta, z_cond], dim=1)

#         return z_cat

#     def decode(self, latents):
#         z_conv = self.decompress(latents)
#         return self._decoder(z_conv)

#     def encode_one_np(self, inputs, cont=True):
#         return ptu.get_numpy(self.encode(ptu.from_numpy(inputs), cont=cont))[0]

#     def encode_np(self, inputs, cont=True):
#         return ptu.get_numpy(self.encode(ptu.from_numpy(inputs), cont=cont))

#     def decode_one_np(self, inputs, cont=True):
#         return np.clip(ptu.get_numpy(self.decode(ptu.from_numpy(inputs).reshape(1, -1), cont=cont))[0], 0, 1)

#     def decode_np(self, inputs, cont=True):
#         return np.clip(ptu.get_numpy(self.decode(ptu.from_numpy(inputs), cont=cont)), 0, 1)


class CVAE1(nn.Module):
    def __init__(
        self,
        embedding_dim,
        input_channels=3,
        num_hiddens=128,
        num_residual_layers=3,
        num_residual_hiddens=64,
        num_embeddings=512,
        commitment_cost=0.25,
        decoder_output_activation=None, #IGNORED FOR NOW
        architecture=None, #IGNORED FOR NOW
        min_variance=1e-3,
        imsize=48,
        decay=0):
        super(CVAE, self).__init__()
        self.imsize = imsize
        self.embedding_dim = embedding_dim
        self.pixel_cnn = None
        self.input_channels = input_channels
        self.imlength = imsize * imsize * input_channels
        self.num_embeddings = num_embeddings
        self.log_min_variance = float(np.log(min_variance))

        #Calculate latent sizes
        if imsize == 48: self.root_len = 12
        elif imsize == 84: self.root_len = 21
        else: raise ValueError(imsize)

        self.discrete_size = self.root_len * self.root_len
        self.latent_sizes = [self.discrete_size * self.embedding_dim, self.discrete_size * self.embedding_dim]
        self.representation_size = sum(self.latent_sizes)
        #Calculate latent sizes

        self._encoder = Encoder(input_channels * 2, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self.cond_encoder = Encoder(input_channels, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self.f_mu = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        self.f_logvar = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        self.cond_pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
                                              commitment_cost, decay)
            self.cond_vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                                           commitment_cost)

            self.cond_vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                                              commitment_cost)

        self._decoder = Decoder(self.embedding_dim * 2,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self.cond_decoder = Decoder(self.embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)


    def compute_loss(self, obs, cond):
        obs = obs.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        cond = cond.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        inputs = torch.cat([obs, cond], dim=1)
        z_delta = self._encoder(inputs)
        z_cond = self.cond_encoder(cond)
        z_cond = self.cond_pre_vq_conv(z_cond)
        z_s, kle = self.reparameterize(z_delta)

        vq_loss, quantized, perplexity, _ = self._vq_vae(z_s.detach()) 
        cond_vq_loss, cond_quantized, cond_perplexity, _ = self.cond_vq_vae(z_cond.detach())
        
        cat_quantized = torch.cat([z_s, z_cond], dim=1)
        x_recon = self._decoder(cat_quantized)
        cond_recon = self.cond_decoder(z_cond)
        vq_losses = [vq_loss, cond_vq_loss]
        perplexities = [perplexity, cond_perplexity]
        recons = [x_recon, cond_recon]
        #errors = [F.mse_loss(x_recon, obs), F.mse_loss(cond_recon, cond)]
        errors = [F.mse_loss(x_recon, obs, reduction='sum'), F.mse_loss(cond_recon, cond, reduction='sum')]
        return vq_losses, perplexities, recons, errors, kle


    def latent_to_square(self, latents):
        latents = latents.reshape(-1, 2, self.root_len, self.root_len)
        return latents[:, 0], latents[:, 1]

    def encode(self, obs, cond, cont=True):
        obs = obs.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        cond = cond.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        inputs = torch.cat([obs, cond], dim=1)

        z_delta = self._encoder(inputs)
        z_cond = self.cond_encoder(cond)
        z_cond = self.cond_pre_vq_conv(z_cond)
        z_s, kle = self.reparameterize(z_delta)

        vq_loss, quantized, perplexity, _ = self._vq_vae(z_s.detach())
        cond_vq_loss, cond_quantized, cond_perplexity, _ = self.cond_vq_vae(z_cond.detach())

        cat_quantized = torch.cat([z_s, z_cond], dim=1)

        return cat_quantized.reshape(-1, self.representation_size)

    def reparameterize(self, latent):
        mu = self.f_mu(latent).reshape(-1, self.latent_sizes[0])
        logvar = self.log_min_variance +  torch.abs(self.f_logvar(latent))
        logvar = logvar.reshape(-1, self.latent_sizes[0])
        
        if self.training: z_s = self.rsample(mu, logvar)
        else: z_s = mu
        
        z_s = z_s.reshape(-1, self.embedding_dim, self.root_len, self.root_len)
        kle = self.kl_divergence(mu, logvar)
        return z_s, kle

    def kl_divergence(self, mu, logvar):
        return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def rsample(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = ptu.randn(*mu.size())
        latents = epsilon * stds + mu
        return latents

    def conditioned_discrete_to_cont(self, e_indices):
        z_ind, cond_ind = self.latent_to_square(e_indices)
        z = self.discrete_to_cont(z_ind, self._vq_vae._embedding.weight)
        z_cond = self.discrete_to_cont(cond_ind, self.cond_vq_vae._embedding.weight)
        cat_quantized = torch.cat([z, z_cond], dim=1)
        return cat_quantized


    def discrete_to_cont(self, e_indices, e_weights):
        input_shape = e_indices.shape + (self.embedding_dim,)
        e_indices = e_indices.reshape(-1).unsqueeze(1)
        
        min_encodings = torch.zeros(e_indices.shape[0], self.num_embeddings, device=e_indices.device)
        min_encodings.scatter_(1, e_indices, 1)
        
        quantized = torch.matmul(
            min_encodings, e_weights).view(input_shape)
        
        z_q = torch.matmul(min_encodings, e_weights).view(input_shape) 
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

    def set_pixel_cnn(self, pixel_cnn):
        self.pixel_cnn = pixel_cnn

    def decode(self, latents, cont=True):
        if cont:
            # for i in range(len(self.fc_out)):
            #     latents = self.fc_out[i](self.dropout(self.bn_out[i](F.relu(latents))))
            z = latents.reshape(-1, self.embedding_dim * 2, self.root_len, self.root_len)
        else:
            z = self.conditioned_discrete_to_cont(latents)
        return self._decoder(z)

    def sample_prior(self, batch_size, cond, cont=True):
        if cond.shape[0] == 1:
            cond = cond.repeat(batch_size, 1)

        cond = cond.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)
        z_cond = self.cond_encoder(cond)
        z_cond = self.cond_pre_vq_conv(z_cond)
        _, cond_quantized, _, cond_encodings = self.cond_vq_vae(z_cond)

        z = ptu.randn(batch_size, self.embedding_dim, self.root_len, self.root_len)
        _, quantized, _, encodings = self._vq_vae(z)

        cat_quantized = torch.cat([z, z_cond], dim=1)

        return cat_quantized.reshape(-1, self.representation_size)


        if cont:
            z, z_c = z, cond_quantized
            #z, z_c = quantized, cond_quantized
        else:
            z, z_c = encodings, cond_encodings

        z = z.reshape(batch_size, -1)
        z_c = z_c.reshape(batch_size, -1)
        z_cond = torch.cat([z, z_c], dim=1)
        return z_cond


    def spatial_encoder(self, latent):
        sofmax = nn.Softmax2d()
        latent = sofmax(latent)

        maps_x = torch.sum(latent, 2)
        maps_y = torch.sum(latent, 3)

        weights = ptu.from_numpy(np.arange(maps_x.shape[-1]) / maps_x.shape[-1])
        fp_x = torch.sum(maps_x * weights, 2)
        fp_y = torch.sum(maps_y * weights, 2)
        latent = torch.cat([fp_x, fp_y], 1)

        mu = self.mu(latent)
        logvar = self.log_min_variance + torch.abs(self.logvar(latent))
        return (mu, logvar)
