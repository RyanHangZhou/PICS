import einops
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import numpy as np


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, c_mask=None, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            c_mask_ = c_mask.permute(0, 3, 1, 2)
            c_mask_ = F.interpolate(c_mask_, size=(h.shape[2], h.shape[3]), mode='nearest')
            c_mask_ = c_mask_.permute(0, 2, 3, 1)
            c_mask_ = c_mask_.unsqueeze(1)
            for module in self.input_blocks:
                h = module(h, emb, context, c_mask_)
                hs.append(h)
                c_mask_ = c_mask.permute(0, 3, 1, 2)
                c_mask_ = F.interpolate(c_mask_, size=(h.shape[2], h.shape[3]), mode='nearest')
                c_mask_ = c_mask_.permute(0, 2, 3, 1)
                c_mask_ = c_mask_.unsqueeze(1)
            h = self.middle_block(h, emb, context, c_mask_)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            c_mask_ = c_mask.permute(0, 3, 1, 2)
            c_mask_ = F.interpolate(c_mask_, size=(h.shape[2], h.shape[3]), mode='nearest')
            c_mask_ = c_mask_.permute(0, 2, 3, 1)
            c_mask_ = c_mask_.unsqueeze(1)
            h = module(h, emb, context, c_mask_)

        h = h.type(x.dtype)
        return self.out(h)

    # def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
    #     hs = []
    #     with torch.no_grad():
    #         t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
    #         emb = self.time_embed(t_emb)
    #         h = x.type(self.dtype)
    #         for module in self.input_blocks:
    #             h = module(h, emb, context)
    #             hs.append(h)
    #         h = self.middle_block(h, emb, context)

    #     if control is not None:
    #         h += control.pop()

    #     for i, module in enumerate(self.output_blocks):
    #         if only_mid_control or control is None:
    #             h = torch.cat([h, hs.pop()], dim=1)
    #         else:
    #             h = torch.cat([h, hs.pop() + control.pop()], dim=1)
    #         h = module(h, emb, context)

    #     h = h.type(x.dtype)
    #     return self.out(h)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.use_spatial_transformer = use_spatial_transformer
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, c_mask, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb) # 1,1280
        
        # import pdb; pdb.set_trace()
        # 1,320,64,64
        # print('hint shape: ', np.shape(hint)) # [1, 4, 512, 512]
        # print('hint max and min: ', torch.max(hint), torch.min(hint))
        # 1, -1
        # print('context shape: ', np.shape(context)) # [1, 257, 1024, 2]
        # print('c_mask shape: ', np.shape(c_mask)) # [1, 512, 512, 2]
        
        # print('use_spatial_transformer:', self.use_spatial_transformer) # True
        # import pdb; pdb.set_trace()
        # context = context.permute(0, 2, 3, 1)
        # print('after context shape: ', np.shape(context)) # [1, 257, 1024, 2]
        # import pdb; pdb.set_trace()

        # resize masks
        # print('before c_mask shape: ', np.shape(c_mask)) # [1, 512, 512, 2]
        c_mask_ = c_mask.permute(0, 3, 1, 2)
        c_mask_ = F.interpolate(c_mask_, size=(x.shape[2], x.shape[3]), mode='nearest')
        c_mask_ = c_mask_.permute(0, 2, 3, 1)
        # print('mask shape: ', np.shape(c_mask)) # [1, 64, 64, 2]
        # print('context shape: ', np.shape(context)) # [1, 257, 1024, 2]
        c_mask_ = c_mask_.unsqueeze(1)
        # print('c_mask shape: ', np.shape(c_mask)) # [1, 1, 64, 64, 2]
        # import pdb; pdb.set_trace()

        guided_hint = self.input_hint_block(hint, emb, context, c_mask_) # only encode hint
        # print('guided_hint shape: ', np.shape(guided_hint)) # [1, 320, 64, 64]
        # print('end 1')
        # import pdb; pdb.set_trace()

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                # skip the first layer
                h = guided_hint
                # print('h shape: ', np.shape(h))
                # print('module: ', module)
                # print('111111111111111')
                guided_hint = None
            else:
                # print('222222222222222')
                c_mask_ = c_mask.permute(0, 3, 1, 2)
                c_mask_ = F.interpolate(c_mask_, size=(h.shape[2], h.shape[3]), mode='nearest')
                c_mask_ = c_mask_.permute(0, 2, 3, 1)
                c_mask_ = c_mask_.unsqueeze(1)
                h_new = module(h, emb, context, c_mask_)
                h =  h_new
                # print('h_new shape: ', np.shape(h_new))
            outs.append(zero_conv(h, emb, context, c_mask_))
            # print('3333333333333333')

        # print('end 2')
        # import pdb; pdb.set_trace()
        # print('44444444444444444')
        c_mask_ = c_mask.permute(0, 3, 1, 2)
        # print('ssss h shape: ', np.shape(h))
        c_mask_ = F.interpolate(c_mask_, size=(h.shape[2], h.shape[3]), mode='nearest')
        c_mask_ = c_mask_.permute(0, 2, 3, 1)
        c_mask_ = c_mask_.unsqueeze(1)
        # print('h shape: ', np.shape(h))
        # print('c_mask_ shape: ', np.shape(c_mask_))
        h_new = self.middle_block(h, emb, context, c_mask_)
        # print('h_new after shape: ', np.shape(h_new))
        # print('5555555555555555')
        c_mask_ = c_mask.permute(0, 3, 1, 2)
        c_mask_ = F.interpolate(c_mask_, size=(h_new.shape[2], h_new.shape[3]), mode='nearest')
        c_mask_ = c_mask_.permute(0, 2, 3, 1)
        c_mask_ = c_mask_.unsqueeze(1)
        outs.append(self.middle_block_out(h_new, emb, context, c_mask_))
        # print('66666666666666')
        # print('outs shape: ', np.shape(outs))
        # import pdb; pdb.set_trace()
        return outs

    # def forward(self, x, hint, timesteps, context, c_mask, **kwargs):
    #     t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
    #     emb = self.time_embed(t_emb) # 1,1280
        
    #     # 1,320,64,64
    #     # print('hint shape: ', np.shape(hint)) # [1, 4, 512, 512]
    #     # print('context shape: ', np.shape(context)) # [1, 257, 1024] -> [1, 2, 257, 1024]
    #     # print('use_spatial_transformer:', self.use_spatial_transformer) # True
    #     # import pdb; pdb.set_trace()
    #     # context = context.permute(0, 2, 3, 1)
    #     # print('after context shape: ', np.shape(context)) # [1, 257, 1024, 2]
    #     # import pdb; pdb.set_trace()
    #     guided_hint = self.input_hint_block(hint, emb, context) # only encode hint
    #     # print('guided_hint shape: ', np.shape(guided_hint)) # [1, 320, 64, 64]
    #     # print('end 1')
    #     # import pdb; pdb.set_trace()
    #     outs = []

    #     h = x.type(self.dtype)
    #     for module, zero_conv in zip(self.input_blocks, self.zero_convs):
    #         if guided_hint is not None:
    #             # skip the first layer
    #             h = guided_hint
    #             guided_hint = None
    #         else:
    #             h_new = module(h, emb, context) 
    #             h =  h_new 
    #         outs.append(zero_conv(h, emb, context))

    #     # print('end 2')
    #     # import pdb; pdb.set_trace()

    #     h_new = self.middle_block(h, emb, context)  
    #     outs.append(self.middle_block_out(h_new, emb, context))        
    #     return outs


class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        # import pdb; pdb.set_trace()
        x, c, c_mask = super().get_input(batch, self.first_stage_key, *args, **kwargs) # obtain source image and object patch
        # print('source image code shape:', np.shape(x), 'c code shape:', np.shape(c))
        # [1, 4, 64, 64], [1, 2, 257, 1024]
        # import pdb; pdb.set_trace()
        control = batch[self.control_key] # obtain condition image (hint)
        # print('control shape: ', np.shape(control)) # [1, 512, 512, 4] -> [1, 512, 512, 4, 2]
        # print(torch.max(x), torch.min(x))
        # print(torch.max(c), torch.min(c))
        # print(torch.max(c_mask), torch.min(c_mask))
        # print(torch.max(control), torch.min(control)) # [1, -1]
        # tensor(2.6057, device='cuda:0') tensor(-3.0823, device='cuda:0')     
        # tensor(4.2695, device='cuda:0', dtype=torch.float16) tensor(-3.7422, device='cuda:0', dtype=torch.float16) 
        # tensor(1.) tensor(0.)                                                

        # import pdb; pdb.set_trace()
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        # control = einops.rearrange(control, 'b h w c d -> b c h w d')
        control = control.to(memory_format=torch.contiguous_format).float()
        self.time_steps = batch['time_steps']
        return x, dict(c_crossattn=[c], c_concat=[control], c_mask=[c_mask])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)
        # print('c_crossattn shape', np.shape(cond_txt)) # [1, 257, 1024, 2]
        # wwwww
        hint = torch.cat(cond['c_concat'], 1)
        # print('c_concat shape', np.shape(hint)) # [1, 257, 1024, 2]
        # print(np.shape(hint)) # [1, 4, 512, 512]
        c_mask = torch.cat(cond['c_mask'], 1)
        # print('c_mask shape: ', np.shape(c_mask)) # [1, 1, 224, 224, 2]
        c_mask = c_mask.squeeze(1)
        # import pdb; pdb.set_trace()
        # print(np.shape(c_mask)) # [1, 224, 224, 2]
        # import pdb; pdb.set_trace()

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else: # check
            # import pdb; pdb.set_trace()
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt, c_mask=c_mask)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control, c_mask=c_mask)
        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N, obj_thr):
        # uncond = self.get_learned_conditioning([ torch.zeros((1, 3, 224, 224)) ] * N)
        x = [ torch.zeros((1, 3, 224, 224)) ] * N
        uc = []
        for i in range(obj_thr):
            uc_i = self.get_learned_conditioning(x)
            uc.append(uc_i)
        uc = torch.stack(uc)
        uc = uc.permute(1, 2, 3, 0)
        return uc
    # def get_unconditional_conditioning(self, N, obj_thr):
    #     uncond = self.get_learned_conditioning([ torch.zeros((1, 3, 224, 224)) ] * N)
    #     return uncond

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True, 
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        
        # import pdb; pdb.set_trace()
        c_cat, c, c_mask = c["c_concat"][0][:N], c["c_crossattn"][0][:N], c["c_mask"][0][:N]
        # print(np.shape(c_cat), np.shape(c), np.shape(c_mask))
        # import pdb; pdb.set_trace()
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z) 

        # ==== visualize the shape mask or the high-frequency map ====
        guide_mask = (c_cat[:,-1,:,:].unsqueeze(1) + 1) * 0.5
        guide_mask = torch.cat([guide_mask,guide_mask,guide_mask],1)
        HF_map  = c_cat[:,:3,:,:] #* 2.0 - 1.0

        log["control"] = HF_map

        cond_image = batch[self.cond_stage_key+'0'].cpu().numpy().copy() ####### need adjustment
        log["conditioning"] = torch.permute( torch.tensor(cond_image), (0,3,1,2)) * 2.0 - 1.0  
        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample: # not go into
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c], "c_mask": [c_mask]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0: # checked
            obj_thr = kwargs.get('obj_thr', 1)
            # print('obj_thr: ', obj_thr)
            # import pdb; pdb.set_trace()
            uc_cross = self.get_unconditional_conditioning(N, obj_thr)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross], "c_mask": [c_mask]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c], "c_mask": [c_mask]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg #* 2.0 - 1.0
        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        params += list(self.cond_stage_model.projector.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
