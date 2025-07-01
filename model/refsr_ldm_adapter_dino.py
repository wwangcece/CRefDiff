import os
from PIL import Image
import cv2
import torch
import numpy as np
import math
from torch.nn import functional as F
from ldm.modules.diffusionmodules.util import (
    timestep_embedding,
)
from ldm.modules.diffusionmodules.openaimodel import (
    UNetModel,
)
from ldm.util import default
from ldm.models.diffusion.ddpm import LatentDiffusion
from model.modules import ImplicitPromptModule
from .spaced_sampler import SpacedSampler
from img_utils.metrics import LPIPS
from .adapters import Spade_Adapter, LCA_Adapter, Cat_Adapter


# Do forward process for UNetModel with prepared "control" tensors
class ControlledUnetModel(UNetModel):
    def forward(
        self,
        x,
        timesteps=None,
        context=None,
        control=None,
        only_mid_control=False,
        **kwargs,
    ):
        # "control" is output of "ControlNet" model
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        h = x.type(self.dtype)
        for i, module in enumerate(self.input_blocks):
            h = module(h, emb, context)
            if (i - 1) % 3 == 0 and ((i - 1) / 3 < len(control)):
                curr_control = control[int((i - 1) / 3)]
                if (
                    curr_control.shape[2] != h.shape[2]
                    or curr_control.shape[3] != h.shape[3]
                ):
                    curr_control = F.interpolate(
                        curr_control, (h.shape[2], h.shape[3]), mode="nearest"
                    )
                h = h + curr_control
            hs.append(h)

        h = self.middle_block(h, emb, context)

        for i, module in enumerate(self.output_blocks):
            h_enc = hs.pop()
            if h_enc.shape[2] != h.shape[2] or h_enc.shape[3] != h.shape[3]:
                h = F.interpolate(h, (h_enc.shape[2], h_enc.shape[3]), mode="nearest")
            h = torch.cat([h, h_enc], dim=1)
            h = module(h, emb, context)
            if i % 3 == 0 and ((3 - i / 3) < len(control)):
                curr_control = control[int(3 - i / 3)]
                if (
                    curr_control.shape[2] != h.shape[2]
                    or curr_control.shape[3] != h.shape[3]
                ):
                    curr_control = F.interpolate(
                        curr_control, (h.shape[2], h.shape[3]), mode="nearest"
                    )
                h = h + curr_control

        h = h.type(x.dtype)
        return self.out(h)


class ControlLDM(LatentDiffusion):
    def __init__(
        self,
        lr_key: str,
        ref_key: str,
        sd_locked: bool,
        only_mid_control: bool,
        learning_rate: float,
        disable_preprocess=False,
        frozen_diff=True,
        use_map=True,
        *args,
        **kwargs,
    ) -> "ControlLDM":
        super().__init__(*args, **kwargs)
        # instantiate control module
        self.adapter = LCA_Adapter(use_map=use_map)
        self.lr_key = lr_key
        self.ref_key = ref_key
        self.disable_preprocess = disable_preprocess
        self.sd_locked = sd_locked
        self.only_mid_control = only_mid_control
        self.learning_rate = learning_rate
        self.frozen_diff = frozen_diff
        self.proj_image = ImplicitPromptModule(image_feat_dim=768, num_queries=96)

        self.lpips_metric = LPIPS(net="alex")

        if self.frozen_diff:
            self.model.eval()
            # self.model.train = disabled_train
            for name, param in self.model.named_parameters():
                if "attn" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    def apply_cond_ref_encoder(self, control_lr, control_ref, sim_lamuda=None):
        cond_latent = self.adapter(control_lr, control_ref, sim_lamuda=sim_lamuda)
        cond_latent = [cond * self.scale_factor for cond in cond_latent]
        return cond_latent

    @torch.no_grad()
    def get_learned_conditioning(self, c):
        return self.cond_stage_model(c)

    def get_input(self, batch, bs=None, sim_lamuda=None, *args, **kwargs):
        with torch.no_grad():
            x, c = super().get_input(
                batch, self.hr_key, cond_key=self.ref_key, *args, **kwargs
            )
            # x: HR encoded
            # c: conditional text
            lr_cond = batch[self.lr_key]
            ref_cond = batch[self.ref_key]
            hr_cond = batch[self.hr_key]

            if bs is not None:
                lr_cond = lr_cond[:bs]
                ref_cond = ref_cond[:bs]
                hr_cond = hr_cond[:bs]

            lr_cond = lr_cond.to(self.device)
            ref_cond = ref_cond.to(self.device)
            hr_cond = hr_cond.to(self.device)

            lr_cond = lr_cond.to(memory_format=torch.contiguous_format).float()
            ref_cond = ref_cond.to(memory_format=torch.contiguous_format).float()
            hr_cond = hr_cond.to(memory_format=torch.contiguous_format).float()

        # apply condition encoder
        cond_latent = self.apply_cond_ref_encoder(
            lr_cond, ref_cond, sim_lamuda=sim_lamuda
        )
        # apply image projection
        c = self.proj_image(c, sim_lamuda=sim_lamuda)

        return x, dict(
            c_crossattn=[c],
            cond_latent=[cond_latent],
            lq=[lr_cond],
            ref=[ref_cond],
            hr=[hr_cond],
        )

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond["c_crossattn"], 1)
        cond_control = cond["cond_latent"][0]

        eps = diffusion_model(
            x=x_noisy,
            timesteps=t,
            context=cond_txt,
            control=cond_control,
            only_mid_control=self.only_mid_control,
        )

        return eps

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = "train" if self.training else "val"

        if self.parameterization == "x0":
            target = x_start
            model_output = self.predict_start_from_noise(x_noisy, t, model_output)
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        elif self.parameterization == "x0_pixel":
            print("x0-pixel loss function!")
            sampler = SpacedSampler(self)
            sampler.make_schedule(num_steps=1000)
            model_output = self.decode_first_stage(
                sampler._predict_xstart_from_eps(x_noisy, t, model_output)
            )
            target = cond["hq"][0]
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f"{prefix}/loss_simple": loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f"{prefix}/loss_gamma": loss.mean()})
            loss_dict.update({"logvar": self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f"{prefix}/loss_vlb": loss_vlb})
        loss += self.original_elbo_weight * loss_vlb
        loss_dict.update({f"{prefix}/loss": loss})

        return loss, loss_dict

    @torch.no_grad()
    def log_images(self, batch, sim_lamuda=None, sample_steps=50):
        log = dict()
        # [0 ,1]
        log["hq"] = (batch[self.hr_key] + 1) / 2
        # [0, 1]
        log["lq"] = (batch[self.lr_key] + 1) / 2
        # [0, 1]
        log["ref"] = (batch[self.ref_key] + 1) / 2

        z, cond = self.get_input(batch, sim_lamuda=sim_lamuda)

        samples = self.sample_log(
            cond=cond,
            steps=sample_steps,
        )
        # [0, 1]
        log["samples"] = samples

        return log

    @torch.no_grad()
    def sample_log(self, cond, steps):
        sampler = SpacedSampler(self)
        b, c, h, w = cond["cond_latent"][0][0].shape
        shape = (b, self.channels, h, w)
        samples = sampler.sample(steps, shape, cond)
        return samples

    def configure_optimizers(self):
        lr = self.learning_rate
        params = (
            list(self.adapter.parameters())
            + list(self.model.parameters())
            + list(self.proj_image.parameters())
        )
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def on_validation_epoch_start(self):
        # PSNR、LPIPS metrics are set zero
        self.val_psnr = 0
        self.val_lpips = 0

    def validation_step(self, batch, batch_idx):
        val_results = self.log_images(batch)

        save_dir = os.path.join(
            self.logger.save_dir, "validation", f"step--{self.global_step}"
        )
        os.makedirs(save_dir, exist_ok=True)
        # calculate psnr
        # bchw;[0, 1];tensor
        hr_batch_tensor = val_results["hq"].detach().cpu()
        sr_batch_tensor = val_results["samples"].detach().cpu()
        this_psnr = 0
        for i in range(len(hr_batch_tensor)):
            curr_hr = hr_batch_tensor[i].numpy().astype(np.float64)
            curr_sr = sr_batch_tensor[i].numpy().astype(np.float64)
            curr_psnr = 20 * math.log10(
                1.0 / math.sqrt(np.mean((curr_hr - curr_sr) ** 2))
            )
            self.val_psnr += curr_psnr
            this_psnr += curr_psnr
        this_psnr /= len(hr_batch_tensor)

        # calculate lpips
        this_lpips = 0
        hq = val_results["hq"].clamp_(0, 1).detach().cpu()
        pred = val_results["samples"].clamp_(0, 1).detach().cpu()
        curr_lpips = self.lpips_metric(hq, pred).sum().item()
        self.val_lpips += curr_lpips
        this_lpips = curr_lpips / len(hr_batch_tensor)

        # log metrics out
        self.log("val_psnr", this_psnr)
        self.log("val_lpips", this_lpips)

        # save images
        for image_key in val_results:
            os.makedirs(os.path.join(save_dir, image_key), exist_ok=True)
            image = val_results[image_key].detach().cpu()
            N = len(image)

            for i in range(N):
                img_name = os.path.splitext(os.path.basename(batch["path"][i]))[0]
                curr_img = image[i]
                curr_img = curr_img.transpose(0, 1).transpose(1, 2).numpy()
                curr_img = (curr_img * 255).clip(0, 255).astype(np.uint8)
                filename = "{}_{}.png".format(img_name, image_key)
                path = os.path.join(os.path.join(save_dir, image_key), filename)
                Image.fromarray(curr_img).save(path)

    def on_validation_epoch_end(self):
        # calculate average metrics
        self.val_psnr /= self.trainer.datamodule.val_config.dataset.params.data_len
        self.val_lpips /= self.trainer.datamodule.val_config.dataset.params.data_len
        # make saving dir
        save_dir = os.path.join(
            self.logger.save_dir, "validation", f"step--{self.global_step}"
        )
        save_dir = os.path.join(
            save_dir, f"psnr-{round(self.val_psnr, 2)}-lpips-{round(self.val_lpips, 2)}"
        )

    def validation_inference(self, batch, batch_idx, save_dir, sim_lamuda=None):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        starter.record()
        val_results = self.log_images(batch, sim_lamuda=sim_lamuda)
        ender.record()
        torch.cuda.synchronize()
        elapsed_time = starter.elapsed_time(ender)  # ms
        max_memory = torch.cuda.memory_allocated() / 1024**2  # MB

        print(f"[Inference Time] {elapsed_time:.2f} ms")
        print(f"[Current Memory] {max_memory:.2f} MB")

        os.makedirs(save_dir, exist_ok=True)

        # save images
        for image_key in val_results:
            image = val_results[image_key].detach().cpu()
            N = len(image)

            for i in range(N):
                img_name = os.path.splitext(os.path.basename(batch["path"][i]))[0]
                curr_img = image[i]
                curr_img = curr_img.transpose(0, 1).transpose(1, 2).numpy()
                curr_img = (curr_img * 255).clip(0, 255).astype(np.uint8)
                filename = "{}_{}.png".format(img_name, image_key)
                path = os.path.join(save_dir, filename)
                Image.fromarray(curr_img).save(path)

    def visual_sim_map(
        self,
        batch,
        batch_idx,
        save_dir,
        attn_val_list=None,
        return_cos_sim_map=False,
        return_learned_sim_map=False,
    ):
        lr_cond = batch[self.lr_key]
        ref_cond = batch[self.ref_key]

        lr_cond = lr_cond.to(self.device)
        ref_cond = ref_cond.to(self.device)

        lr_cond = lr_cond.to(memory_format=torch.contiguous_format).float()
        ref_cond = ref_cond.to(memory_format=torch.contiguous_format).float()

        _, sim_map_list = self.adapter(
            lr_cond,
            ref_cond,
            return_cos_sim_map=return_cos_sim_map,
            return_learned_sim_map=return_learned_sim_map,
        )

        for i in range(sim_map_list[0].shape[0]):
            img_name = os.path.splitext(os.path.basename(batch["path"][i]))[0]
            img_save_path = os.path.join(save_dir, img_name)
            os.makedirs(img_save_path, exist_ok=True)
            attn_val_list.append(sim_map_list[0][i].mean().item())

            for j in range(len(sim_map_list)):
                sim_map = sim_map_list[j][i].unsqueeze(0).detach().cpu()  # [1, 1, h, w]

                # 上采样到480×480
                sim_map = F.interpolate(sim_map, size=(480, 480), mode="nearest")
                # sim_map = (sim_map**0.5).clip(0, 1)
                sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min()) * 0.9

                # squeeze到[h, w]
                sim_map_vis = sim_map.squeeze(0).squeeze(0)  # [h, w]

                # 转成numpy，归一化到[0,255]
                sim_map_np = (sim_map_vis.numpy() * 255).astype(np.uint8)  # uint8

                # 使用OpenCV的applyColorMap生成热力图
                sim_map_color = cv2.applyColorMap(
                    sim_map_np, cv2.COLORMAP_RAINBOW
                )  # (h, w, 3), BGR

                # BGR转RGB
                sim_map_color = cv2.cvtColor(sim_map_color, cv2.COLOR_BGR2RGB)

                # 保存
                save_path = os.path.join(img_save_path, f"{img_name}_{j}.png")
                cv2.imwrite(save_path, sim_map_color)

    def visual_steps(self, batch, batch_idx, save_dir):
        sampler = SpacedSampler(self)
        z, cond = self.get_input(batch)
        b, c, h, w = cond["cond_latent"][0][0].shape
        shape = (b, self.channels, h, w)
        pixel_each_step = sampler.sample(50, shape, cond, return_each_step=True)
        for i in range(pixel_each_step[0].shape[0]):
            # the i-th image
            img_name = os.path.splitext(os.path.basename(batch["path"][i]))[0]
            os.makedirs(os.path.join(save_dir, img_name), exist_ok=True)
            for j in range(len(pixel_each_step)):
                # the j-th step
                curr_img = pixel_each_step[j][i].detach().cpu()
                curr_img = curr_img.transpose(0, 1).transpose(1, 2).numpy()
                curr_img = (curr_img * 255).clip(0, 255).astype(np.uint8)
                filename = "{}_{}.png".format(img_name, j)
                path = os.path.join(os.path.join(save_dir, img_name), filename)
                Image.fromarray(curr_img).save(path)
