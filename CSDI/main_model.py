import numpy as np
import torch
import torch.nn as nn
from diff_models import diff_CSDI


class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        self.use_duration = config["model"].get("use_duration", False)
        self.duration_embed_dim = config["model"].get("duration_embed_dim", 0)

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.use_duration:
            self.emb_total_dim += self.duration_embed_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )
        if self.use_duration:
            self.duration_mlp = nn.Sequential(
                nn.Linear(1, self.duration_embed_dim),
                nn.SiLU(),
                nn.Linear(self.duration_embed_dim, self.duration_embed_dim),
            )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.register_buffer(
            "alpha_torch",
            torch.tensor(self.alpha).float().unsqueeze(1).unsqueeze(1),
        )

    def get_device(self):
        return next(self.parameters()).device

    def time_embedding(self, pos, d_model=128):
        device = pos.device
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:  # draw another sample for histmask (i-1 corresponds to another sample)
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1] 
        return cond_mask

    def get_test_pattern_mask(self, observed_mask, test_pattern_mask):
        return observed_mask * test_pattern_mask


    def get_side_info(self, observed_tp, cond_mask, duration=None):
        B, K, L = cond_mask.shape
        device = cond_mask.device

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.use_duration:
            if duration is None:
                raise ValueError("duration is required when model.use_duration is true")
            duration = duration.to(device).float().view(B, 1)
            duration_embed = self.duration_mlp(duration)
            duration_embed = duration_embed.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, K, L
            )
            side_info = torch.cat([side_info, duration_embed], dim=1)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                predicted = self.diffmodel(
                    diff_input,
                    side_info,
                    torch.tensor([t]).to(observed_data.device),
                )

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
        else:
            cond_mask = self.get_randmask(observed_mask)

        side_info = self.get_side_info(observed_tp, cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp


class CSDI_PM25(CSDI_base):
    def __init__(self, config, device, target_dim=36):
        super(CSDI_PM25, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CSDI_Physio(CSDI_base):
    def __init__(self, config, device, target_dim=35):
        super(CSDI_Physio, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )



class CSDI_Forecasting(CSDI_base):
    def __init__(self, config, device, target_dim):
        super(CSDI_Forecasting, self).__init__(target_dim, config, device)
        self.target_dim_base = target_dim
        self.num_sample_features = config["model"]["num_sample_features"]

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        feature_id=torch.arange(self.target_dim_base).unsqueeze(0).expand(observed_data.shape[0],-1).to(self.device)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            feature_id, 
        )        

    def sample_features(self,observed_data, observed_mask,feature_id,gt_mask):
        size = self.num_sample_features
        self.target_dim = size
        extracted_data = []
        extracted_mask = []
        extracted_feature_id = []
        extracted_gt_mask = []
        
        for k in range(len(observed_data)):
            ind = np.arange(self.target_dim_base)
            np.random.shuffle(ind)
            extracted_data.append(observed_data[k,ind[:size]])
            extracted_mask.append(observed_mask[k,ind[:size]])
            extracted_feature_id.append(feature_id[k,ind[:size]])
            extracted_gt_mask.append(gt_mask[k,ind[:size]])
        extracted_data = torch.stack(extracted_data,0)
        extracted_mask = torch.stack(extracted_mask,0)
        extracted_feature_id = torch.stack(extracted_feature_id,0)
        extracted_gt_mask = torch.stack(extracted_gt_mask,0)
        return extracted_data, extracted_mask,extracted_feature_id, extracted_gt_mask


    def get_side_info(self, observed_tp, cond_mask,feature_id=None):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, self.target_dim, -1)

        if self.target_dim == self.target_dim_base:
            feature_embed = self.embed_layer(
                torch.arange(self.target_dim).to(self.device)
            )  # (K,emb)
            feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        else:
            feature_embed = self.embed_layer(feature_id).unsqueeze(1).expand(-1,L,-1,-1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            feature_id, 
        ) = self.process_data(batch)
        if is_train == 1 and (self.target_dim_base > self.num_sample_features):
            observed_data, observed_mask,feature_id,gt_mask = \
                    self.sample_features(observed_data, observed_mask,feature_id,gt_mask)
        else:
            self.target_dim = self.target_dim_base
            feature_id = None

        if is_train == 0:
            cond_mask = gt_mask
        else: #test pattern
            cond_mask = self.get_test_pattern_mask(
                observed_mask, gt_mask
            )

        side_info = self.get_side_info(observed_tp, cond_mask, feature_id)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)



    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            feature_id, 
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask * (1-gt_mask)

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

        return samples, observed_data, target_mask, observed_mask, observed_tp


class CSDI_Express4D(CSDI_base):
    """CSDI variant for 12-frame Express4D blendshape interpolation.

    The batch keeps complete ground-truth values in observed_data, while
    observed_mask marks only the two endpoint conditions and gt_mask marks the
    ten middle target frames.
    """

    def __init__(self, config, device, target_dim=52):
        super(CSDI_Express4D, self).__init__(target_dim, config, device)
        loss_config = config.get("loss", {})
        self.lambda_recon = float(loss_config.get("lambda_recon", 1.0))
        self.lambda_vel = float(loss_config.get("lambda_vel", 0.5))
        self.lambda_acc = float(loss_config.get("lambda_acc", 0.2))
        self.lambda_range = float(loss_config.get("lambda_range", 0.1))
        dataset_config = config.get("dataset", {})
        self.seq_len = int(dataset_config.get("seq_len", config["model"].get("seq_len", 12)))
        self.num_middle = int(dataset_config.get("num_middle", config["model"].get("num_middle", 10)))
        self.clamp_min = float(dataset_config.get("clamp_min", 0.0))
        self.clamp_max = float(dataset_config.get("clamp_max", 1.0))

    def process_data(self, batch):
        device = self.get_device()
        observed_data = batch["observed_data"].to(device).float()
        observed_mask = batch["observed_mask"].to(device).float()
        observed_tp = batch["timepoints"].to(device).float()
        target_mask = batch["target_mask"] if "target_mask" in batch else batch["gt_mask"]
        target_mask = target_mask.to(device).float()
        duration = batch.get("duration", batch.get("duraction"))
        if duration is None:
            raise KeyError("Express4D batch must contain duration")
        duration = duration.to(device).float().view(-1)

        # Dataset returns [B,L,K]; CSDI diffusion blocks use [B,K,L].
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        target_mask = target_mask.permute(0, 2, 1)

        return observed_data, observed_mask, observed_tp, target_mask, duration

    def _calc_loss_at_t(
        self,
        observed_data,
        cond_mask,
        target_mask,
        side_info,
        duration,
        is_train,
        set_t=-1,
    ):
        B, K, L = observed_data.shape
        if is_train != 1:
            t = (torch.ones(B) * set_t).long().to(observed_data.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(observed_data.device)

        current_alpha = self.alpha_torch[t]
        sqrt_alpha = current_alpha ** 0.5
        sqrt_one_minus_alpha = (1.0 - current_alpha) ** 0.5
        noise = torch.randn_like(observed_data)
        noisy_data = sqrt_alpha * observed_data + sqrt_one_minus_alpha * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        predicted_noise = self.diffmodel(total_input, side_info, t)

        num_target = target_mask.sum()
        denom = num_target if num_target > 0 else 1
        residual = (noise - predicted_noise) * target_mask
        diffusion_loss = (residual ** 2).sum() / denom

        x0_pred = (noisy_data - sqrt_one_minus_alpha * predicted_noise) / sqrt_alpha
        pred_full = cond_mask * observed_data + target_mask * x0_pred
        gt_full = observed_data

        recon_loss = torch.abs((x0_pred - gt_full) * target_mask).sum() / denom

        pred_velocity = pred_full[:, :, 1:] - pred_full[:, :, :-1]
        gt_velocity = gt_full[:, :, 1:] - gt_full[:, :, :-1]
        vel_loss = torch.mean(torch.abs(pred_velocity - gt_velocity))

        pred_acc = pred_full[:, :, 2:] - 2 * pred_full[:, :, 1:-1] + pred_full[:, :, :-2]
        gt_acc = gt_full[:, :, 2:] - 2 * gt_full[:, :, 1:-1] + gt_full[:, :, :-2]
        acc_loss = torch.mean(torch.abs(pred_acc - gt_acc))

        pred_middle = x0_pred[:, :, 1 : 1 + self.num_middle]
        range_low = torch.relu(self.clamp_min - pred_middle) ** 2
        range_high = torch.relu(pred_middle - self.clamp_max) ** 2
        range_loss = torch.mean(range_low + range_high)

        total_loss = (
            diffusion_loss
            + self.lambda_recon * recon_loss
            + self.lambda_vel * vel_loss
            + self.lambda_acc * acc_loss
            + self.lambda_range * range_loss
        )
        return total_loss

    def forward(self, batch, is_train=1):
        observed_data, cond_mask, observed_tp, target_mask, duration = self.process_data(batch)
        side_info = self.get_side_info(observed_tp, cond_mask, duration=duration)

        if is_train == 1:
            return self._calc_loss_at_t(
                observed_data, cond_mask, target_mask, side_info, duration, is_train
            )

        loss_sum = 0
        for t in range(self.num_steps):
            loss_sum += self._calc_loss_at_t(
                observed_data,
                cond_mask,
                target_mask,
                side_info,
                duration,
                is_train,
                set_t=t,
            ).detach()
        return loss_sum / self.num_steps

    def evaluate(self, batch, n_samples):
        observed_data, cond_mask, observed_tp, target_mask, duration = self.process_data(batch)
        with torch.no_grad():
            side_info = self.get_side_info(observed_tp, cond_mask, duration=duration)
            samples = self.impute(observed_data, cond_mask, side_info, n_samples)
        return samples, observed_data, target_mask, cond_mask, observed_tp

    def generate_middle(self, start, end, duration, num_samples=1):
        device = self.get_device()
        was_single = start.dim() == 1
        if was_single:
            start = start.unsqueeze(0)
            end = end.unsqueeze(0)
        start = start.to(device).float()
        end = end.to(device).float()
        B, K = start.shape
        if K != self.target_dim or end.shape != (B, K):
            raise ValueError(
                f"start/end must have shape [52] or [B,52], got {tuple(start.shape)} and {tuple(end.shape)}"
            )

        duration = torch.as_tensor(duration, device=device).float()
        if duration.dim() == 0:
            duration = duration.repeat(B)
        duration = duration.view(B)

        observed_data = torch.zeros(B, K, self.seq_len, device=device)
        observed_data[:, :, 0] = start
        observed_data[:, :, -1] = end
        cond_mask = torch.zeros_like(observed_data)
        cond_mask[:, :, 0] = 1.0
        cond_mask[:, :, -1] = 1.0
        observed_tp = torch.arange(self.seq_len, device=device).float().unsqueeze(0).expand(B, -1)

        with torch.no_grad():
            side_info = self.get_side_info(observed_tp, cond_mask, duration=duration)
            samples = self.impute(observed_data, cond_mask, side_info, num_samples)
            samples = samples.permute(0, 1, 3, 2)[:, :, 1 : 1 + self.num_middle, :]
        if was_single:
            return samples[0]
        return samples
