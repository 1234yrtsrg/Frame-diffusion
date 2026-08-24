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
        self.use_condition = config["model"].get("use_condition", False)
        self.condition_embed_dim = config["model"].get("condition_embed_dim", 0)

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.use_duration:
            self.emb_total_dim += self.duration_embed_dim
        if self.use_condition:
            self.emb_total_dim += self.condition_embed_dim
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
        if self.use_condition:
            self.condition_mlp = nn.Sequential(
                nn.Linear(1, self.condition_embed_dim),
                nn.SiLU(),
                nn.Linear(self.condition_embed_dim, self.condition_embed_dim),
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
        # DataParallel replicas may not expose parameters through parameters().
        # Registered buffers are replicated to the active device reliably.
        return self.alpha_torch.device

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

    def get_side_info(self, observed_tp, cond_mask, duration=None, condition=None):
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

        if self.use_condition:
            if condition is None:
                raise ValueError("condition is required when model.use_condition is true")
            condition = condition.to(device).float().view(B, 1)
            condition_embed = self.condition_mlp(condition)
            condition_embed = condition_embed.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, K, L
            )
            side_info = torch.cat([side_info, condition_embed], dim=1)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

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


class CSDI_Express4D(CSDI_base):
    """CSDI variant for 12-frame Express4D blendshape interpolation.

    The batch keeps complete ground-truth values in observed_data, while
    observed_mask marks endpoints plus any visible internal keyframes, and
    gt_mask marks every hidden target value.
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
        if self.use_condition:
            condition = batch.get("condition", None)
            if condition is None:
                raise KeyError("Express4D batch must contain condition when use_condition is enabled")
            side_condition = condition.to(device).float().view(-1)
        else:
            duration = batch.get("duration", batch.get("duraction"))
            if duration is None:
                raise KeyError("Express4D batch must contain duration")
            side_condition = duration.to(device).float().view(-1)

        # Dataset returns [B,L,K]; CSDI diffusion blocks use [B,K,L].
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        target_mask = target_mask.permute(0, 2, 1)

        return observed_data, observed_mask, observed_tp, target_mask, side_condition

    def _calc_loss_at_t(
        self,
        observed_data,
        cond_mask,
        target_mask,
        observed_tp,
        side_info,
        side_condition,
        is_train,
        set_t=-1,
        return_loss_components=False,
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
        denom = num_target.clamp_min(1.0)
        residual = (noise - predicted_noise) * target_mask
        diffusion_loss = (residual ** 2).sum() / denom

        x0_pred = (noisy_data - sqrt_one_minus_alpha * predicted_noise) / sqrt_alpha
        pred_full = cond_mask * observed_data + target_mask * x0_pred
        gt_full = observed_data

        recon_loss = torch.abs((x0_pred - gt_full) * target_mask).sum() / denom

        vel_loss, acc_loss = self.motion_losses(
            pred_full, gt_full, target_mask, observed_tp
        )

        range_low = torch.relu(self.clamp_min - x0_pred) ** 2
        range_high = torch.relu(x0_pred - self.clamp_max) ** 2
        range_loss = ((range_low + range_high) * target_mask).sum() / denom

        total_loss = (
            diffusion_loss
            + self.lambda_recon * recon_loss
            + self.lambda_vel * vel_loss
            + self.lambda_acc * acc_loss
            + self.lambda_range * range_loss
        )
        if return_loss_components:
            legacy_vel, legacy_acc = self.legacy_motion_losses(pred_full, gt_full, target_mask)
            return total_loss, {
                "diffusion": diffusion_loss,
                "recon": recon_loss,
                "velocity": vel_loss,
                "acceleration": acc_loss,
                "range": range_loss,
                "legacy_velocity": legacy_vel,
                "legacy_acceleration": legacy_acc,
            }
        return total_loss

    @staticmethod
    def _masked_mean(values, valid_mask):
        valid_mask = valid_mask.to(dtype=values.dtype)
        count = valid_mask.sum()
        return (values * valid_mask).sum() / count.clamp_min(1.0)

    @classmethod
    def motion_losses(cls, pred_full, gt_full, target_mask, timepoints):
        """L1 velocity/acceleration errors on target-touching irregular intervals."""
        if pred_full.shape != gt_full.shape or pred_full.shape != target_mask.shape:
            raise ValueError("pred_full, gt_full, and target_mask must have identical [B,K,L] shapes")
        if timepoints.ndim != 2 or timepoints.shape != (pred_full.shape[0], pred_full.shape[2]):
            raise ValueError("timepoints must have shape [B,L]")
        dt = timepoints[:, 1:] - timepoints[:, :-1]
        if torch.any(dt <= 0):
            raise ValueError("timepoints must be strictly increasing for every sample")

        dt_expanded = dt.unsqueeze(1)
        pred_velocity = (pred_full[:, :, 1:] - pred_full[:, :, :-1]) / dt_expanded
        gt_velocity = (gt_full[:, :, 1:] - gt_full[:, :, :-1]) / dt_expanded
        velocity_valid = torch.maximum(target_mask[:, :, 1:], target_mask[:, :, :-1])
        velocity_loss = cls._masked_mean(
            torch.abs(pred_velocity - gt_velocity), velocity_valid
        )

        pred_acceleration = 2.0 * (pred_velocity[:, :, 1:] - pred_velocity[:, :, :-1]) / (
            dt_expanded[:, :, 1:] + dt_expanded[:, :, :-1]
        )
        gt_acceleration = 2.0 * (gt_velocity[:, :, 1:] - gt_velocity[:, :, :-1]) / (
            dt_expanded[:, :, 1:] + dt_expanded[:, :, :-1]
        )
        acceleration_valid = torch.maximum(
            torch.maximum(target_mask[:, :, :-2], target_mask[:, :, 1:-1]),
            target_mask[:, :, 2:],
        )
        acceleration_loss = cls._masked_mean(
            torch.abs(pred_acceleration - gt_acceleration), acceleration_valid
        )
        return velocity_loss, acceleration_loss

    @classmethod
    def legacy_motion_losses(cls, pred_full, gt_full, target_mask):
        """Exact previous unit-step/all-position losses, retained only for scale reports."""
        pred_velocity = pred_full[:, :, 1:] - pred_full[:, :, :-1]
        gt_velocity = gt_full[:, :, 1:] - gt_full[:, :, :-1]
        velocity_loss = torch.mean(torch.abs(pred_velocity - gt_velocity))
        pred_acc = pred_full[:, :, 2:] - 2 * pred_full[:, :, 1:-1] + pred_full[:, :, :-2]
        gt_acc = gt_full[:, :, 2:] - 2 * gt_full[:, :, 1:-1] + gt_full[:, :, :-2]
        acceleration_loss = torch.mean(torch.abs(pred_acc - gt_acc))
        return velocity_loss, acceleration_loss

    def forward(self, batch, is_train=1, return_loss_components=False):
        observed_data, cond_mask, observed_tp, target_mask, side_condition = self.process_data(batch)
        side_info = self.get_side_info(
            observed_tp,
            cond_mask,
            duration=None if self.use_condition else side_condition,
            condition=side_condition if self.use_condition else None,
        )

        if is_train == 1:
            return self._calc_loss_at_t(
                observed_data,
                cond_mask,
                target_mask,
                observed_tp,
                side_info,
                side_condition,
                is_train,
                return_loss_components=return_loss_components,
            )

        loss_sum = 0
        component_sums = None
        for t in range(self.num_steps):
            result = self._calc_loss_at_t(
                observed_data, cond_mask, target_mask, observed_tp, side_info, side_condition,
                is_train, set_t=t, return_loss_components=return_loss_components,
            )
            if return_loss_components:
                loss_at_t, components = result
                component_sums = components if component_sums is None else {
                    key: component_sums[key] + value for key, value in components.items()
                }
            else:
                loss_at_t = result
            loss_sum += loss_at_t.detach()
        average_loss = loss_sum / self.num_steps
        if return_loss_components:
            return average_loss, {
                key: value.detach() / self.num_steps for key, value in component_sums.items()
            }
        return average_loss

    def evaluate(self, batch, n_samples):
        observed_data, cond_mask, observed_tp, target_mask, side_condition = self.process_data(batch)
        with torch.no_grad():
            side_info = self.get_side_info(
                observed_tp,
                cond_mask,
                duration=None if self.use_condition else side_condition,
                condition=side_condition if self.use_condition else None,
            )
            samples = self.impute(observed_data, cond_mask, side_info, n_samples)
        return samples, observed_data, target_mask, cond_mask, observed_tp

    def generate_middle(
        self,
        start,
        end,
        duration,
        num_samples=1,
        condition=None,
        timepoints=None,
    ):
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

        if self.use_condition:
            if condition is None:
                condition = duration
            if condition is None:
                raise ValueError("condition is required when model.use_condition is true")
            side_condition = torch.as_tensor(condition, device=device).float()
            if side_condition.dim() == 0:
                side_condition = side_condition.repeat(B)
            side_condition = side_condition.view(B)
        else:
            if duration is None:
                if self.use_duration:
                    raise ValueError("duration is required when model.use_duration is true")
                side_condition = torch.zeros(B, device=device)
            else:
                side_condition = torch.as_tensor(duration, device=device).float()
                if side_condition.dim() == 0:
                    side_condition = side_condition.repeat(B)
                side_condition = side_condition.view(B)

        observed_data = torch.zeros(B, K, self.seq_len, device=device)
        observed_data[:, :, 0] = start
        observed_data[:, :, -1] = end
        cond_mask = torch.zeros_like(observed_data)
        cond_mask[:, :, 0] = 1.0
        cond_mask[:, :, -1] = 1.0
        if timepoints is None:
            observed_tp = torch.arange(self.seq_len, device=device).float().unsqueeze(0).expand(B, -1)
        else:
            observed_tp = torch.as_tensor(timepoints, device=device).float()
            if observed_tp.dim() == 1:
                observed_tp = observed_tp.unsqueeze(0).expand(B, -1)
            if observed_tp.shape != (B, self.seq_len):
                raise ValueError(
                    f"timepoints must have shape [{self.seq_len}] or [B,{self.seq_len}], "
                    f"got {tuple(observed_tp.shape)}"
                )
            if torch.any(observed_tp[:, 1:] <= observed_tp[:, :-1]):
                raise ValueError("timepoints must be strictly increasing")

        with torch.no_grad():
            side_info = self.get_side_info(
                observed_tp,
                cond_mask,
                duration=None if self.use_condition else side_condition,
                condition=side_condition if self.use_condition else None,
            )
            samples = self.impute(observed_data, cond_mask, side_info, num_samples)
            samples = samples.permute(0, 1, 3, 2)[:, :, 1 : 1 + self.num_middle, :]
        if was_single:
            return samples[0]
        return samples
