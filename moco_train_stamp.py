import copy
import warnings
from typing import Optional
from typing import Union

import attr
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities import AttributeDict
from sklearn.linear_model import LogisticRegression

import utils
from batchrenorm import BatchRenorm1d
from lars import LARS


class Deconv2dBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=False, dilation=1, batch_norm=True):
        super(Deconv2dBnRelu, self).__init__()
        self.batch_norm = batch_norm
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding,
                                         groups, bias, dilation)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.batch_norm:
            return self.relu(self.bn(self.deconv(x)))
        else:
            return self.relu(self.deconv(x))


class SimpleDecoder(nn.Module):
    def __init__(self, batch_norm=True, width=1, input_dim=20, output_act='id'):
        super(SimpleDecoder, self).__init__()
        self.conv = Deconv2dBnRelu(in_channels=input_dim, out_channels=256 * width, kernel_size=3, padding=0, stride=1)
        self.deconv1 = Deconv2dBnRelu(in_channels=256 * width, out_channels=128 * width, kernel_size=3, padding=1,
                                      stride=2,
                                      output_padding=1, batch_norm=batch_norm)
        self.deconv2 = Deconv2dBnRelu(in_channels=128 * width, out_channels=64 * width, kernel_size=3, padding=1,
                                      stride=2,
                                      output_padding=1, batch_norm=batch_norm)
        self.deconv3 = Deconv2dBnRelu(in_channels=64 * width, out_channels=32 * width, kernel_size=3, padding=1,
                                      stride=2,
                                      output_padding=1, batch_norm=batch_norm)
        self.deconv4 = Deconv2dBnRelu(in_channels=32 * width, out_channels=16 * width, kernel_size=3, padding=1,
                                      stride=2,
                                      output_padding=1, batch_norm=batch_norm)
        self.deconv5 = Deconv2dBnRelu(in_channels=16 * width, out_channels=8 * width, kernel_size=3, padding=1,
                                      stride=2,
                                      output_padding=1, batch_norm=batch_norm)
        self.conv6 = nn.Conv2d(in_channels=8 * width, out_channels=3, kernel_size=3, padding=1, bias=False)

        if output_act == 'id':
            self.out = nn.Identity()
        elif output_act == 'sigmoid':
            self.out = torch.sigmoid
        elif output_act == 'tanh-3':
            self.out = lambda x: torch.tanh(x) * 3

    def forward(self, feature):
        feature = self.conv(feature[:, :, None, None])
        feature = self.deconv1(feature)
        feature = self.deconv2(feature)
        feature = self.deconv3(feature)
        feature = self.deconv4(feature)
        feature = self.deconv5(feature)
        pred_img = self.conv6(feature)
        pred_img = self.out(pred_img)
        return pred_img


from moco import MoCoMethodParams, get_mlp_normalization, MoCoMethod


class MoCoMethodStamp(MoCoMethod):
    model: torch.nn.Module
    dataset: utils.DatasetBase
    hparams: AttributeDict
    embedding_dim: Optional[int]

    def __init__(
            self, hparams: Union[MoCoMethodParams, dict, None] = None, **kwargs,
    ):
        super().__init__()

        if hparams is None:
            hparams = self.params(**kwargs)
        elif isinstance(hparams, dict):
            hparams = self.params(**hparams, **kwargs)

        self.hparams = AttributeDict(attr.asdict(hparams))

        # Check for configuration issues
        if (
                hparams.gather_keys_for_queue
                and not hparams.shuffle_batch_norm
                and not hparams.encoder_arch.startswith("ws_")
        ):
            warnings.warn(
                "Configuration suspicious: gather_keys_for_queue without shuffle_batch_norm or weight standardization"
            )

        some_negative_examples = hparams.use_negative_examples_from_batch or hparams.use_negative_examples_from_queue
        if hparams.loss_type == "ce" and not some_negative_examples:
            warnings.warn("Configuration suspicious: cross entropy loss without negative examples")

        # Create encoder model
        self.model = utils.get_encoder(hparams.encoder_arch)

        # Create dataset
        transforms = utils.MoCoTransforms(
            s=hparams.transform_s, crop_size=hparams.transform_crop_size, apply_blur=hparams.transform_apply_blur
        )
        self.dataset = utils.get_moco_dataset(hparams.dataset_name, transforms,
                                              id_weight=hparams.id_weight,
                                              id_type=hparams.id_type,
                                              strip_len=hparams.strip_len)

        # "key" function (no grad)
        self.lagging_model = copy.deepcopy(self.model)
        for param in self.lagging_model.parameters():
            param.requires_grad = False

        self.projection_model = utils.MLP(
            hparams.embedding_dim,
            hparams.dim,
            hparams.mlp_hidden_dim,
            num_layers=hparams.projection_mlp_layers,
            normalization=get_mlp_normalization(hparams),
            weight_standardization=hparams.use_mlp_weight_standardization,
        )

        self.prediction_model = utils.MLP(
            hparams.dim,
            hparams.dim,
            hparams.mlp_hidden_dim,
            num_layers=hparams.prediction_mlp_layers,
            normalization=get_mlp_normalization(hparams, prediction=True),
            weight_standardization=hparams.use_mlp_weight_standardization,
        )

        # decode model
        self.decoder_model = SimpleDecoder(
            input_dim=hparams.embedding_dim,
            output_act='tanh-3',
        )

        # stamp creator
        self.stamp_model = SimpleDecoder(
            input_dim=20,
            output_act='sigmoid',
        )

        #  "key" function (no grad)
        self.lagging_projection_model = copy.deepcopy(self.projection_model)
        for param in self.lagging_projection_model.parameters():
            param.requires_grad = False

        # this classifier is used to compute representation quality each epoch
        self.sklearn_classifier = LogisticRegression(max_iter=100, solver="liblinear")

        # create the queue
        self.register_buffer("queue", torch.randn(hparams.dim, hparams.K))
        self.queue = torch.nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        x, class_labels, x_code = batch

        if optimizer_idx == 1:
            """
            decoder
            """
            im_q = x[:, 0].contiguous()
            stamp = self.stamp_model(x_code)  # NCHW
            im_s = im_q - im_q * stamp * 0.5
            emb_s = self.model(im_s)
            im_pred = self.decoder_model(emb_s)
            decoder_loss = F.smooth_l1_loss(im_pred, im_q)
            log_data = {"step_decoder_loss": decoder_loss, }
            self.log_dict(log_data)
            return {"loss": decoder_loss}

        if optimizer_idx == 2:
            """
            stamp
            """
            im_q = x[:, 0].contiguous()
            stamp = self.stamp_model(x_code)  # NCHW
            im_s = im_q - im_q * stamp * 0.5
            emb_s = self.model(im_s)
            im_pred = self.decoder_model(emb_s)

            stamp_loss = - F.smooth_l1_loss(im_pred, im_q)
            reg_loss = F.mse_loss(stamp, torch.zeros_like(stamp))
            log_data = {"step_stamp_loss": stamp_loss, "step_reg_loss": reg_loss, }
            self.log_dict(log_data)
            return {"loss": stamp_loss + reg_loss * 0.0}

        if optimizer_idx == 0:
            stamp = self.stamp_model(x_code).detach()  # NCHW
            x_s = x.clone()
            x_s[:, 0] -= x_s[:, 0] * stamp * 0.5
            x_s[:, 1] -= x_s[:, 1] * stamp * 0.5

            emb_q, q, k = self._get_embeddings(x)
            logits, labels = self._get_contrastive_predictions(q, k)
            pos_ip, neg_ip = self._get_pos_neg_ip(emb_q, k)

            contrastive_loss = self._get_contrastive_loss(logits, labels)

            if self.hparams.use_both_augmentations_as_queries:
                x_flip = torch.flip(x, dims=[1])
                emb_q2, q2, k2 = self._get_embeddings(x_flip)
                logits2, labels2 = self._get_contrastive_predictions(q2, k2)

                pos_ip2, neg_ip2 = self._get_pos_neg_ip(emb_q2, k2)
                pos_ip = (pos_ip + pos_ip2) / 2
                neg_ip = (neg_ip + neg_ip2) / 2
                contrastive_loss += self._get_contrastive_loss(logits2, labels2)

            contrastive_loss = contrastive_loss.mean() * self.hparams.loss_constant_factor

            log_data = {"step_train_loss": contrastive_loss, "step_pos_cos": pos_ip, "step_neg_cos": neg_ip}

            with torch.no_grad():
                self._momentum_update_key_encoder()

            some_negative_examples = (
                    self.hparams.use_negative_examples_from_batch or self.hparams.use_negative_examples_from_queue
            )
            if some_negative_examples:
                acc1, acc5 = utils.calculate_accuracy(logits, labels, topk=(1, 5))
                log_data.update({"step_train_acc1": acc1, "step_train_acc5": acc5})

            # dequeue and enqueue
            if self.hparams.use_negative_examples_from_queue:
                self._dequeue_and_enqueue(k)

            self.log_dict(log_data)
            return {"loss": contrastive_loss}

    def configure_optimizers(self):
        # exclude bias and batch norm from LARS and weight decay
        regular_parameters = []
        regular_parameter_names = []
        excluded_parameters = []
        excluded_parameter_names = []
        for name, parameter in self.named_parameters():
            if parameter.requires_grad is False:
                continue
            if any(x in name for x in self.hparams.exclude_matching_parameters_from_lars):
                excluded_parameters.append(parameter)
                excluded_parameter_names.append(name)
            else:
                regular_parameters.append(parameter)
                regular_parameter_names.append(name)

        param_groups = [
            {"params": regular_parameters, "names": regular_parameter_names, "use_lars": True},
            {"params": excluded_parameters, "names": excluded_parameter_names, "use_lars": False, "weight_decay": 0, },
        ]
        if self.hparams.optimizer_name == "sgd":
            optimizer = torch.optim.SGD
        elif self.hparams.optimizer_name == "lars":
            optimizer = LARS
        else:
            raise NotImplementedError(f"No such optimizer {self.hparams.optimizer_name}")

        encoding_optimizer = optimizer(
            param_groups, lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            encoding_optimizer, self.hparams.max_epochs, eta_min=0
        )

        # decoder
        decoder_optimizer = torch.optim.Adam(self.decoder_model.parameters(), lr=self.hparams.lr)
        decoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            decoder_optimizer, self.hparams.max_epochs, eta_min=0
        )
        # stampe generator
        stamp_optimizer = torch.optim.Adam(self.stamp_model.parameters(), lr=self.hparams.lr)
        stamp_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            stamp_optimizer, self.hparams.max_epochs, eta_min=0
        )
        return [encoding_optimizer, decoder_optimizer, stamp_optimizer], [self.lr_scheduler, decoder_scheduler,
                                                                          stamp_scheduler]
