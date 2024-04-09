# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from tokenize import Double

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion

import torch.nn.functional as F
import scipy.stats as stats
import numpy as np

# nll_loss: negative log likelihood loss
def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True, weight=None):
    # print(lprobs.shape, target.shape) # torch.Size([8000, 35544]) torch.Size([8000]) 
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    nll_loss = nll_loss.squeeze(-1)
    smooth_loss = smooth_loss.squeeze(-1)
    # print(nll_loss.shape, smooth_loss.shape, weight.shape)
    if weight is not None:
        assert list(nll_loss.shape)[0]%list(weight.shape)[0]==0
        times=list(nll_loss.shape)[0]//list(weight.shape)[0]
        weight=weight.repeat_interleave(times)
    #     print(weight.sum()/list(weight.shape)[0])
    #     print(nll_loss, weight)
        nll_loss=nll_loss*weight
    #     print(nll_loss)
        smooth_loss=smooth_loss*weight
    #     print(nll_loss, smooth_loss)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    # print("eps_i", eps_i)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion("label_smoothed_cross_entropy_with_gaussian_context")
class LabelSmoothedCrossEntropyGaussianContextCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        balancing_lambda,
        inverse_softmax_tempreature,
        decrease_addictive_object,
        weight,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.l=balancing_lambda
        self.max_l=balancing_lambda
        self.T=1/inverse_softmax_tempreature
        self.gaussian=torch.zeros(1).cuda()
        self.decrease=decrease_addictive_object
        self.weight=weight # weight only works on label smoothed loss, not on addictive loss

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--report-accuracy', action='store_true',
                            help='report accuracy metric')
        parser.add_argument('--ignore-prefix-size', default=0, type=int,
                            help='Ignore first N tokens')
        parser.add_argument('--balancing-lambda', default=0.1, type=float,
                            help='balancing hyperparameter for addictive object')
        parser.add_argument('--inverse-softmax-tempreature', default=1., type=float,
                            help='inversed parameter T of softmax temperature mechanism')
        parser.add_argument('--decrease-addictive-object', default=0, type=int,
                            help='keep addictive loss for 10 epoch and decreace to 0 in 10 epoch')
        parser.add_argument('--weight', default=0, type=float,
                            help='set weight of translation set, if 0 then no special weight')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # model.current_epoch
        if self.decrease:
#             print(model.current_epoch)
            if model.current_epoch>10 and model.current_epoch<21:
                self.l=(20-model.current_epoch)/10*self.max_l
            elif model.current_epoch>20:
                self.l=0
        # print(self.weight)
        if self.weight>0:
            imgs_dist_tensor=sample["net_input"]["imgs_dist_list"][0]
#             imgs_dist_tensor=(imgs_dist_tensor+6)/8 # convert 2-10 to 1-2
            ##########
#             imgs_dist_tensor=(imgs_dist_tensor+14)/16 # convert 2-10 to 1-1.5
#             imgs_dist_tensor[imgs_dist_tensor<1]=0
#             imgs_dist_tensor=imgs_dist_tensor*self.weight
#             imgs_dist_tensor[imgs_dist_tensor<1]=1
            ##########
            imgs_dist_tensor=imgs_dist_tensor*self.weight
            imgs_dist_tensor=torch.add(imgs_dist_tensor, 1)
            #########
            # print(imgs_dist_tensor)
        else:
            imgs_dist_tensor=None
        # print(sample["net_input"])
        if "imgs_dist_list" in sample["net_input"]:
            del sample["net_input"]["imgs_dist_list"]
        net_output = model(**sample["net_input"])
        total_loss, loss, addictive_loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce, weight=imgs_dist_tensor)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": total_loss.data,
            "addictive_loss": addictive_loss.data,
            "nll_loss": nll_loss.data,
            "lambda":self.l,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        # print(model.attns.shape, sample["target"].size(), sample["net_input"]) 
        # torch.Size([16, 4, 12]) bsz x n_text x n_video
        # torch.Size([16, 6])
        # sample["net_input"]["src_tokens"].size = 16 x 4

        return total_loss.double(), sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output) # 2-dim
#         torch.set_printoptions(profile="full")
#         print(lprobs)
#         raise
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous() # 800*10
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True, weight=None):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        addictive_loss=self.compute_gaussian_loss(model.attns.double(), weight)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
            weight=weight, 
        )
        ##########
        if self.l==0:
            total_loss=loss
        else:
            total_loss=loss+addictive_loss
#         total_loss=loss
        ##########
        return total_loss, loss, addictive_loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    def compute_gaussian_loss(self, attns, weight):
        if (self.gaussian.equal(torch.zeros(1).cuda())):
            mu = 0
            variance = 1
            sigma = math.sqrt(variance)
            x = np.linspace(mu - 3*sigma, mu + 3*sigma, attns.shape[-1], dtype=np.double)
            # x = np.linspace(0, 10, 11)
            norm_x=torch.from_numpy(stats.norm.pdf(x, mu, sigma))
            self.gaussian=F.softmax(norm_x/self.T, dim=-1).double()
            self.gaussian.requires_grad=False
            self.gaussian=self.gaussian.cuda()
        gaussian=self.gaussian.repeat(attns.shape[0], attns.shape[1], 1)
        
        B=attns.shape[0]
        # print(attns.shape, gaussian.shape) # torch.Size([1136, 7, 12]) torch.Size([1136, 7, 12])
        # addictive_object=F.kl_div(attns.log(), gaussian, reduction="batchmean")
        addictive_loss=torch.cat([torch.nn.functional.kl_div(attns[i][None, :].log(), gaussian[i][None, :],reduction = 'batchmean').view(1) for i in range(B)]) # tensor([0.1, 0.2 , ...])
        if weight is not None:
            # print(addictive_loss.shape, weight.shape)
            addictive_loss=addictive_loss*weight
        addictive_object=self.l*torch.mean(addictive_loss)
        return addictive_object.double()

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        addictive_loss_sum = sum(log.get("addictive_loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
#         l = logging_outputs[0].get("lambda", 0)
        l = logging_outputs[0].get("lambda", 0)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "addictive_loss", addictive_loss_sum / sample_size / math.log(2), sample_size, round=8
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "lambda", l
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
