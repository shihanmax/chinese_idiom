import torch
import torch.nn as nn


def binary_focal_loss_with_logits(
    input: torch.Tensor,
    target: torch.Tensor,
    input_prob=False,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'none',
    eps: float = 1e-8,
):
    r"""Computes Binary Focal loss.
    
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        input: input data tensor with shape :math:`(N, 1, *)`.
        target: the target tensor with shape :math:`(N, 1, *)`.
        alpha: Weighting factor for the rare class :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: for numerically stability when dividing.
    Returns:
        the computed loss.
    Examples:
        >>> num_classes = 1
        >>> kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
        >>> logits = torch.tensor([[[[6.325]]],[[[5.26]]],[[[87.49]]]])
        >>> labels = torch.tensor([[[1.]],[[1.]],[[0.]]])
        >>> binary_focal_loss_with_logits(logits, labels, **kwargs)
        tensor(4.6052)
    """

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(
            f"Invalid input shape, we expect BxCx*. Got: {input.shape}"
        )

    if input.size(0) != target.size(0):
        raise ValueError(f"Expected input batch_size ({input.size(0)}) to "
                         f"match target batch_size ({target.size(0)}).")

    if input_prob:
        probs = input
    else:
        probs = torch.sigmoid(input)

    target = target.unsqueeze(dim=1)
    loss_tmp = (
        - alpha * torch.pow((1.0 - probs + eps), gamma) * target * torch.log(probs + eps) - (1 - alpha) * torch.pow(probs + eps, gamma) * (1.0 - target) * torch.log(1.0 - probs + eps))

    loss_tmp = loss_tmp.squeeze(dim=1)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss


class BinaryCls(nn.Module):
    
    def __init__(self, bert_model, bert_out_dim, hidden_dim, cls_dim=4):
        super(BinaryCls, self).__init__()
        self.bert_model = bert_model
        
        self.linear = nn.Linear(bert_out_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, cls_dim)
        self.tanh = nn.Tanh()
        
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, **input):
        bert_out = self.bert_model(**input)  # bs, seq_len, hidden
        
        cls_repr = bert_out.pooler_output  # bs, hidden
        
        cls_repr = self.dropout(cls_repr)
        
        hid = self.tanh(self.linear(cls_repr))
        out_prob = self.out_linear(hid)  # bs, 4
        
        return out_prob
        
    def forward_loss(self, out_prob, label):
        # loss = binary_focal_loss_with_logits(
        #     input=out_prob, target=label, input_prob=False, reduction="sum",
        # )
        loss = self.ce(out_prob, label)
        return loss
