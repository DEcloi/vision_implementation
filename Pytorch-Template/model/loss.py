import torch.nn.functional as F
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)


def age_smoother(one_hot):
    _, idx = one_hot.max(1)
    n_class = one_hot.size(1)

    for l in range(one_hot.size(0)):
        # Self defined age continuity distribution
        age_safe_dist = torch.tensor([0.025, 0.175, 0.6, 0.175, 0.025]).cuda()
        pre = 2
        post = 3

        # For boundary cases
        if idx[l] >= n_class - 2:
            post = n_class - idx[l].item()
            age_safe_dist = age_safe_dist[:-3+post]
        if idx[l] < 2:
            pre = idx[l].item()
            age_safe_dist = age_safe_dist[2-pre:]

        # Substitute one-hot encoding with smooth labels
        one_hot[l][idx[l]-pre:idx[l]+post] = age_safe_dist

    return one_hot


def cross_entropy_with_age_smoothing(output, target, smoothing=True):
    if smoothing:

        # One-hot Encoding of label
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)

        # Regularize one-hot encoding with age-smoother
        one_hot = age_smoother(one_hot)

        # Cross entropy is defined as E_p[-log(q)] where p is the data distribution and q the model distribution
        # Need Log Probability for model outputs
        log_prb = F.log_softmax(output, dim=1)

        # Calculate Cross Entropy = - SUM[smoothed_label_distribution * log(model_distribution)]
        loss = -(one_hot * log_prb).sum(dim=1)

        # Take mean over batch (that are non-zeros)
        non_pad_mask = target.ne(0)
        loss = loss.masked_select(non_pad_mask).mean()

    else:
        loss = F.cross_entropy(output, target, reduction='mean')

    return loss


def focal_loss_with_age_smoothing(output, target, gamma=2, smoothing=True):
    if smoothing:

        # One-hot Encoding of label
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)

        # Regularize one-hot encoding with age-smoother
        one_hot = age_smoother(one_hot)

        # Cross entropy is defined as E_p[-log(q)] where p is the data distribution and q the model distribution
        # Need Log Probability for model outputs
        prb = F.softmax(output, dim=1) + 0.0000001
        log_prb = torch.log(prb)

        # Focal Loss weight
        weight = torch.pow(torch.tensor(1.).cuda() - prb, torch.tensor(gamma).cuda().to(output.dtype))
        focal = -1 * weight * log_prb
        loss = torch.sum(one_hot * focal, dim=1).mean()

    else:
        loss = F.cross_entropy(output, target, reduction='mean')

    return loss
