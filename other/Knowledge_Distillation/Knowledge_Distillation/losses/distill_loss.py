import torch.nn.functional as F

def logit_distill_loss(student, teacher, T=4.0):
    s = F.log_softmax(student / T, dim=-1)
    t = F.softmax(teacher / T, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)


def feature_distill_loss(student_feat, teacher_feat):
    return F.mse_loss(student_feat, teacher_feat)
