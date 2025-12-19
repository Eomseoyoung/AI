import torch
import torch.nn as nn
from models.teacher import load_teacher
from models.student import load_student
from hooks.feature_hook import FeatureHook
from losses.distill_loss import logit_distill_loss, feature_distill_loss

C_STUDENT = 256   # YOLOv8n neck channel
C_TEACHER = 640   # YOLOv8x neck channel

device = "cuda" if torch.cuda.is_available() else "cpu"

teacher = load_teacher(device)
student = load_student(device)



align_conv = nn.Conv2d(
    in_channels=C_STUDENT,
    out_channels=C_TEACHER,
    kernel_size=1,
    bias=False
).to(device)

teacher_hook = FeatureHook()
student_hook = FeatureHook()

# Neck 마지막 레이어 (YOLOv8 기준, 실험용)
teacher.model[10].register_forward_hook(teacher_hook)
student.model[10].register_forward_hook(student_hook)

optimizer = torch.optim.Adam(
    list(student.parameters()) + list(align_conv.parameters()),
    lr=1e-4
)

# 더미 입력 (실제 이미지 대신)
dummy_input = torch.randn(1, 3, 640, 640).to(device)

for step in range(100):
    teacher_hook.clear()
    student_hook.clear()

    with torch.no_grad():
        t_out = teacher(dummy_input)

    s_out = student(dummy_input)

    # 더미 logit (실제론 cls head에서 추출)
    loss_logit = logit_distill_loss(
        s_out[0].mean(),
        t_out[0].mean()
    )

    student_feat = student_hook.features[-1]
    teacher_feat = teacher_hook.features[-1]

    # Student feature를 Teacher channel로 정렬
    student_feat_aligned = align_conv(student_feat)

    loss_feat = feature_distill_loss(
        student_feat_aligned,
        teacher_feat
    )
    loss = loss_logit + loss_feat

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"[Step {step}] Loss: {loss.item():.4f}")
