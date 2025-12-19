def detection_loss(outputs, targets):
    # 실제 실험에서는 ultralytics 내부 loss 사용
    return outputs.sum() * 0.0
