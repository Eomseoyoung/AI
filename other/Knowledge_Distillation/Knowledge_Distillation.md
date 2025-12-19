
# Knowledge Distillation 기반 YOLO 모델 경량화 실험 프로젝트

## 1. 프로젝트 개요 (Overview)

본 프로젝트는 대형 Vision 모델(Teacher)의 지식을 경량 모델(Student)에 전이하는  
**Knowledge Distillation 기반 파인튜닝 구조**를 실험적으로 구현한 예제입니다.

YOLOv8 계열 모델을 대상으로,

- 대형 모델의 **출력 분포(Logit)**
- 대형 모델의 **중간 Feature 표현**

을 Student 모델이 모방하도록 학습함으로써,  
**모델 크기를 줄이면서도 성능 저하를 최소화하는 구조적 원리**를 이해하는 것을 목표로 합니다.

> 본 프로젝트는 **최종 성능(mAP) 향상**이 목적이 아니라,  
> Knowledge Distillation의 **개념·구조·구현 흐름을 이해**하는 데 초점을 둡니다.

---

## 2. 프로젝트 목적 (Goal)

- Knowledge Distillation의 개념을 **코드 구조 단위로 이해**
- Teacher–Student 학습 구조의 실제 구현 방식 학습
- Feature Distillation 과정에서 발생하는 **차원 불일치 문제와 해결 방식** 체험
- 추후 **실제 YOLO 학습, Quantization, TensorRT 배포**로 확장 가능한 기반 확보

---

## 3. 프로젝트 범위 (Scope)

### 포함 사항
- YOLOv8 Teacher / Student 모델 로딩
- Forward Hook을 이용한 중간 Feature 추출
- Logit Distillation / Feature Distillation Loss 구현
- 1×1 Convolution 기반 Feature Alignment
- Distillation 기반 파인튜닝 학습 루프 구성

### 제외 사항
- 실제 데이터셋 기반 학습
- mAP 성능 평가
- Ultralytics Trainer 내부 로직 수정
- 실서비스 배포

---

## 4. 프로젝트 디렉터리 구조

```text
yolo_distillation/
├── train_distill.py          # 메인 실행 파일
├── requirements.txt          # 의존성 정의
├── models/
│   ├── teacher.py            # Teacher 모델 로딩
│   └── student.py            # Student 모델 로딩
├── hooks/
│   └── feature_hook.py       # 중간 Feature 추출용 Hook
├── losses/
│   ├── distill_loss.py       # Distillation 전용 Loss
│   └── detection_loss.py     # (더미) Detection Loss 위치
└── data/
    └── dummy.txt             # 데이터 위치 표시용
