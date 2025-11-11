
# 🚀 BiSeNet V2 (Bilateral Segmentation Network V2) 설명
BiSeNet V2는 **실시간 의미론적 분할(Real-Time Semantic Segmentation)**을 위해 고안된 효율적이고 효과적인 신경망 구조입니다. 
의미론적 분할은 이미지의 모든 픽셀을 해당 클래스(예: 하늘, 자동차, 건물)로 분류하는 작업입니다.

BiSeNet V2의 핵심 목표는 **정확도(Accuracy)**와 속도(Speed) 사이의 균형을 맞추는 것입니다.
이는 일반적인 분할 모델(예: FCN, DeepLab)이 고해상도 정보와 문맥적 정보를 모두 처리하느라 속도가 느려지는 문제를 해결하기 위해 두 개의 병렬 브랜치를 사용하는 것이 특징입니다.

## BiSeNet V2의 핵심 구조
BiSeNet V2는 서로 다른 정보를 담당하는 두 개의 주요 브랜치와 이를 결합하는 레이어로 구성됩니다.

# 1. Detail Branch (세부 정보 브랜치)
목표: 고해상도의 공간적 세부 정보(Spatial Details), 즉 물체의 경계와 미세한 디테일을 포착합니다.

구조적 특징:

얕은 층 (Shallow Layers): 레이어 수가 적습니다.

넓은 채널 (Wide Channels): 채널 용량이 풍부하여 저수준 정보를 충분히 담을 수 있습니다.

작은 스트라이드 (Small Stride): 다운샘플링을 적게 하여 입력 이미지와 유사한 고해상도 특징 맵을 유지합니다.

# 2. Semantic Branch (의미론적 브랜치)
목표: 고수준의 문맥적 정보(High-level Semantic Context), 즉 이미지의 전체적인 내용과 의미를 파악합니다.

구조적 특징:

깊은 층 (Deep Layers): 레이어 수가 많아 추상적인 특징을 추출합니다.

좁은 채널 (Narrow Channels): 채널 용량을 줄여 계산 비용을 크게 절감했습니다 (경량화).

빠른 다운샘플링 (Fast Down-sampling): 빠르게 특징 맵의 크기를 줄여(Receptive Field 확장) 연산 속도를 높이고 넓은 영역의 문맥 정보를 효율적으로 포착합니다.

2. 융합 메커니즘: Guided Aggregation Layer
두 브랜치는 서로 보완적인 정보를 추출하므로, 이를 효과적으로 결합하는 것이 중요합니다.

Guided Aggregation Layer (유도 집계 레이어): Detail Branch의 고해상도 특징과 Semantic Branch의 고수준 문맥 특징을 융합하는 역할을 합니다.

속도가 빠른 Semantic Branch의 특징 맵을 Detail Branch의 특징 맵 크기에 맞게 업샘플링합니다.

두 특징 맵을 곱셈(Element-wise Product)과 덧셈(Addition) 등의 연산을 통해 결합하여 최종적인 분할 예측을 위한 특징을 생성합니다.

3. 훈련 전략: Booster Training
BiSeNet V2는 추론 속도에는 영향을 주지 않으면서 학습 성능을 높이기 위해 **보조 훈련 전략(Booster Training Strategy)**을 사용합니다. Semantic Branch의 중간 단계 출력에 보조 손실(Auxiliary Loss)을 추가하여 Semantic Branch가 더 풍부한 의미론적 정보를 조기에 학습하도록 유도합니다.
