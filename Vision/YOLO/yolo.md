> # YOLO (You Only Look Once) — 객체 탐지(Object Detection) 모델
>
> ### 1. 개요
>
> YOLO는 You Only Look Once의 약자로, 한 번의 신경망 연산으로 __객체의 위치(Bounding Box)__ 와 __클래스(Class)__ 를 동시에 예측하는 실시간 객체 탐지 모델.
>  
> 기존의 탐지 모델(Faster R-CNN 등)은 “두 번” 이상 이미지를 살펴야 하지만, YOLO는 한 번(Once) 만에 예측하기 때문에 빠름.



> ### 2. 핵심
> 
> 기존 방식 (R-CNN 계열)과 다른 YOLO 방식
> 
> 이미지에서 객체 후보(Region Proposal) → 분류(Classification) 이미지를 S×S 그리드로 분할하고, 각 셀에서 바로 객체의 위치·클래스 예측 느리지만 정확, 빠르고 실시간 가능



> ### 3. 구조 개념도
> 
>       입력 이미지
>           ↓
>      CNN 기반 Feature Extractor
>           ↓
>      S x S grid로 나눔
>           ↓
>      각 셀(Cell)마다:
>        - B개의 Bounding Box 예측
>        - 각 Box의 (x, y, w, h)
>        - Confidence score (객체 존재 확률)
>        - 클래스 확률 (Class probability)
>
> 최종적으로는
>
> (x, y, w, h, confidence, class_prob) 을 예측하여 객체의 위치와 종류를 동시에 파악.


> ### 4. 구조
>
> YOLO는 이미지를 S×S 그리드로 나누고, 각 그리드 셀이 자신의 영역 안에 있는 객체를 예측합니다.
>
>     예시:
>      
>      입력 이미지: 416×416
>      
>      S = 13 → 13×13 셀
>      
>      각 셀은 B개의 박스 예측 (보통 B=3~5)
>      
>      즉, 13×13×(B×(5+C)) 형태의 텐서를 출력.
>      
>      여기서 5 = (x, y, w, h, confidence), C = 클래스 개수





> ### 5. YOLO의 출력값 의미
>
>|항목|설명|
>|---|---|
>|x, y|그리드 셀 기준 중심 좌표 (0~1 정규화)|
>|w, h|이미지 전체에 대한 너비·높이 비율|
>|confidence|박스 안에 객체가 존재할 확률|
>|class_probs|객체가 각 클래스일 확률 분포 (Softmax or Sigmoid)|
>|최종 객체 확률|P(class)×confidence|
>
>
>|항목|설명|
>|---|---|
>|Image Count|검증(validation) 데이터에 포함된 이미지 개수|
>|Instance Count|검증 데이터에 포함된 객체(annotation) 총 개수|
>|Precision (정밀도)|모델이 “객체라고 예측한 것” 중에서 실제로 맞춘 비율|
>|Recall (재현율)|실제로 존재하는 객체 중 모델이 맞춘 비율|
>|mAP (mean Average Precision)|정확도(Precision)와 재현율(Recall)의 전체 곡선(PR Curve)을 종합해 모델의 진짜 성능을 나타내는 값|

> ### 6. 학습 방식
>
>  손실 함수(Loss)는 세 부분으로 구성
>
>  * 좌표 손실 (Localization Loss)
>    → 예측된 (x, y, w, h)와 실제 박스 차이(MSE)
>
>  * 신뢰도 손실 (Confidence Loss)
>    → 객체가 존재하는지(1) / 아닌지(0)
>
>  * 클래스 손실 (Classification Loss)
>    → 예측된 클래스와 실제 클래스의 차이(Cross Entropy)



>  ### 7. 장점과 단점
>
>|장점|단점|
>|---|---|
>|매우 빠른 실시간 탐지 가능|작은 객체 탐지에 약함 (grid 해상도 한계)|
>|단일 CNN 구조 (End-to-End)|anchor 설정에 따라 성능 변동|
>|전체 이미지 문맥을 고려한 예측||




> ### 8. 성능평가
>
>|항목|설명|
>|---|---|
>|map50 < min_map50(0.65)|전반적 검출력 부족|
>|map5095 < min_map5095(0.40)|박스 정밀도 낮음|
>|precision < min_precision(0.70)|오탐이 많음|
>|recall < min_recall(0.65)|놓치는 객체가 많음|


> ### 9. 판단 기준표
>|상황|원인|
>|---|---|
>|train 성능도 안 나옴|모델 or 학습 설정|
>|train은 좋은데 val만 나쁨|데이터|
>|클래스별 성능 편차 큼|데이터|
>|특정 상황에서만 실패|데이터|
>|전반적으로 다 못 잡음|모델 or 해상도|



> ### 10. 모델 사이즈별 구조 & 성능 비교
>
> 
> <img width="929" height="314" alt="image" src="https://github.com/user-attachments/assets/57104f3a-fe4d-4435-a90b-67bd1f50cdb3" />
> mAP(mean Average Precision)은 COCO 데이터셋 기준, FPS는 640×640 입력 기준 추정치




> ### 11. 버전별 발전 요약
>
>  
>|버전|설명|
>|---|---|
>|YOLOv1 (2016)|최초 모델. 단일 CNN으로 객체 탐지 시도. 작은 객체 성능 낮음|
>|YOLOv2 (YOLO9000)|Anchor box 도입, BatchNorm 적용, 더 깊은 네트워크|
>|YOLOv3|Multi-scale 예측(13×13, 26×26, 52×52)로 작은 객체 성능 개선|
>|YOLOv4|CSPDarknet backbone, Mosaic augmentation, SPP 추가|
>|YOLOv5 (Ultralytics)|PyTorch 구현, 쉬운 학습/배포, 모델 크기별 버전(S, M, L, X)|
>|YOLOv6 / YOLOv7|산업용 최적화, 더 빠르고 정확한 구조|
>|YOLOv8 (2023)|Ultralytics 최신 버전, CNN + Transformer 결합, Segment/Track 지원|
>|YOLO-NAS / RT-DETR 등|하이브리드 또는 차세대 구조 (attention 기반)|
>|YOLO26|엔드 투 엔드 NMS-free 추론을 통해 에지 배포에 최적화된 Ultralytics의 차세대 YOLO 모델|


