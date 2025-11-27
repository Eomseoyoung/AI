> # 🧠 YOLO (You Only Look Once) — 객체 탐지(Object Detection) 모델





> ### 📌 1. 개요

> YOLO는 “You Only Look Once”의 약자로, 한 번의 신경망 연산으로 **객체의 위치(Bounding Box)**와 **클래스(Class)**를 동시에 예측하는 실시간 객체 탐지 모델입니다.

> 🔍 기존의 탐지 모델(Faster R-CNN 등)은 “두 번” 이상 이미지를 살펴야 하지만, YOLO는 한 번(Once) 만에 예측하기 때문에 빠릅니다.




> ### 🧩 2. YOLO의 핵심 아이디어
> 기존 방식 (R-CNN 계열)	YOLO 방식
> 이미지에서 객체 후보(Region Proposal) → 분류(Classification)	이미지를 S×S 그리드로 분할하고, 각 셀에서 바로 객체의 위치·클래스 예측 느리지만 정확	빠르고 실시간 가능 
> ### 🖼️ 구조 개념도
> 
       입력 이미지
           ↓
      CNN 기반 Feature Extractor
           ↓
      S x S grid로 나눔
           ↓
      각 셀(Cell)마다:
        - B개의 Bounding Box 예측
        - 각 Box의 (x, y, w, h)
        - Confidence score (객체 존재 확률)
        - 클래스 확률 (Class probability)




> 최종적으로는


> (x, y, w, h, confidence, class_prob) 을 예측하여 객체의 위치와 종류를 동시에 파악합니다.


> ### ⚙️ 3. YOLO의 예측 구조

> YOLO는 이미지를 S×S 그리드로 나누고, 각 그리드 셀이 자신의 영역 안에 있는 객체를 예측합니다.

      예시:
      
      입력 이미지: 416×416
      
      S = 13 → 13×13 셀
      
      각 셀은 B개의 박스 예측 (보통 B=3~5)
      
      즉, 13×13×(B×(5+C)) 형태의 텐서를 출력합니다.
      
      여기서 5 = (x, y, w, h, confidence), C = 클래스 개수





> ### 🧮 4. YOLO의 출력값 의미
>  항목	설명
>  x, y	그리드 셀 기준 중심 좌표 (0~1 정규화)
>  w, h	이미지 전체에 대한 너비·높이 비율
>  confidence	박스 안에 객체가 존재할 확률
>  class_probs	객체가 각 클래스일 확률 분포 (Softmax or Sigmoid)

>  최종 객체 확률은 다음과 같이 계산합니다.

> P(class)×confidence




> ### 🧱 5. YOLO의 학습 방식

>  손실 함수(Loss)는 세 부분으로 구성됩니다.
>
>  좌표 손실 (Localization Loss)
>    → 예측된 (x, y, w, h)와 실제 박스 차이(MSE)
>
>  신뢰도 손실 (Confidence Loss)
>    → 객체가 존재하는지(1) / 아닌지(0)
>
>  클래스 손실 (Classification Loss)
>    → 예측된 클래스와 실제 클래스의 차이(Cross Entropy)



>  ### 🚀 6. YOLO의 장점과 단점

>  ✅ 장점	매우 빠른 실시간 탐지 가능
>  단일 CNN 구조 (End-to-End)
>  전체 이미지 문맥을 고려한 예측
>  ⚠️ 단점	작은 객체 탐지에 약함 (grid 해상도 한계)
> anchor 설정에 따라 성능 변동




> ### 🧬 7. YOLO의 버전별 발전 요약

>  YOLOv1 (2016)	최초 모델. 단일 CNN으로 객체 탐지 시도. 작은 객체 성능 낮음
>  YOLOv2 (YOLO9000)	Anchor box 도입, BatchNorm 적용, 더 깊은 네트워크
>  YOLOv3	Multi-scale 예측(13×13, 26×26, 52×52)로 작은 객체 성능 개선
>  YOLOv4	CSPDarknet backbone, Mosaic augmentation, SPP 추가
>  YOLOv5 (Ultralytics)	PyTorch 구현, 쉬운 학습/배포, 모델 크기별 버전(S, M, L, X)
>  YOLOv6 / YOLOv7	산업용 최적화, 더 빠르고 정확한 구조
>  YOLOv8 (2023)	Ultralytics 최신 버전, CNN + Transformer 결합, Segment/Track 지원
>  YOLO-NAS / RT-DETR 등	하이브리드 또는 차세대 구조 (attention 기반)

