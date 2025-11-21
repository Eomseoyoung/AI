> ## PIDNet (Pixel-wise Interacting Dual Branch Network)
> 
> 실시간 Semantic Segmentation을 위해 설계된 네트워크로,BiSeNet의 아이디어를 발전시켜 속도와 정확도 사이의 균형을 극대화한 모델
> 주요 목표
> Real-time 속도 (60+ FPS)
> High Accuracy (Cityscapes mIoU 80% 이상)
> Edge(경계) 인식 강화
>



      입력 → Shared Stem
           ├── P-Branch (Proportional)
           ├── I-Branch (Integral)
           └── D-Branch (Derivative)
                 ↓
              Fusion
                 ↓
              Final Segmentation Map
>
> 
> <img width="653" height="285" alt="image" src="https://github.com/user-attachments/assets/16527238-eba1-47cc-b479-6922539474ba" />


> ## 네트워크 상세 구조
> 공유 Stem: 입력 이미지를 초기 특징 맵으로 변환 (Conv, BN, ReLU)

> P-Branch: 경량 백본 (예: ShuffleNet, MobileNet 계열) 기반 빠른 특징 추출

> I-Branch: dilated convolution 등을 활용하여 넓은 수용영역 확보

> D-Branch: 경계 정보를 학습하기 위한 shallow-layer 기반 경량 sub-network
