> ### ResNet (Residual Network)
>
> ResNet은 딥러닝 이미지 인식 분야에서 널리 사용되는 합성곱 신경망(CNN) 구조로,2015년 Microsoft Research에서 발표된 모델이다.
> 기존의 네트워크가 깊어질수록 학습이 어려워지는 문제를 해결하고, 더 깊은 네트워크에서도 안정적으로 학습이 가능하도록 설계되었다.

> ### 1. 등장 배경
>
>     딥러닝 모델의 성능은 일반적으로 층(layer)을 깊게 쌓을수록 향상되지만,실제로는 다음과 같은 문제가 발생한다.
>     기울기 소실(Vanishing Gradient): 역전파 시 앞단으로 갈수록 기울기가 0에 가까워져 학습이 멈춘다.
>     성능 저하(Degradation Problem): 층을 더 추가해도 훈련 오차가 오히려 증가한다.
>     ResNet은 이러한 문제를 해결하기 위해 "잔차 연결(Skip Connection)" 개념을 도입했다.

> ### 2. 핵심 아이디어
>
>     기존의 CNN은 입력을 변환하는 함수를 직접 학습한다.
> 
>                         𝑦=𝐹(𝑥)
> 
>     ResNet은 입력을 그대로 더하는 방식을 사용한다.
> 
>                         y=F(x)+x
> 
>     입력 x를 그대로 출력에 더해줌으로써,
>     모델이 F(x)=0만 학습하더라도 최소한의 정보 전달이 가능해진다.
>     이 단순한 연결로 인해 기울기 소실이 완화되고, 훨씬 깊은 네트워크 학습이 가능해졌다.


> ### 3. Resiual Block 구조
>     
>


```python
Basic Block(ResNet-18, 34)
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)



  입력 → 1×1 Conv (채널 축소)
       → 3×3 Conv (공간 특징 추출)
       → 1×1 Conv (채널 복원)
       → + 입력 → ReLU → 출력
```



> ### 4. 모델 구성
> <img width="795" height="262" alt="image" src="https://github.com/user-attachments/assets/c8c2e02c-c5fb-48fe-b92b-8c53dbc75bd6" />

> ### 5. 주요 특징
>     기울기 소실 완화: Skip Connection을 통해 역전파 정보 손실을 최소화
>
>     깊은 네트워크 학습: 100층 이상의 네트워크에서도 안정적인 학습 가능
>
>     빠른 수렴: 학습 속도 향상 및 최적화 안정성 개선
>
>     높은 확장성: Detection, Segmentation, Recognition 등 다양한 분야에 적용 가능
