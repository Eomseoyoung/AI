> # Pruning
> ------------------------------------------------------------------------
> ### 개념
>  * 중요도가 낮은 weight/channel/layer 제거
>  * 덜 중요한 연결을 끊는다고 생각
>
> ### 대표 방식
> #### 1. Unstruchured Pruning
>   * weight 단위 제거
>   * 장점 : 이론적 압축률 높음
>   * 단점 : 실제 속도 개선 거의 없음(GPU)
>
> #### 2. Structured Pruning(실무 중요)
>   * channel/filter/layer 단위 제거
>   * 장점 : 실제 FPS 개선, HW 친화적
>   * 단점 : 성능 하락 위험, 구조 변경 필요
>
> ### Pruning 전략
>   * Train -> Prune -> Find-tune -> Repeat
