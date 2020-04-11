---
title: "Neural Style Transfer로 고흐풍 이미지로 변환"
date: 2020-04-11
header:
  teaser: /assets/images/neural-style-transfer/mangchi-output1.jpg
  og_image: /assets/images/page-header-teaser.png
categories:
  - project
tags:
  - Deep Learning
  - CNN
---

1. Neural Style Transfer
   * Info
   * Style Transfer
   * 환경 및 실습

---

### Info

* 원본

[Source code](https://github.com/anishathalye/neural-style)

* Data Files
[Pre-trained VCG network](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat)

* Dependencies:
  - Python
  - Tensorflow
  - numpy
  - scipy
  - pillow

---
### Style Transfer

Style Transfer, image-to-image translation, 또는 texture transfer 등으로 불리는 이 문제는 한 이미지 P를 다른 이미지 A의 스타일을 가지는 새로운 이미지 X를 생성하는 방식이다.

![구조](/assets/images/neural-style-transfer/algorithm.png)

[논문 설명 link](https://www.popit.kr/neural-style-transfer-%eb%94%b0%eb%9d%bc%ed%95%98%ea%b8%b0/)
{: .text-center}

---

### 환경 및 실습

* 환경

Colab 사용(Tensorflow GPU를 사용을 위해 - 맥북을 사용하다 보니 로컬에서 텐서플로 사용이 어려웠습니다.)

* 실습

깃허브는 한개의 파일이 100mb 초과하면 올려놓지 못합니다! pre-trained model 용량이 약 530mb인 관계로 코랩과 구글드라이브 연동 -> 구글드라이브에 git clone -> 구글 드라이브상에 model 업로드 -> 실행 순서로 진행했습니다.
{: .notice--info}

```yaml
from google.colab import drive
drive.mount('/gdrive')
```
*코랩과 구글 드라이브 연동*

```yaml
!pip install -r ../gdrive/My\ Drive/ns/requirements.txt
```
*정의해놓은 Dependencies 설치*

```yaml
!git clone https://github.com/gachonyws/neural-style.git ../gdrive/My\ Drive/ns/
```
*fork 해놓았던 소스를 제 드라이브에 clone 합니다. (원본소스에서 바로 clone 하지 않았습니다.)*

```yaml
Pre-trained 모델을 /gdrive/My\ Drive/ns/ 상에 직접 업로드.
```

```yaml
!python ../gdrive/My\ Drive/ns/neural_style.py --content ../gdrive/My\ Drive/ns/mang.jpg --styles ../gdrive/My\ Drive/ns/examples/1-style.jpg --output ../gdrive/My\ Drive/ns/mang_output.jpg --network ../gdrive/My\ Drive/ns/imagenet-vgg-verydeep-19.mat
```
*고양이 사진을 input 하여 보라색 계열의 스타일(1-style.jpg)적용하도록 설정하고 실행.*

![output1](/assets/images/neural-style-transfer/mangchi-output1.jpg)
*output1*

```
!python ../gdrive/My\ Drive/ns/neural_style.py --content ../gdrive/My\ Drive/ns/m.jpeg --styles ../gdrive/My\ Drive/ns/examples/1-style.jpg --output ../gdrive/My\ Drive/ns/m_output.jpeg --network ../gdrive/My\ Drive/ns/imagenet-vgg-verydeep-19.mat --preserve-colors
```
*--preserve-colors 옵션 추가로 Input image의 원래 색깔 보존*

![output2](/assets/images/neural-style-transfer/mangchi-output2.jpeg)
*output2*

```
Iteration  984/1000 (13 min 4 sec elapsed, 13 sec remaining)
Iteration  985/1000 (13 min 5 sec elapsed, 12 sec remaining)
Iteration  986/1000 (13 min 6 sec elapsed, 11 sec remaining)
Iteration  987/1000 (13 min 6 sec elapsed, 11 sec remaining)
Iteration  988/1000 (13 min 7 sec elapsed, 10 sec remaining)
Iteration  989/1000 (13 min 8 sec elapsed, 9 sec remaining)
Iteration  990/1000 (13 min 9 sec elapsed, 8 sec remaining)
Iteration  991/1000 (13 min 10 sec elapsed, 7 sec remaining)
Iteration  992/1000 (13 min 10 sec elapsed, 7 sec remaining)
Iteration  993/1000 (13 min 11 sec elapsed, 6 sec remaining)
Iteration  994/1000 (13 min 12 sec elapsed, 5 sec remaining)
Iteration  995/1000 (13 min 13 sec elapsed, 4 sec remaining)
Iteration  996/1000 (13 min 14 sec elapsed, 3 sec remaining)
Iteration  997/1000 (13 min 14 sec elapsed, 3 sec remaining)
Iteration  998/1000 (13 min 15 sec elapsed, 2 sec remaining)
Iteration  999/1000 (13 min 16 sec elapsed, 1 sec remaining)
Iteration 1000/1000 (13 min 17 sec elapsed, 0 sec remaining)
content loss: 779094
  style loss: 106161
     tv loss: 43878.2
  total loss: 929134
```
*제대로 실행되었다면 1,000번의 Iteration을 돌고 종료됩니다. (사진 파일의 크기가 너무 크다면 에러가 발생했었습니다. 리사이즈 후 진행해야 변환속도도 빠릅니다.)*

 
---

* 결과

| ![total](/assets/images/neural-style-transfer/mangchi-total.jpeg) |
|:--:|
| 샘플 이미지는 저랑 동거중인 고양이 사진으로 테스트 했습니다.🥰 |
