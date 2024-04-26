# AI_Face-recognition_music
#인공지능 #얼굴 표정 인식 #얼굴 표정 기반 음악 재생 웹 개발 #Tensorflow & Keras

## :eyes: 프로젝트 상세 설명

* 안녕하세요, 제가 맡은 역할은 **데이터 수집, 얼굴 표정 인식 모델 개발, 웹 사이트에서 모델 동작시키기** 까지입니다.

![슬라이드1](https://github.com/yorsing/AI_Face-recognition_music/assets/48310109/69306058-b56f-46ea-bcda-f93f12b3cf0e)
![슬라이드2](https://github.com/yorsing/AI_Face-recognition_music/assets/48310109/6b3d2cbc-8968-483a-9e3a-714dc6e7c465)

* 이 프로젝트는 코로나19 팬데믹으로 인해 야외 활동이 제한되면서 느끼게 된 심리적 우울감과 스트레스를 해소하기 위해 시작하게 되었습니다.
* 이 시기에 음악 감상은 제 취미가 되었고, 이에 일일이 플레이리스트를 직접 고르지 않고 그날의 제 얼굴 표정에 나타난 감정에 맞춰 음악을 들을 수 있는 편리한 기능이 있다면 어떨까하는 생각으로 개발하게 되었습니다.

![슬라이드3](https://github.com/yorsing/AI_Face-recognition_music/assets/48310109/24cf43fd-2862-4165-8263-b0d1b8b0a63b)

* 개발 환경은 구글 코랩에서 Tensorflow와 Keras를 사용하여 GPU 학습을 진행했습니다.

![슬라이드4](https://github.com/yorsing/AI_Face-recognition_music/assets/48310109/6ac1b804-64b5-4e3f-b947-f17e615e1a3b)

* 딥러닝은 인공지능(AI)의 한 분야로서, 기계 학습(머신 러닝)의 특정 기법에 속합니다. 이 방법은 데이터로부터 복잡한 구조와 패턴을 학습하고 예측하는 데 초점을 맞춥니다. 특히, 딥러닝은 여러 계층을 가진 인공 신경망을 사용하여 고차원적인 데이터 문제를 해결합니다.

![슬라이드5](https://github.com/yorsing/AI_Face-recognition_music/assets/48310109/6435b06c-e659-4c5d-a6c6-917848297dbb)

* 인공 신경망은 사람의 뇌가 신경 세포(뉴런)로 정보를 처리하는 방식에서 영감을 받았습니다. 각 뉴런은 다른 뉴런으로부터 정보를 받아 처리하고, 그 결과를 다시 다른 뉴런에게 전달합니다. 딥러닝에서의 "딥(deep)"은 이러한 신경망의 계층이 여러 겹(깊게) 쌓여 있다는 의미입니다.
* 딥러닝 모델은 일반적으로 큰 데이터셋을 필요로 하며, 이 데이터에서 자동으로 특징을 추출하는 학습 과정을 거칩니다. 초기에는 단순한 패턴이나 특징을 인식하고, 점차 더 복잡하고 추상적인 특징을 계층적으로 학습하게 됩니다.
* 딥러닝의 대표적인 아키텍처로는 **이미지 인식**에 특화된 **합성곱 신경망(CNN)** 과 **시계열 데이터나 텍스트 처리** 에 특화된 **순환 신경망(RNN), 장단기 메모리(LSTM)** 등이 있습니다.

![슬라이드6](https://github.com/yorsing/AI_Face-recognition_music/assets/48310109/acd50eae-65e5-496f-815b-2ae7b6ce2890)

* 합성곱 신경망(CNN)은 합성곱 층과 풀링 층을 여러 겹 쌓아 올려 구성하며, 각 층을 거치면서 점점 더 고수준의 특징을 추출합니다.
* 초기 층은 가장자리, 색상, 질감 같은 기본적인 이미지 특성을 인식하고, 깊은 층으로 갈수록 객체의 부분이나 전체 구조 같은 복잡한 특성을 학습하게 됩니다.

![슬라이드7](https://github.com/yorsing/AI_Face-recognition_music/assets/48310109/19a23076-74ed-43fd-81d4-8b67d71d5da9)

* 합성곱층은 입력 이미지에 대해 N x N 크기의 필터를 통해 슬라이딩하여 각 위치에서 필터와 입력 데이터의 요소별 곱을 계산하고 결과를 합산합니다.
* 이 과정은 특징 맵(feature map)을 생성하며, 이미지의 중요한 특징을 추출하는 데 도움을 줍니다.

![슬라이드8](https://github.com/yorsing/AI_Face-recognition_music/assets/48310109/de73d1bb-027b-44b1-95ca-128c0d7de504)

* 그 다음은 합성곱 층에서 생성된 특징맵의 각 픽셀 값에 비선형 변환을 적용합니다. 일반적으로 ReLU(Rectified Linear Unit) 함수가 사용되며, 이는 음수 값을 0으로 설정하여 비선형성을 추가합니다.
* 풀링층(Pooling Layer)에서는 주로 최대 풀링(max pooling) 또는 평균 풀링(average pooling)을 사용하여 특징 맵의 크기를 줄입니다. 이 과정은 이미지의 공간 크기를 줄이면서 중요한 특징을 유지하고, 모델의 계산 부담을 감소시킵니다.

![슬라이드9](https://github.com/yorsing/AI_Face-recognition_music/assets/48310109/1b36bf74-568e-4977-92a5-16675207912e)

* 이러한 기본적인 cnn 구조를 바탕으로 ImageNet 경진대회에서 우승을 차지한 다양한 구조의 신경망이 해마다 등장하면서 2015년 이후엔 인간의 오차율 5%를 뛰어넘는 수준에 이르렀습니다.
* 저는 2014년에 준우승을 차지한 VGG Net를 사용하였는데요, VGG Net은 옥스포드 대학의 연구팀 Visual Geometry Group에 의해 개발된 모델입니다.

![슬라이드10](https://github.com/yorsing/AI_Face-recognition_music/assets/48310109/bb8d8a79-af5e-4d27-ac83-18ca0c19463f)

* VGG 네트워크는 층의 깊이에 따라 16레이어와 19레이어로 나뉘는데, 저는 모델의 전체 용량을 줄이기 위해 vgg16 레이어를 사용하였습니다.
* 3x3의 작은필터를 사용하여 합성곱 연산을 거치면 이미지의 사이즈가 빠르게 줄어드는 것을 막을 수 있어서 상대적으로 깊은 모델을 만들 수 있고, 연산할 때 발생하는 파라미터의 개수가 줄어들어 학습의 효율성이 뛰어나다는 장점을 가지고 있습니다.

![슬라이드11](https://github.com/yorsing/AI_Face-recognition_music/assets/48310109/54bfb464-56f9-459c-a33e-d55fe0c33513)

* 얼굴표정을 학습시키기 위해 총 4가지 감정을 가진 이미지들을 수집하였습니다. 화난표정, 웃는표정, 무표정, 우는표정으로 구성되어있고 train data는 1170장, test data 293장으로 구성되어 있습니다.

![슬라이드12](https://github.com/yorsing/AI_Face-recognition_music/assets/48310109/790c22d0-83d8-4f8f-a210-0a51f03a40d4)
![슬라이드13](https://github.com/yorsing/AI_Face-recognition_music/assets/48310109/b98e576d-a5e7-42b6-aa7b-76617ad79752)
![슬라이드14](https://github.com/yorsing/AI_Face-recognition_music/assets/48310109/29187ddb-5e57-44f9-8092-4daf787735e5)
![슬라이드15](https://github.com/yorsing/AI_Face-recognition_music/assets/48310109/82049e7d-7c41-40f3-90e9-e4737c362d2d)
![슬라이드16](https://github.com/yorsing/AI_Face-recognition_music/assets/48310109/07fe606c-efda-4aec-af9c-8a4fe911a9b8)
![슬라이드17](https://github.com/yorsing/AI_Face-recognition_music/assets/48310109/44f94c49-0ee3-4c26-adc9-359606d4c0d1)
![슬라이드18](https://github.com/yorsing/AI_Face-recognition_music/assets/48310109/178b3a9f-95be-410b-b689-0749fb818d5d)
![슬라이드19](https://github.com/yorsing/AI_Face-recognition_music/assets/48310109/6484d97e-6e5f-4a2f-adca-75348d2f8aff)
![슬라이드20](https://github.com/yorsing/AI_Face-recognition_music/assets/48310109/0a1615cb-10b5-4516-a27f-182074512500)
![슬라이드21](https://github.com/yorsing/AI_Face-recognition_music/assets/48310109/4932ac0e-df69-45b3-8c44-3ae6afa386f8)
![슬라이드22](https://github.com/yorsing/AI_Face-recognition_music/assets/48310109/4abda95d-c5e3-48c3-a381-fe6e7d976ea3)
![슬라이드23](https://github.com/yorsing/AI_Face-recognition_music/assets/48310109/d6dce308-a293-4424-84de-560946855878)
![슬라이드24](https://github.com/yorsing/AI_Face-recognition_music/assets/48310109/632a8eb3-8b2f-45d2-a67f-200d56ab9177)
![슬라이드25](https://github.com/yorsing/AI_Face-recognition_music/assets/48310109/eaa39b95-d93e-41d7-a210-1fba269b6299)
![슬라이드26](https://github.com/yorsing/AI_Face-recognition_music/assets/48310109/40dde62f-545d-4c3b-b443-86fb05e0be16)
![슬라이드27](https://github.com/yorsing/AI_Face-recognition_music/assets/48310109/2d52958d-380a-44ef-92ad-100422fca382)
![슬라이드28](https://github.com/yorsing/AI_Face-recognition_music/assets/48310109/2c86689f-21e3-430f-8937-01e1f638ab50)
![슬라이드29](https://github.com/yorsing/AI_Face-recognition_music/assets/48310109/bd28f130-4f64-4054-9146-70df892db7b4)
![슬라이드30](https://github.com/yorsing/AI_Face-recognition_music/assets/48310109/a32f338a-a9c1-4f20-b37c-7a96efdb6bd5)
![슬라이드31](https://github.com/yorsing/AI_Face-recognition_music/assets/48310109/f3b9aa54-c0df-44ee-9fa3-3d7ef274cce7)
