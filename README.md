# water_level_prediction   
### 22.08.05 :   
lstm 모형에 데이터 학습시키기 위해서 데이터셋을 sliding window 형식의 numpy array로 변형하는 과정에서 메모리 이슈가 발생.   
30000개씩 쪼개서 변형한 뒤 concatenate 하려했지만 array size가 너무 커져서 실패했다. torch의 DataLoader를 이용하여 적용해보려고 한다.   
이 때 두가지 문제 발생.   
1. batch size는 적용이 되지만 window_size는 적용 불가능. dataset을 변경하는 class 생성하여 해결함.   
2. 지금 구현해놓은 모델은 tensorflow의 keras 모형. 모델을 변경해야 한다.   
   
### 22.08.06 :   
torch로 LSTM 모형 구현했는데 DataLoader로 학습시키는 방법을 모르겠다.   
일단 통째로 집어넣어서 학습시키는 방식으로 구현했다. 학습이 매우 오래 걸리고 성능이 너무 떨어지는 문제.   
일단 sliding window 적용해봐야 할 듯.   
   
### 22.08.07 :   
torch LSTM에 DataLoader로 iteration마다 window size에 맞게 잘라 가져와서 학습하는 방식으로 구현.   
그래도 데이터 사이즈가 커서 epoch 한 번 돌리는데도 굉장히 오래걸리고, 수렴이 안되는 문제 발생.   
epoch마다 learning_rate를 줄여가면서 학습하는 방식으로 돌려보는 중.   
효과가 조금 있는 것 같은데 학습 속도가 너무 느리다. 모형이랑 학습, 검증 부분 gpu 설정 한 뒤 workstation으로 돌려봐야 할 듯.
