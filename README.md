### water_level_prediction   
22.08.05 :   
lstm 모형에 데이터 학습시키기 위해서 데이터셋을 sliding window 형식의 numpy array로 변형하는 과정에서 메모리 이슈가 발생.   
30000개씩 쪼개서 변형한 뒤 concatenate 하려했지만 array size가 너무 커져서 실패했다.   
torch의 DataLoader를 이용하여 적용해보려고 한다.   
이 때 두가지 문제 발생.   
1. batch size는 적용이 되지만 window_size는 적용 불가능. dataset을 변경하는 class 생성하여 해결함.   
2. 지금 구현해놓은 모델은 tensorflow의 keras 모형. 모델을 변경해야 한다.   

