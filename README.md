# multimic_enhancement

1. main.py: 메인 함수
2. trainer_DCE_multiCH.py: 학습 코드
3. model.py & model_complex.py: 모델 정의
   - model_complex.py > class LineartoMel_real(): 현재 사용중인 모델
4. data_loader.py: 데이터 HDD로부터 로딩 & minibatch 만들기
5. config.py: 프로그램에 지정가능한 옵션 모음
6. utils.py: 잡다한 함수 모음

실행 명령예
python3 main.py --gpu 0 --trainer DCE_multiCH --DB_name dereverb_4IR --multiCH True --BSE True --complexNN True --nCH 4 --batch_size 5 --lr 1e-4 --convW 1 --nMap_per_F 100 --L_CNN 2 --complex_BN True --nFFT 6400 --linear_to_mel True

1. phase difference 지금까지는 하드디스크에 저장해두고 불러왔었는데, 앞으로는 phase를 로딩한다음 코드 상에서 difference를 구하도록 수정해서 써야할 것 같아요.
2. 실행하시기 전에 data_sorted내에 있는 경로와 feature들을 맞추어야할 것이에요
