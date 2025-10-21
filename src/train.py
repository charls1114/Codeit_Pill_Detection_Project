from tqdm import tqdm
import torch

def train_ssd_rcnn_model(
        model, train_loader, optim, weight_path=None, 
        model_type='SSD',
        num_epochs=10,
        schedule_step=5, 
        gamma=0.1):
    
    # 최적의 모델 저장을 위한 변수 초기화
    best_loss = float('inf')

    # 사전 학습된 가중치가 제공된 경우 모델에 로드
    if weight_path:
        model = model.load_state_dict(torch.load(weight_path))
    # 장치 설정 (CPU 또는 GPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # 학습률 스케줄러 설정
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=schedule_step, gamma=gamma)
    
    # 학습 루프 시작
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_bar = tqdm(train_loader, desc=f' {epoch+1}/{num_epochs} Epoch Training...')  
        for images, targets in train_bar:
            images = list(image.to(device) for image in images)
            targets = list({k: v.to(device) for k, v in t.items()} for t in targets)

        # 모델에 입력하여 SSD 또는 RCNN 모델의 손실(loss) 계산
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optim.zero_grad()       # 기울기 초기화
        losses.backward()       # 역전파 수행
        optim.step()            # 파라미터 업데이트

        total_loss += losses.item()
        train_bar.set_postfix(loss=f"{losses.item():.3f}")
        

        lr_scheduler.step()  # 에폭이 끝난 후 학습률 업데이트
        avg_train_loss = total_loss / len(train_loader) # 평균 훈련 손실 계산
        
        # 최적의 모델 저장
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            torch.save(model.state_dict(), f'{model_type}_best_model.pth')
        print(f"Epoch: {epoch+1:2d}, Avg Train Loss: {avg_train_loss:.3f}")
    
    # 마지막 에폭의 모델 저장
    torch.save(model.state_dict(), f'{model_type}_last_model.pth')
    
    # 사용하지 않는 GPU 메모리 해제 (메모리 최적화)
    torch.cuda.empty_cache()




