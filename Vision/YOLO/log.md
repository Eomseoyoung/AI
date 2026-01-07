### YOLO 학습 성능 지표
<img width="682" height="289" alt="image" src="https://github.com/user-attachments/assets/93b72eb4-f483-43c6-af10-6d73396f7986" />

### 예시 코드(학습성능지표 추출)
        from ultralytics import YOLO
        import os
        
        # OpenMP 중복 초기화 에러 해결을 위한 설정
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        
        
        def my_callback(trainer):
        
            curr_epoch = trainer.epoch + 1
            total_epochs = trainer.epochs
        
            loss_items = trainer.loss_items
            
        
            metrics = trainer.metrics 
        
            map50 = metrics.get('metrics/mAP50(B)', 0) # mAP50
            map95 = metrics.get('metrics/mAP50-95(B)', 0) # mAP50-95
            precision = metrics.get('metrics/precision(B)', 0) # 정밀도
            recall = metrics.get('metrics/recall(B)', 0) # 재현율
        
        
        
        model = YOLO("yolov8n.pt")
        
        
        model.add_callback("on_train_epoch_end", my_callback)
        
        
        model.train(data="coco8.yaml", epochs=3, workers=0)
