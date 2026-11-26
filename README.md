# Heart-Brain Axis Wearable AI (Cloud Inference Server)

Cloud deployment bundle for the **ESP32-S3 Heart-Brain Axis Wearable** (`CardiacModel1A`, `StrokeModel1B`, `UnifiedHeartBrainModel1`, PyTorch + XGBoost).

## 1-Click Free Cloud Launch (Google Colab — 12 GB RAM, No Credit Card)
Open directly in Google Colab:
https://colab.research.google.com/github/George-Web-Developer/heart-brain-ai-cloud/blob/main/Heart_Brain_Cloud_Server.ipynb

## Active Models Included
- `CardiacModel1A` (`cardiac_model_1a.pkl` + `cardiac_model_1a_cnn.pt`)
- `StrokeModel1B` (`stroke_model_1b.pkl` + `stroke_model_1b_cnn.pt`)
- `UnifiedHeartBrainModel1` (`unified_model_1.pkl` + `unified_model_1_cnn.pt`)
- `GoldenHourTimerEngine` (`golden_hour_timer_engine.npz`)

## Endpoints
- `GET /health`
- `POST /api/v1/esp32-predict` (`?compact=true` for ESP32-S3)
- `POST /api/v1/cardiac-predict`
- `POST /api/v1/stroke-predict`
