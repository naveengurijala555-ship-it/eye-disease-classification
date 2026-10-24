# Deep Learning-Based Multi-Class Eye Disease Classification using CNN

College mini project, Dept. of Artificial Intelligence & Data Science, Lakireddy Bali Reddy College of Engineering (Autonomous), Mylavaram.
Team: G. Naveen, G. Chakravarthy, S. Bhargav. Guide: K. Lakshmi Padmavathi.

Classifies eye images into four classes: Normal, Diabetic Retinopathy, Cataract, Glaucoma, using a fine-tuned MobileNetV2 CNN.

> Screening aid for a college project. Not a medical diagnosis.

## Results (fine-tuned MobileNetV2, 446 test images)

Test accuracy about 84.5%, macro F1 0.86. The main confusion is glaucoma vs normal.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Cataract | 1.00 | 1.00 | 1.00 |
| Diabetic retinopathy | 0.95 | 0.94 | 0.94 |
| Glaucoma | 0.80 | 0.71 | 0.76 |
| Normal | 0.69 | 0.80 | 0.74 |

## Limitations

- Glaucoma vs normal is often confused.
- Cataract images are outer-eye photos while the other classes are retina photos, so cataract is unusually easy for the model.
- Not tested on other cameras or datasets.

## Run locally

```bash
pip install -r requirements.txt streamlit
streamlit run app.py
```

Place `model.keras` in the same folder as `app.py`.
