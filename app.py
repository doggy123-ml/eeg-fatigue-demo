import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 載入資料
train = pd.read_csv("EEG_LDA_train.csv")
test = pd.read_csv("EEG_LDA_test.csv")

# 手刻 Naive Bayes 前置計算
X_train = train[["LDA_1"]].values.flatten()
y_train = train["eyeDetection"].values

class_0 = X_train[y_train == 0]
class_1 = X_train[y_train == 1]
mean_0, std_0 = np.mean(class_0), np.std(class_0)
mean_1, std_1 = np.mean(class_1), np.std(class_1)
prior_0 = len(class_0) / len(X_train)
prior_1 = len(class_1) / len(X_train)

def gaussian_pdf(x, mean, std):
    return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(- ((x - mean) ** 2) / (2 * std ** 2))

# 頁面設定
st.set_page_config(page_title="🧠 EEG Fatigue Prediction System", layout="centered")
st.title("🧠 EEG Fatigue Prediction System")
st.markdown("""
<style>
    .result-box {
        border-radius: 10px;
        padding: 15px;
        font-size: 18px;
        font-weight: bold;
        color: white;
        text-align: center;
    }
    .awake {
        background-color: #4CAF50;
    }
    .fatigued {
        background-color: #f44336;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("---")
st.subheader("📥 單筆預測輸入")
st.write("請輸入一個 LDA 降維後的特徵值，系統將預測是否處於疲勞狀態")
input_val = st.number_input("輸入 LDA_1 數值：", value=0.0, format="%.4f")

if st.button("🔍 進行預測"):
    likelihood_0 = gaussian_pdf(input_val, mean_0, std_0) * prior_0
    likelihood_1 = gaussian_pdf(input_val, mean_1, std_1) * prior_1
    total = likelihood_0 + likelihood_1
    prob_0 = likelihood_0 / total
    prob_1 = likelihood_1 / total
    pred = 0 if prob_0 >= prob_1 else 1

    status_label = "🟢 清醒" if pred == 0 else "🔴 疲勞"
    box_class = "awake" if pred == 0 else "fatigued"

    st.markdown(f"""
    <div class="result-box {box_class}">
        預測結果：{status_label}<br>
        清醒機率：{prob_0*100:.2f}% ｜ 疲勞機率：{prob_1*100:.2f}%
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 LDA Feature Distribution")
    fig, ax = plt.subplots()
    ax.hist(class_0, bins=30, alpha=0.5, label='Awake', color='green')
    ax.hist(class_1, bins=30, alpha=0.5, label='Fatigued', color='red')
    ax.axvline(input_val, color='blue', linestyle='--', label='Input')
    ax.set_xlabel("LDA_1 Value")
    ax.set_ylabel("Sample Count")
    ax.set_title("Distribution of LDA Feature Values")
    ax.legend()
    st.pyplot(fig)

st.markdown("---")
with st.expander("📊 Batch Test Evaluation (Developer Section)"):
    st.write("This section shows model prediction performance on test dataset:")
    X_test = test[["LDA_1"]].values.flatten()
    y_test = test["eyeDetection"].values
    y_pred = []
    for x in X_test:
        l0 = gaussian_pdf(x, mean_0, std_0) * prior_0
        l1 = gaussian_pdf(x, mean_1, std_1) * prior_1
        pred = 0 if l0 >= l1 else 1
        y_pred.append(pred)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.success(f"Model Accuracy: {acc*100:.2f}%")
    st.write("Confusion Matrix:")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred Awake', 'Pred Fatigued'], yticklabels=['True Awake', 'True Fatigued'])
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

st.markdown("---")
st.caption("Model: Manually implemented Gaussian Naive Bayes | Data: EEG LDA-reduced features")