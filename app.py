import streamlit as st
import xgboost as xgb
import pandas as pd

title = "ICU-PH-Mortality-Risk"

st.set_page_config(    
    page_title=f"{title}",
    page_icon="logo.jpg",
    layout="wide"
)

df = pd.read_csv("origin_data.csv")

BOOL = {"No":1, "Yes":0}
RACE = {"Unknown":0, "White":1, "Black":2, "Asian":3, "Hispanic":4}

st.markdown(f'''
    <h1 style="font-size: 36px; text-align: center; color: black; background: #008BFB; border-radius: .5rem; margin-bottom: 1rem; color: white;">
    {title}
    </h1>''', unsafe_allow_html=True)
    
# 连续特征
features1 = [
    'hosp_days', 'admission_age', 'sbp', 'dbp', 'resp_rate', 'temperature', 'wbc',
    'lymphocytes', 'eosinophils', 'rbc', 'rdw', 'platelet', 'so2', 'po2', 'pco2',
    'baseexcess', 'lactate', 'bun', 'creatinine', 'aniongap', 'glucose', 'sodium',
    'potassium', 'albumin', 'alp', 'ast', 'bilirubin_total', 'fibrinogen', 'ptt',
    'gcs', 'height', 'weight', 'paps', 'papd', 'papm', 'cvp']
    
# 分类特征
features2 = [
    'race', 'nitrates', 'anticoagulants', 'Norepinephrine', 'Dopamine', 
    'Epinephrine', 'Phenylephrine', 'Vasopressin', 'crrt',
    'congestive_heart_failure', 'cerebrovascular_disease',
    'chronic_pulmonary_disease', 'liver_disease', 'renal_disease'
]

data = {}

col = st.columns(6)
for k, i in enumerate(features1+features2):
    if k<36:
        data[i] = col[k%6].number_input(i, value=df[i].tolist()[1])
    else:
        if i=="race":
            data[i] = RACE[col[k%6].selectbox(i, RACE, index=list(RACE.values()).index(df[i].tolist()[1]))]
        else:
            data[i] = BOOL[col[k%6].selectbox(i, BOOL, index=list(RACE.values()).index(df[i].tolist()[1]))]
    

# 创建一个新的模型实例  
model = xgb.XGBClassifier()  

# 从JSON文件加载模型  
model.load_model('xgboost_model.json')  

# 现在可以使用 loaded_model 进行预测  
res = model.predict(pd.DataFrame([data]))
res_proba = model.predict_proba(pd.DataFrame([data]))
if res_proba[0][-1]<=0.3:
    res_proba = res_proba[0][-1]
    color = "green"
    res = "Low risk"
elif 0.3<=res_proba[0][-1]<=0.6:
    res_proba = res_proba[0][-1]
    color = "orange"
    res = "Medium risk"
else:
    res_proba = res_proba[0][-1]
    color = "red"
    res = "High risk"
    
st.markdown(f'''
    <div style="font-size: 28px; text-align: center; color: black; background: transparent; border-radius: .5rem; border: 1px solid red; padding: 1rem; font-weight: bold;">
    Predict result: <span style="color:{color}; font-weight: bold;">{res}.</span>
    </div>''', unsafe_allow_html=True) # , probability: {round(res_proba*100, 3)}%.