import pandas as pd
import streamlit as st
from pycaret.regression import load_model, predict_model

st.set_page_config(page_title="SuperMarket App")

@st.cache(allow_output_mutation=True)
def get_model():
  return load_model('sales')

def predict(model, df):
    predictions = predict_model(model, data = df)
    return predictions['Label'][0]

model = get_model()

st.title("Sales App")
st.markdown("Enter your personal data to get a prediction of your incomes")
st.markdown("https://www.linkedin.com/in/yesner-salgado/")


form = st.form("benefits")

Tv = form.slider('TV', min_value=0, max_value=200000, value=0)
Radio = form.slider('Radio', min_value=0, max_value=200000, value=0)
Social_Media = form.slider('Social Media', min_value=0, max_value=200000, value=0)

Influencer_list = ['Macro', 'Mega', 'Micro','Nano']
Influencer = form.selectbox('Influencer', Influencer_list)

predict_button = form.form_submit_button('Predict')

input_dict = {'TV' : Tv, 'Radio' : Radio, 'Social Media' : Social_Media,
'Influencer' : Influencer}

input_df = pd.DataFrame([input_dict])

if predict_button:
 out = predict(model, input_df)
 st.success('The incomes are ${:,.2f}'.format(out))
