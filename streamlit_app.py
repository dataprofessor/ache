import streamlit as st
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.markdown('# 💊 AChEpred')
st.info('Prediction of Acetylcholinesterase inhibitors and non-inhibitors')

# Load dataset
st.markdown('## 1. Load dataset')
st.info('''
A dataset consisting of Acetylcholinesterase bioactivity data was compiled from the ChEMBL database.

Each compounds were labeled as inhibitors (pIC50 ≥ 6) or non-inhibitors (pIC50 ≤ 5) on the basis of their bioactivity data values.
''')

dataset_url = 'https://raw.githubusercontent.com/dataprofessor/data/master/acetylcholinesterase_07_bioactivity_data_2class_pIC50_pubchem_fp.csv'
dataset = pd.read_csv(dataset_url)

with st.expander('See dataset'):
  st.write(dataset)

# Data pre-processing
st.markdown('## 2. Data pre-processing')
          
# Prepare class label column
st.markdown('#### Prepare class label column')
bioactivity_threshold = []
for i in dataset.pIC50:
  if float(i) <= 5:
    bioactivity_threshold.append("inactive")
  elif float(i) >= 6:
    bioactivity_threshold.append("active")
  else:
    bioactivity_threshold.append("intermediate")
    
# Add class label column to the dataset DataFrame
bioactivity_class = pd.Series(bioactivity_threshold, name='class')
df = pd.concat([dataset, bioactivity_class], axis=1)

with st.expander('See dataset (with class label column)'):
  st.write(df)

# Select X and Y variables
st.markdown('#### Select X and Y variables')

X = df.drop(['pIC50', 'class'], axis=1)

def target_encode(val):
  target_mapper = {'inactive':0, 'active':1}
  return target_mapper[val]

Y = df['class'].apply(target_encode)

with st.expander('See X variables'):
  st.write(X)

with st.expander('See Y variable'):
  st.write(Y)

# Remove low variance features
st.markdown('#### Remove low variance features')

def remove_low_variance(input_data, threshold=0.1):
    selection = VarianceThreshold(threshold)
    selection.fit(input_data)
    return input_data[input_data.columns[selection.get_support(indices=True)]]

X = remove_low_variance(X, threshold=0.1)

with st.expander('See X variables (low variance features removed)'):
  st.write(X)

# Random Forest Classification Model
st.markdown('## 3. Random Forest Classification Model')

# Data splitting
st.markdown('#### Data splitting')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

with st.expander('See X_train, y_train dimensions'):
  st.write(X_train.shape, y_train.shape)
with st.expander('See X_train, y_train dimensions'):
  st.write(X_test.shape, y_test.shape)
  
# Model Building
st.markdown('#### Model Building')

model = RandomForestClassifier(n_estimators=500, random_state=42)
with st.spinner('Model is building...'):
  model.fit(X_train, y_train)


