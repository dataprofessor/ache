import streamlit as st
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


st.markdown('''
# 💊
# AChEpred: Prediction of Acetylcholinesterase inhibitors and non-inhibitors')
''')

# Load dataset
st.header('# Load dataset')
dataset_url = 'https://raw.githubusercontent.com/dataprofessor/data/master/acetylcholinesterase_07_bioactivity_data_2class_pIC50_pubchem_fp.csv'
dataset = pd.read_csv(dataset_url)
st.write(dataset)

# Data pre-processing
st.header('Data pre-processing')
          
# Prepare class label column
st.subheader('# Prepare class label column')
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
st.write(df)

# Select X and Y variables
st.subheader('# Select X and Y variables')

X = df.drop(['pIC50', 'class'], axis=1)

def target_encode(val):
  target_mapper = {'inactive':0, 'active':1}
  return target_mapper[val]

Y = df['class'].apply(target_encode)

st.write(X)
st.write(Y)

# Remove low variance features
st.subheader('# Remove low variance features')

def remove_low_variance(input_data, threshold=0.1):
    selection = VarianceThreshold(threshold)
    selection.fit(input_data)
    return input_data[input_data.columns[selection.get_support(indices=True)]]

X = remove_low_variance(X, threshold=0.1)


