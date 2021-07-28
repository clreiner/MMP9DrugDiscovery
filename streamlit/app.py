import streamlit as st
import pandas as pd 
import subprocess
import os
import base64
import pickle

# Molecular descriptor calculator
def padel_desc():
    # Uses PaDEL-Descriptor to find PubChem fingerprints for input data
    bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes ./PaDEL-Descriptor/PubchemFingerprinter.xml -dir ./ -file descriptors_output.csv"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    os.remove('molecule.smi')

# Download data input file
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href

def make_pred(input_data):
    # Load model
    model = pickle.load(open('stacked_model.sav', 'rb'))
    # Use model to make prediction
    prediction = model.predict(input_data)
    st.header('Your Predictions...')
    prediction_output = pd.Series(prediction, name='Active')
    molecule_name = pd.Series(load_data[1], name='molecule_name')
    df = pd.concat([molecule_name, prediction_output], axis=1)
    st.write(df)
    st.markdown(filedownload(df), unsafe_allow_html=True)

st.markdown('# Matrix Metalloproteinase 9 Bioactivity Prediction App')


# Create Sidebar for data input
with st.sidebar.header('1. Upload your CSV data'):
    upload = st.sidebar.file_uploader('Upload your input file')
    st.sidebar.markdown('[Example input file](https://raw.githubusercontent.com/dataprofessor/bioactivity-prediction-app/main/example_acetylcholinesterase.txt)')

if st.sidebar.button('Predict'):
    load_data = pd.read_csv(upload, header=None)
    
    st.header('Original input data')
    st.write(load_data)

    with st.spinner('Calculating PubChem fingerprints'):
        padel_desc

    st.header('Calculated PubChem fingerprints')
    desc = pd.read_csv('descriptors_output.csv')
    st.write(desc)
    st.write(desc.shape)

    st.header('Subset of fingerprints from previously built models')
    X_list = list(pd.read_csv('../datasets/X_descriptors.csv').columns)
    desc_subset = desc[X_list]
    st.write(desc_subset)
    st.write(desc_subset.shape)

    make_pred(desc_subset)

else: 
    st.info('Upload input data in sidebar to start!')