import streamlit as st
import pickle
import pandas as pd
import os
import joblib
import numpy as np



def load_model(model_name, directory = 'saved_models'):
    filename = os.path.join(directory, f'{model_name}.pkl')
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

def load_label_encoders(directory='saved_models'):
    encoders = {}
    for item in os.listdir(directory):
        if item.endswith('_encoder.pkl'):
            col = item.replace('_encoder.pkl', '')
            encoders[col] = joblib.load(os.path.join(directory, item))
    return encoders

def preprocess_input(input_data, encoders):
    for col, encoder in encoders.items():
        if col in input_data:
            # Handling unseen labels: Assign a common category for unseen labels
            known_labels = set(encoder.classes_)
            # Apply lambda function to handle unseen labels
            input_data[col] = input_data[col].apply(lambda x: x if x in known_labels else 'Unseen')
            # Temporary solution: Fit the encoder again on the fly with 'Unseen' label
            # (Note: This is not an ideal solution for a production environment.
            #  A better approach would be adjusting the training phase to anticipate unseen categories.)
            all_labels = np.append(encoder.classes_, 'Unseen')
            encoder.fit(all_labels)
            # Transform the column with the adjusted encoder
            input_data[col] = encoder.transform(input_data[col])
    return input_data

# Streamlit user interface
def main():
    st.title("Penguin Species Prediction")

    # Input features based on the dataset
    island = st.selectbox('Island', options=['Biscoe', 'Dream', 'Torgersen'])
    bill_length_mm = st.slider('Bill Length (mm)', min_value=30.0, max_value=60.0, value=45.0, step=0.1)
    bill_depth_mm = st.slider('Bill Depth (mm)', min_value=13.0, max_value=21.0, value=17.0, step=0.1)
    flipper_length_mm = st.slider('Flipper Length (mm)', min_value=170, max_value=230, value=200)
    body_mass_g = st.slider('Body Mass (g)', min_value=2700, max_value=6300, value=4500)
    sex = st.selectbox('Sex', ['Male', 'Female'])

    # Button to make prediction
    if st.button('Predict Species'):
        # Create a DataFrame with the input features
        input_data = pd.DataFrame([[island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex]],
                                  columns=['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex'])

        st.write('Raw input data:', input_data)

        # Load encoders
        encoders = load_label_encoders()

        # Preprocess the input data
        input_data = preprocess_input(input_data, encoders)

        # Load the model (adjust the model name if needed)
        model = load_model('XGBoost')

        # Make prediction
        prediction = model.predict(input_data)

        # Convert numerical prediction back to species name (adjust according to your encoding)
        species_dict = {0: 'Adelie', 1: 'Gentoo', 2: 'Chinstrap'}
        predicted_species = species_dict.get(prediction[0], "Unknown")

        # Display the prediction
        st.write(f'Predicted Species: {predicted_species}')

# Run the app
if __name__ == '__main__':
    main()
