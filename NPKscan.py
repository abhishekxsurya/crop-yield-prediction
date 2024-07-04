#final code
import os
import tempfile
import pandas as pd
import warnings
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
import cv2
import pytesseract
import re
import streamlit as st
from streamlit_option_menu import option_menu
# Set page config
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon=":seedling:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set theme
# See more themes at: https://www.streamlit.io/gallery
COLOR_THEME = {
    "primaryColor": "#7fec8e",
    "backgroundColor": "#212735",
    "secondaryBackgroundColor": "#4f5054",
    "textColor": "#95f1ae"
}

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: {COLOR_THEME['backgroundColor']};
        color: {COLOR_THEME['textColor']};
    }}
    .sidebar .sidebar-content {{
        background: {COLOR_THEME['secondaryBackgroundColor']};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

def home():
    css = """
    body {
        background-color: #000000;
    }
    """

    st.markdown("<h1>Welcome to the Crop Yield Predictor</h1>",
                unsafe_allow_html=True)
    # st.write("This web app allows you to predict the health status of your soil based on the values of Nitrogen, Phosphorous and Potassium.")
    #
    # st.write("Click on Menu to navigate to predictor")
    st.markdown(
        "<p>The aim is to build crop yield prediction model for a given historical data on NPK values and other environmental factors that affect crop yield, the goal is to develop a model that can accurately predict crop yield based on NPK values. The model should able to take into account the interactions between NPK values and provide accurate predictions for different regions and crop types.</p>",
        unsafe_allow_html=True)

    st.markdown(
        "<p>The site will take npk values as input. The user can enter these values manually or extract these values from the image uploaded. The model calculates yields based on npk values and whichever crop produces the highest yield is suggested to the user with the estimated yield in tons per hectare. The target user of the site our predominately Farmers, Agricultural Research Centers, etc.</p>",
        unsafe_allow_html=True)

# Define the layout of the upload page
# Define the layout of the upload page
def upload():
    st.title("Crop Yield Predictor")
    st.write("Please select an option to input soil data:")

    option = st.radio("Input Method", ("Upload Soil Health Card Image", "Manual Input"))

    if option == "Upload Soil Health Card Image":
        st.write("Please upload an image of your soil health card")
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Call the function that predicts the soil health card values from the uploaded image
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

            # Load image, grayscale, Otsu's threshold
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file.flush()
                image = cv2.imread(temp_file.name)
            os.unlink(temp_file.name)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # Repair horizontal table lines
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

            # Remove horizontal lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (55, 2))
            detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                cv2.drawContours(image, [c], -1, (255, 255, 255), 9)

            # Remove vertical lines
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 55))
            detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                cv2.drawContours(image, [c], -1, (255, 255, 255), 9)

            # Perform OCR
            data = pytesseract.image_to_string(image, lang='eng', config='--psm 6')

            # Extract values of nitrogen, phosphorus, and potassium
            lines = data.split('\n')
            n, p, k = 0, 0, 0
            for line in lines:
                if line.startswith('4 Available Nitrogen (N) '):
                    n = line.split()[4]
                elif line.startswith('5 Available Phosphorus (P) '):
                    p = line.split()[4]
                elif line.startswith('6 Available Potassium (K) '):
                    k = line.split()[4]
            if n==0 and p==0 and k==0:
                st.warning("Unable to extract values from Image please enter manually")
                st.stop()

            cv2.waitKey()
            n = float(n)
            p = float(p)
            k = float(k)

            # redgram-random forest
            py = {}
            data_rg = pd.read_csv('C:/Users/hbadh/Desktop/Mini Project/Datasets/Crop-wise/Original/Redgram.csv')
            min_max_data_rg = pd.read_csv('C:/Users/hbadh/Desktop/Mini Project/Datasets/Crop-wise/Normalized/Redgram_N.csv')
            min_val_n = min_max_data_rg['N'].min()
            warnings.filterwarnings('ignore', category=UserWarning)
            max_val_n = min_max_data_rg['N'].max()
            min_val_p = min_max_data_rg['P'].min()
            max_val_p = min_max_data_rg['P'].max()
            min_val_k = min_max_data_rg['K'].min()
            max_val_k = min_max_data_rg['K'].max()
            new_n = (n - min_val_n) / (max_val_n - min_val_n)
            new_p = (p - min_val_p) / (max_val_p - min_val_p)
            new_k = (k - min_val_k) / (max_val_k - min_val_k)
            new_data = [[new_n, new_p, new_k]]
            X = data_rg[['N', 'P', 'K']]
            y = data_rg['Yield']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
            model = RandomForestRegressor(n_estimators=100, random_state=2)
            model.fit(X_train, y_train)
            predicted_yield_rg = model.predict(new_data)
            py['redgram'] = predicted_yield_rg

            # rice-svr
            data_r = pd.read_csv('C:/Users/hbadh/Desktop/Mini Project/Datasets/Crop-wise/Original/Rice.csv')
            min_max_data_r = pd.read_csv('C:/Users/hbadh/Desktop/Mini Project/Datasets/Crop-wise/Normalized/Rice_N.csv')
            min_val_n = min_max_data_r['N'].min()
            max_val_n = min_max_data_r['N'].max()
            min_val_p = min_max_data_r['P'].min()
            max_val_p = min_max_data_r['P'].max()
            min_val_k = min_max_data_r['K'].min()
            max_val_k = min_max_data_r['K'].max()
            new_n = (n - min_val_n) / (max_val_n - min_val_n)
            new_p = (p - min_val_p) / (max_val_p - min_val_p)
            new_k = (k - min_val_k) / (max_val_k - min_val_k)
            new_data = [[new_n, new_p, new_k]]
            X_r = data_r[['N', 'P', 'K']]
            y_r = data_r['Yield']
            X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y, test_size=0.2, random_state=2)
            model = SVR(kernel='linear', C=1.0, epsilon=0.2)
            model.fit(X_train_r, y_train_r)
            predicted_yield_r = model.predict(new_data)
            py['rice'] = predicted_yield_r

            # maize-gradient boosting regressor
            data_m = pd.read_csv('C:/Users/hbadh/Desktop/Mini Project/Datasets/Crop-wise/Original/Maize.csv')
            min_max_data_m = pd.read_csv('C:/Users/hbadh/Desktop/Mini Project/Datasets/Crop-wise/Normalized/Maize_N.csv')
            min_val_n = min_max_data_m['N'].min()
            max_val_n = min_max_data_m['N'].max()
            min_val_p = min_max_data_m['P'].min()
            max_val_p = min_max_data_m['P'].max()
            min_val_k = min_max_data_m['K'].min()
            max_val_k = min_max_data_m['K'].max()
            new_n = (n - min_val_n) / (max_val_n - min_val_n)
            new_p = (p - min_val_p) / (max_val_p - min_val_p)
            new_k = (k - min_val_k) / (max_val_k - min_val_k)
            new_data = [[new_n, new_p, new_k]]
            X = data_m[['N', 'P', 'K']]
            y = data_m['Yield']
            X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X, y, test_size=0.3, random_state=3)
            model = GradientBoostingRegressor(random_state=2)
            model.fit(X_train_m, y_train_m)
            predicted_yield_m = model.predict(new_data)
            py['maize'] = predicted_yield_m

            # groundnut svr
            data_g = pd.read_csv('C:/Users/hbadh/Desktop/Mini Project/Datasets/Crop-wise/Original/Groundnut_cleaned.csv')
            min_max_data_g = pd.read_csv('C:/Users/hbadh/Desktop/Mini Project/Datasets/Crop-wise/Normalized/Groundnut_cleaned_N.csv')
            min_val_n = min_max_data_g['N'].min()
            max_val_n = min_max_data_g['N'].max()
            min_val_p = min_max_data_g['P'].min()
            max_val_p = min_max_data_g['P'].max()
            min_val_k = min_max_data_g['K'].min()
            max_val_k = min_max_data_g['K'].max()
            new_n = (n - min_val_n) / (max_val_n - min_val_n)
            new_p = (p - min_val_p) / (max_val_p - min_val_p)
            new_k = (k - min_val_k) / (max_val_k - min_val_k)
            new_data = [[new_n, new_p, new_k]]
            X = data_g[['N', 'P', 'K']]
            y = data_g['Yield']
            X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X, y, test_size=0.3, random_state=2)
            model = HistGradientBoostingRegressor(random_state=2)
            model.fit(X_train_g, y_train_g)
            predicted_yield_g = model.predict(new_data)
            py['groundnut'] = predicted_yield_g
            sorted_crop_yield = {k: v for k, v in sorted(py.items(), key=lambda item: item[1], reverse=True)}

            cy = next(iter(sorted_crop_yield.items()))
            st.write(cy[0], cy[1][0])

    elif option == "Manual Input":
        st.write("Please enter the values of Nitrogen, Phosphorous, and Potassium below.")
        n = st.number_input("Enter Nitrogen value", min_value=0.0, max_value=10000.0, value=0.0, step=0.01)
        p = st.number_input("Enter Phosphorous value", min_value=0.0, max_value=10000.0, value=0.0, step=0.01)
        k = st.number_input("Enter Potassium value", min_value=0.0, max_value=10000.0, value=0.0, step=0.01)

        if st.button("Predict"):
            # Call the function that predicts the soil health card values from manual input
            py = {}
            # redgram-random forest

            data_rg = pd.read_csv('C:/Users/hbadh/Desktop/Mini Project/Datasets/Crop-wise/Original/Redgram.csv')
            min_max_data_rg = pd.read_csv('C:/Users/hbadh/Desktop/Mini Project/Datasets/Crop-wise/Normalized/Redgram_N.csv')
            min_val_n = min_max_data_rg['N'].min()
            warnings.filterwarnings('ignore', category=UserWarning)
            max_val_n = min_max_data_rg['N'].max()
            min_val_p = min_max_data_rg['P'].min()
            max_val_p = min_max_data_rg['P'].max()
            min_val_k = min_max_data_rg['K'].min()
            max_val_k = min_max_data_rg['K'].max()
            new_n = (n - min_val_n) / (max_val_n - min_val_n)
            new_p = (p - min_val_p) / (max_val_p - min_val_p)
            new_k = (k - min_val_k) / (max_val_k - min_val_k)
            new_data = [[new_n, new_p, new_k]]
            X = data_rg[['N', 'P', 'K']]
            y = data_rg['Yield']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
            model = RandomForestRegressor(n_estimators=100, random_state=2)
            model.fit(X_train, y_train)
            predicted_yield_rg = model.predict(new_data)
            py['redgram'] = predicted_yield_rg

            # rice-svr
            data_r = pd.read_csv('C:/Users/hbadh/Desktop/Mini Project/Datasets/Crop-wise/Original/Rice.csv')
            min_max_data_r = pd.read_csv('C:/Users/hbadh/Desktop/Mini Project/Datasets/Crop-wise/Normalized/Rice_N.csv')
            min_val_n = min_max_data_r['N'].min()
            max_val_n = min_max_data_r['N'].max()
            min_val_p = min_max_data_r['P'].min()
            max_val_p = min_max_data_r['P'].max()
            min_val_k = min_max_data_r['K'].min()
            max_val_k = min_max_data_r['K'].max()
            new_n = (n - min_val_n) / (max_val_n - min_val_n)
            new_p = (p - min_val_p) / (max_val_p - min_val_p)
            new_k = (k - min_val_k) / (max_val_k - min_val_k)
            new_data = [[new_n, new_p, new_k]]
            X_r = data_r[['N', 'P', 'K']]
            y_r = data_r['Yield']
            X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y, test_size=0.2, random_state=2)
            model = SVR(kernel='linear', C=1.0, epsilon=0.2)
            model.fit(X_train_r, y_train_r)
            predicted_yield_r = model.predict(new_data)
            py['rice'] = predicted_yield_r

            # maize-gradient boosting regressor
            data_m = pd.read_csv('C:/Users/hbadh/Desktop/Mini Project/Datasets/Crop-wise/Original/Maize.csv')
            min_max_data_m = pd.read_csv('C:/Users/hbadh/Desktop/Mini Project/Datasets/Crop-wise/Normalized/Maize_N.csv')
            min_val_n = min_max_data_m['N'].min()
            max_val_n = min_max_data_m['N'].max()
            min_val_p = min_max_data_m['P'].min()
            max_val_p = min_max_data_m['P'].max()
            min_val_k = min_max_data_m['K'].min()
            max_val_k = min_max_data_m['K'].max()
            new_n = (n - min_val_n) / (max_val_n - min_val_n)
            new_p = (p - min_val_p) / (max_val_p - min_val_p)
            new_k = (k - min_val_k) / (max_val_k - min_val_k)
            new_data = [[new_n, new_p, new_k]]
            X = data_m[['N', 'P', 'K']]
            y = data_m['Yield']
            X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X, y, test_size=0.3, random_state=3)
            model = GradientBoostingRegressor(random_state=2)
            model.fit(X_train_m, y_train_m)
            predicted_yield_m = model.predict(new_data)
            py['maize'] = predicted_yield_m

            # groundnut svr
            data_g = pd.read_csv('C:/Users/hbadh/Desktop/Mini Project/Datasets/Crop-wise/Original/Groundnut_cleaned.csv')
            min_max_data_g = pd.read_csv('C:/Users/hbadh/Desktop/Mini Project/Datasets/Crop-wise/Normalized/Groundnut_cleaned_N.csv')
            min_val_n = min_max_data_g['N'].min()
            max_val_n = min_max_data_g['N'].max()
            min_val_p = min_max_data_g['P'].min()
            max_val_p = min_max_data_g['P'].max()
            min_val_k = min_max_data_g['K'].min()
            max_val_k = min_max_data_g['K'].max()
            new_n = (n - min_val_n) / (max_val_n - min_val_n)
            new_p = (p - min_val_p) / (max_val_p - min_val_p)
            new_k = (k - min_val_k) / (max_val_k - min_val_k)
            new_data = [[new_n, new_p, new_k]]
            X = data_g[['N', 'P', 'K']]
            y = data_g['Yield']
            X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X, y, test_size=0.3, random_state=2)
            model = HistGradientBoostingRegressor(random_state=2)
            model.fit(X_train_g, y_train_g)
            predicted_yield_g = model.predict(new_data)
            py['groundnut'] = predicted_yield_g
            sorted_crop_yield = {k: v for k, v in sorted(py.items(), key=lambda item: item[1], reverse=True)}
            cy = next(iter(sorted_crop_yield.items()))
            st.write(cy[0], cy[1][0])




# Call the function that predicts the soil health card values
# Display the output using st.write()

# Define the layout of the about us page
def about():
    st.title("About us")
    # Add a section for the project description
    st.header("Project Description")
    st.write("Food production and prediction is getting depleted due to unnatural climatic changes, which will adversely affect the economy of farmers by getting a poor yield and also help the farmers to remain less familiar in forecasting the future crops. This way it helps the farmer in such a way to guide them for sowing the reasonable crops by deploying machine learning, one of the advanced technologies in crop prediction.")

    st.header("Objectives")
    st.write("Our objective is to develop a model that would accurately suggest crop and predict yield in order to help armers take smart decisions in crop selection. We integerated the model to an easy to use interface .")
    # Add a section to display group members' details
    # Add a section to display group members' details
    st.header("Group Members")
    st.write("3rd Year CSD, Batch-D11")

    # Add a container to hold the pictures and details of each group member
    members_container = st.container()

    # Add the details for each group member to the container
    with members_container:
        col1, col2, col3, col4 = st.columns(4)

        # Add the details for the first group member
        with col1:
            st.image("images/AnuDeepthi.jpg", width=100)
            st.write("**Name:** K Anu Deepthi")
            st.write("**Roll no:** 20251A6712")

        # Add the details for the second group member
        with col2:
            st.image("images/Shreya.jpg", width=100)
            st.write("**Name:** Shreya A")
            st.write("**Roll no:** 20251A6727 ")

        # Add the details for the third group member
        with col3:
            st.image("images/SaiAmogha.jpg", width=100)
            st.write("**Name:** U Sai Amogha")
            st.write("**Role:** 20251A6728")

        # Add the details for the fourth group member
        with col4:
            st.image("images/SiriVennela.jpg", width=100)
            st.write("**Name:** Siri Vennela B")
            st.write("**Roll no:** 20251A6731")

# Create a navigation menu
with st.sidebar:
    choice = option_menu("Main Menu", ["Home", "Crop Yield Predictor", "About"],
        icons=['house', 'cloud-upload','list-task' ], menu_icon="cast", default_index=1)
# Show the appropriate page based on user's choice
if choice == "Home":
    home()
elif choice == "Crop Yield Predictor":
    upload()
elif choice == "About":
    about()

