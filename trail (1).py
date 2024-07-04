# final code
import numpy as np
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


def home():
    css = """
    body {
        background-color: #000000;
    }
    """

    # st.title("Welcome to the Soil Health Card Predictor")
    st.markdown("<h1 style='color: green;'>Welcome to the Crop Suggestor and Yield Predictor</h1>",
                unsafe_allow_html=True)
    # st.write("This web app allows you to predict the health status of your soil based on the values of Nitrogen, Phosphorous and Potassium.")
    #
    # st.write("Click on Menu to navigate to predictor")
    st.markdown(
        "<p style='color: black;'>The aim is to build crop yield prediction model for a given historical data on NPK values and other environmental factors that affect crop yield, the goal is to develop a model that can accurately predict crop yield based on NPK values. The model should able to take into account the interactions between NPK values and provide accurate predictions for different regions and crop types.</p>",
        unsafe_allow_html=True)

    st.markdown(
        "<p style='color: black;'>The site will take npk values as input. The user can enter these values manually or extract these values from the image uploaded. The model calculates yields based on npk values and whichever crop produces the highest yield is suggested to the user with the estimated yield in tons per hectare. The target user of the site our predominately Farmers, Agricultural Research Centers, etc.</p>",
        unsafe_allow_html=True)

    image1 = "Images/rice_1.jpg"
    image2 = "Images/maize_1.jpeg"
    image3 = "Images/groundnut_1.jpeg"
    image4 = "Images/redgram.jpeg"

    # create the first container
    with st.container():
        # create two columns within the container
        col1, col2 = st.columns(2)
        # add content to each column
        with col1:
            st.image(image1, use_column_width=True)
            # st.markdown("<p style='color: blue;'>Rice</p>", unsafe_allow_html=True)
            st.markdown("<h2 style='color: blue;'>Rice</h2>", unsafe_allow_html=True)
            st.write(
                "Rice is an important crop in Telangana, India, and is widely cultivated across the state. The state government has taken several initiatives to promote rice cultivation and increase the productivity of rice crops. In Telangana, rice is mainly cultivated during the kharif season, which typically starts in June and lasts until October. The state government has introduced several initiatives to support rice farmers during the kharif season, such as the provision of seeds, fertilizers, and other inputs at subsidized rates, as well as the implementation of irrigation projects and other infrastructure developments. Rice crops typically require a large amount of nitrogen, as it is essential for plant growth and needs low phosphorus levels & low potassium levels.")
        with col2:
            st.image(image2, use_column_width=True)
            # st.markdown("<p style='color: blue;'>Maize</p>", unsafe_allow_html=True)
            st.markdown("<h2 style='color: blue;'>Maize</h2>", unsafe_allow_html=True)
            st.write(
                "Maize is important as a staple food, economic crop, and a source of genetic diversity. Maize plays a crucial role in global agriculture and food security. Maize (also known as corn) is an important cereal crop grown in many parts of the world, including India. It is a warm-season crop that is mainly grown during the kharif season (June to October) in India. The crop is also used as animal feed and is an important source of energy and protein for livestock. Maize crops require a relatively high amount of nitrogen for optimal growth and yield and need low phosphorus levels & low potassium levels. Adequate nutrient availability is crucial for achieving high yields and maintaining the overall health of maize plants.")
    # create the second container
    with st.container():
        # create two columns within the container
        col1, col2 = st.columns(2)
        # add content to each column
        with col1:
            st.image(image3, use_column_width=True)
            # st.markdown("<p style='color: blue;'>Groundnut</p>", unsafe_allow_html=True)
            st.markdown("<h2 style='color: blue;'>Groundnut</h2>", unsafe_allow_html=True)
            st.write(
                "Groundnut (also known as peanut) is an important crop in many parts of the world, including India. In India, groundnut is mainly grown as a kharif crop (during the monsoon season), although it can also be grown as a rabi crop (during the winter season) in some areas. The crop requires well-drained soil with good water-holding capacityIn India, groundnut is an important source of protein and oil, and is used in a variety of food products. The crop also provides a source of income for many small and marginal farmers in rural areas. Groundnut crops require a moderate amount of nitrogen and needs low phosphorus levels & low potassium levels.")
        with col2:
            st.image(image4, use_column_width=True)
            # st.markdown("<p style='color: blue;'>Redgram</p>", unsafe_allow_html=True)
            st.markdown("<h2 style='color: blue;'>Redgram</h2>", unsafe_allow_html=True)
            st.write(
                "Red gram (also known as pigeon pea) is an important pulse crop grown in India, particularly in the states of Maharashtra, Karnataka, Andhra Pradesh, and Telangana. The crop is also known for its nitrogen-fixing ability, which can help improve soil fertility and reduce the need for nitrogen fertilizers. -Redgram crops have a moderate nitrogen requirement, but they can fix atmospheric nitrogen through a symbiotic relationship with soil bacteria and needs low phosphorus levels & low potassium levels.")
    # st.write("<div style='text-align:center;'>", unsafe_allow_html=True)
    # if st.button("Crop Yield Predictor"):
    #     # Navigate to the upload page
    #     st.experimental_set_query_params(url="upload")
    # st.write("</div>", unsafe_allow_html=True)
    #
    # # Render the current page based on the URL query parameter
    # if "url" in st.experimental_get_query_params():
    #     if st.experimental_get_query_params()["url"] == "upload":
    #         upload()
    # else:
    #     home()


def algo(n, p, k):
    # redgram-random forest
    py = {}
    data_rg = pd.read_csv('C:/Users/usamo/OneDrive/Desktop/miniP/datasets/Redgram.csv')
    min_max_data_rg = pd.read_csv('C:/Users/usamo/OneDrive/Desktop/miniP/datasets/Redgram_N.csv')
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
    py['Redgram'] = predicted_yield_rg

    # rice-svr
    data_r = pd.read_csv('C:/Users/usamo/OneDrive/Desktop/miniP/datasets/Rice.csv')
    min_max_data_r = pd.read_csv('C:/Users/usamo/OneDrive/Desktop/miniP/datasets/Rice_N.csv')
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
    py['Rice'] = predicted_yield_r

    # maize-gradient boosting regressor
    data_m = pd.read_csv('C:/Users/usamo/OneDrive/Desktop/miniP/datasets/Maize.csv')
    min_max_data_m = pd.read_csv('C:/Users/usamo/OneDrive/Desktop/miniP/datasets/Maize_N.csv')
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
    py['Maize'] = predicted_yield_m

    # groundnut svr
    data_g = pd.read_csv('C:/Users/usamo/OneDrive/Desktop/miniP/datasets/Groundnut.csv')
    min_max_data_g = pd.read_csv('C:/Users/usamo/OneDrive/Desktop/miniP/datasets/Groundnut_N.csv')
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
    py['Groundnut'] = predicted_yield_g
    sorted_crop_yield = {k: v for k, v in sorted(py.items(), key=lambda item: item[1], reverse=True)}

    cy = next(iter(sorted_crop_yield.items()))
    st.write("Crop Suggestion and Yield Prediction")
    st.write(cy[0], cy[1][0], "kg/ha")


# Define the layout of the upload page
# Define the layout of the upload page
def upload():
    st.title("Crop Yield Predictor")
    st.write("Please select an option to input soil data:")

    option = st.radio("Input Method", ("Upload Soil Health Card Image", "Manual Input"))

    if option == "Upload Soil Health Card Image":
        st.write("Please upload an image of your soil health card")
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg"])
        if uploaded_file is not None:
            # Call the function that predicts the soil health card values from the uploaded image
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            # Load image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            # Decode the image file
            img = cv2.imdecode(file_bytes, 1)
            # Display the image
            st.image(img, channels="BGR")
            # Calculate threshold and aspect ratio based on image size
            img_size = img.shape[0] * img.shape[1]
            threshold = int(0.01 * img_size)
            aspect_ratio_range = (1.2, 2.5)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Extract NPK values from contours using OCR
            npk_values = ""
            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if area > threshold and aspect_ratio_range[0] < aspect_ratio < aspect_ratio_range[1]:
                    roi = img[y:y + h, x:x + w]
                    npk_values = pytesseract.image_to_string(roi, lang='eng', config='--psm 6')
                    if npk_values:
                        break

            # If NPK values not found, ask user to input manually
            if not npk_values:
                st.warning("Unable to extract values from Image please enter manually")
                st.stop()
            else:
                # Extract decimal values for N, P, and K using regular expressions
                npk_values = npk_values.strip()
                n_match = re.search(r"Available Nitrogen \(N\) \| (\d+\.\d+)", npk_values)
                p_match = re.search(r"Available Phosphorus \(P\)\| (\d+\.\d+)", npk_values)
                k_match = re.search(r"Available Potassium \(K\) \| (\d+\.\d+)", npk_values)
                if not (n_match and p_match and k_match):
                    st.warning("Unable to extract values from Image please enter manually")
                    st.stop()
                else:
                    n = n_match.group(1)
                    p = p_match.group(1)
                    k = k_match.group(1)
            n = float(n)
            p = float(p)
            k = float(k)
            algo(n, p, k)

        elif option == "Manual Input":
            st.write("Please enter the values of Nitrogen, Phosphorous, and Potassium below.")
            n = st.number_input("Enter Nitrogen value", min_value=0.0, max_value=10000.0, value=0.0, step=0.01)
            p = st.number_input("Enter Phosphorous value", min_value=0.0, max_value=10000.0, value=0.0, step=0.01)
            k = st.number_input("Enter Potassium value", min_value=0.0, max_value=10000.0, value=0.0, step=0.01)

            if st.button("Predict"):
                # Call the function that predicts the soil health card values from manual input
                algo(n, p, k)

    elif option == "Manual Input":
        st.write("Please enter the values of Nitrogen, Phosphorous, and Potassium below.")
        n = st.number_input("Enter Nitrogen value", min_value=0.0, max_value=10000.0, value=0.0, step=0.01)
        p = st.number_input("Enter Phosphorous value", min_value=0.0, max_value=10000.0, value=0.0, step=0.01)
        k = st.number_input("Enter Potassium value", min_value=0.0, max_value=10000.0, value=0.0, step=0.01)

        if st.button("Predict"):
            # Call the function that predicts the soil health card values from manual input
            algo(n, p, k)


# Call the function that predicts the soil health card values
# Display the output using st.write()
# Define the layout of the about us page
def about():
    st.title("About us")

    # Add a section for the project description
    st.header("Motivation")

    st.write(
        " Food production and prediction is getting depleted due to unnatural climatic changes, which will adversely affect the economy of farmers by getting a poor yield and also help the farmers to remain less familiar in forecasting the future crops. This way it helps the farmer in such a way to guide them for sowing the reasonable crops by deploying machine learning, one of the advanced technologies in crop prediction.")

    st.header("Objectives")
    st.write(
        "Our objective is to develop a model that would accurately suggest crop and predict yield in order to help armers take smart decisions in crop selection. We integerated the model to an easy to use interface .")
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
            st.image("Images/anu.jpg", width=100)
            st.write("Name: K Anu deepthi")
            st.write("Roll no: 20251A6712")

        # Add the details for the second group member
        with col2:
            st.image("Images/shreya.jpg", width=100)
            st.write("Name: Shreya A")
            st.write("Roll no: 20251A6727 ")

        # Add the details for the third group member
        with col3:
            st.image("Images/amogha.jpg", width=100)
            st.write("Name: U Sai Amogha")
            st.write("Role: 20251A6728")

        # Add the details for the fourth group member
        with col4:
            st.image("Images/venela.jpg", width=100)
            st.write("Name: Siri Vennela B")
            st.write("Roll no: 20251A6731")


# Create a navigation menu
menu = ["Home", "Crop Yield Predictor", "About"]
choice = st.sidebar.selectbox("Menu", menu)

# Show the appropriate page based on user's choice
if choice == "Home":
    home()
elif choice == "Crop Yield Predictor":
    upload()
elif choice == "About":
    about()