import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.neighbors import NearestNeighbors
import ipywidgets as widgets
from ipywidgets import interact
import numpy as np
import streamlit_option_menu
from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu(
    menu_title = "Project MAI3002",
    options = ["Introduction","Data preparation","Data exploration and cleaning","Describe and Visualize the data","Data Analysis", "Conclusion"],
    icons = ["chat-dots","list-task","search","bar-chart-line","graph-up", "folder"],
    menu_icon = "cast",
    default_index = 0,
    #orientation = "horizontal",
)
    
if selected == "Introduction":
    st.title('Introduction')
   # Corrected URL for the raw CSV file
    url = 'https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/main/Framingham%20Dataset.csv'
    #allow all the columns to be visible
    pd.set_option('display.max_columns', None)
    # Read the CSV file from the URL
    df = pd.read_csv(url)





if selected == "Data preparation":
    st.title("Data preparation")
    # Corrected URL for the raw CSV file
    url = 'https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/main/Framingham%20Dataset.csv'
    #allow all the columns to be visible
    pd.set_option('display.max_columns', None)
    # Read the CSV file from the URL
    df = pd.read_csv(url)
    #selection of relevant rows and columns for research question, put into new dataset
    df_rq=df[['BMI', 'AGE', 'SEX', 'TOTCHOL', 'SYSBP', 'DIABP', 'CURSMOKE','DIABETES', 'BPMEDS', 'HEARTRTE', 'GLUCOSE','ANYCHD','PERIOD']]
    st.write("### Summary Statistics of relevant rows")
    st.dataframe(df_rq.describe())

    
    #heatmap
    # Check if the dataset contains numeric data
    if df_rq.select_dtypes(include=[np.number]).empty:
        st.warning("The dataset does not contain numeric columns for correlation analysis.")
    else:
        # Calculate the correlation matrix
        correlation_matrix = df_rq.corr()

        # Create the heatmap
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(18, 12))
        sns.heatmap(
            correlation_matrix,
            annot=False,
            cmap="RdBu_r",
            linewidths=1,
            center=0,
            cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
            ax=ax,
        )

        # Annotate significant correlations
        for row in range(correlation_matrix.shape[0]):
            for col in range(correlation_matrix.shape[1]):
                correlation_value = correlation_matrix.iloc[row, col]
                if abs(correlation_value) >= 0.5 and row != col:
                    ax.text(
                        col + 0.5,
                        row + 0.5,
                        f"{correlation_value:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=12,
                        weight="bold",
                    )

        ax.set_title("Correlation Heatmap", fontsize=20, weight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, rotation=0)
        st.pyplot(fig)
    



if selected == "Data exploration and cleaning":
    st.title("Data exploration and cleaning")
    st.header("Missing Data")
    # Corrected URL for the raw CSV file
    url = 'https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/main/Framingham%20Dataset.csv'
    #allow all the columns to be visible
    pd.set_option('display.max_columns', None)
    # Read the CSV file from the URL
    df = pd.read_csv(url)
    #selection of relevant rows and columns for research question, put into new dataset
    df_rq=df[['BMI', 'AGE', 'SEX', 'TOTCHOL', 'SYSBP', 'DIABP', 'CURSMOKE','DIABETES', 'BPMEDS', 'HEARTRTE', 'GLUCOSE','ANYCHD','PERIOD']]
    
    missing_values_data = {
    "Column": ["Age", "Systolic Blood Pressure", "Diastolic Blood Pressure", "Cholesterol", "Smoking", "BMI" ],
    "Missing type": ["MCAR", "MAR or MCAR", "MAR or MCAR", "MAR or MNAR", "MNAR", "MAR"],
    "Reasoning": ["Generally easy to report, likely missing due to random error.", "Could be MAR if older or sicker participants avoid measurements, or MCAR if random errors occurred.", "Similar reasoning to sysBP.", "Could be MNAR if higher cholesterol individuals avoid reporting, or MAR if related to age/BMI.", "People may underreport smoking status due to social stigma.", "Missingness likely depends on variables like age or cholesterol, but not on BMI itself."],
    }
    # Create a DataFrame
    mdr = pd.DataFrame(missing_values_data)
    # Display the table
    st.write("### Interactive Table")
    st.dataframe(mdr)

    #Identify missing values in dataset
    missing_values = df_rq.isnull().sum().sum()
    # Display warning message if there are missing values
    if missing_values > 0:
        st.markdown(
         f"<span style='color:red; font-weight:bold;'>Warning: The dataset has {missing_values} missing values.</span>", 
        unsafe_allow_html=True
     )
    else:
        st.success("The dataset has no missing values.")
    
    #missing data over period
    st.header("Further Missing Data Analysis")
    # Check for missing data
    missing_data = df_rq.isnull().sum()
    "Missing Data Count for each column:"
    st.write(missing_data[missing_data > 0])

    # Investigate missing values by period
    if 'PERIOD' in df_rq.columns:
        period_missing = df_rq.groupby('PERIOD').apply(lambda x: x.isnull().sum())

        # Plot heatmap
        st.write("Missing Data Across Examination Periods")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(period_missing, cmap="coolwarm", annot=True, fmt=".0f",
                    linewidths=0.5, annot_kws={"size": 8},
                    cbar_kws={'label': 'Count of Missing Values'})
        plt.title('Missing Data Across Examination Periods', fontsize=16)
        plt.xlabel('Variables', fontsize=12)
        plt.ylabel('Examination Period', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        st.pyplot(fig)
    else:
        st.write("The dataset does not contain a 'PERIOD' column.")
 
    st.write("## Imputation of missing values for different columns:")
    st.markdown("""
- **TOTCHOL**: The missing values of TOTCHOL are reasonable and can therefore be calculated with median imputation.   
- **BMI**: The missing values of BMI are reasonable and can therefore be calculated with median imputation.
- **HEARTRTE**: The missing values of HEARTRTE are reasonable and can therefore be calculated with median imputation.
- **BPMEDS**: The missing values of BPMEDS should be solved with categorical imputation. A new category, 'Unknown,' is created.
- **GLUCOSE**: There is a high amount of missing values in GLUCOSE. Therefore, K-Nearest Neighbors (KNN) is used for imputation.
""")
    # Before imputation: Distribution of GLUCOSE
    col1, col2 = st.columns(2)
    with col1:
        st.write("Distribution of GLUCOSE Before Imputation:")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.histplot(df_rq['GLUCOSE'], bins=30, kde=True, ax=ax1)
        ax1.set_title('GLUCOSE Distribution Before Imputation')
        ax1.set_xlabel('GLUCOSE')
        ax1.set_ylabel('Frequency')
        st.pyplot(fig1)

    # KNN imputation for GLUCOSE (Only for GLUCOSE column)
    # Create a copy of the DataFrame to avoid modifying the original data
    df_rq_knn = df_rq.copy()
    # Initialize KNN imputer
    imputer = KNNImputer(n_neighbors=5)
    df_rq_knn['GLUCOSE'] = imputer.fit_transform(df_rq_knn[['GLUCOSE']])

    with col2:
        # After imputation: Distribution of GLUCOSE
        st.write("Distribution of GLUCOSE After Imputation:")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.histplot(df_rq_knn['GLUCOSE'], bins=30, kde=True, ax=ax2)
        ax2.set_title('GLUCOSE Distribution After Imputation')
        ax2.set_xlabel('GLUCOSE')
        ax2.set_ylabel('Frequency')
        st.pyplot(fig2)

    st.header("Identify, report, correct issues with erroneous data (if any)")
    # List of binary columns
    binary_columns = df_rq[['SEX', 'CURSMOKE', 'DIABETES', 'BPMEDS', 'ANYCHD', 'PERIOD']]

    # Streamlit UI for displaying value counts of binary columns
    st.write("### Value Counts for Binary Data Columns:")

    # Iterate over the binary columns to check value counts and display them
    for col in binary_columns.columns:
        value_count = df_rq[col].value_counts()
        st.write(f"#### {col}:")
        st.write(value_count)
        st.write("")  # Add a newline for better readability


    st.header("Identify and correct outliers")
    # Boxplots to visualize outliers
    st.write("Interactive Variable Visualization")
    # Define color map for variables
    color_map = {
        'BMI': 'lightblue',
        'AGE': 'lightgreen',
        'TOTCHOL': 'salmon',
        'SYSBP': 'gold',
        'DIABP': 'orchid',
        'HEARTRTE': 'lightcoral',
        'GLUCOSE': 'lightpink'
    }
    # Select only numerical columns for visualization
    numerical_columns = ['BMI', 'AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'HEARTRTE', 'GLUCOSE']
    df_rq_numeric = df_rq[numerical_columns]
    # Dropdown menu for selecting a variable
    selected_variable = st.selectbox("Select a variable to visualize:", numerical_columns)

    if selected_variable:
        # Plot the selected variable
        color = color_map.get(selected_variable, 'lightgray')
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.boxplot(data=df_rq_numeric, y=selected_variable, color=color, ax=ax)
        ax.set_title(f'{selected_variable} Boxplot', fontsize=16)
        ax.set_xlabel('Index')
        ax.set_ylabel(selected_variable)
        ax.grid(True)
        # Display the plot in Streamlit
        st.pyplot(fig)






if selected == "Describe and Visualize the data":
    st.title("Describe and Visualize the data")
    # Corrected URL for the raw CSV file
    url = 'https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/main/Framingham%20Dataset.csv'
    #allow all the columns to be visible
    pd.set_option('display.max_columns', None)
    # Read the CSV file from the URL
    df = pd.read_csv(url)
    #selection of relevant rows and columns for research question, put into new dataset
    df_rq=df[['BMI', 'AGE', 'SEX', 'TOTCHOL', 'SYSBP', 'DIABP', 'CURSMOKE','DIABETES', 'BPMEDS', 'HEARTRTE', 'GLUCOSE','ANYCHD','PERIOD']]
    

    # Streamlit Title
    st.header("BMI Calculator")
    # Input fields for weight and height
    weight = st.number_input("Insert your weight in kilograms (kg):", min_value=0.0, format="%.2f")
    height = st.number_input("Insert your height in meters (m):", min_value=0.0, format="%.2f")
    # Calculate BMI if both inputs are valid
    if weight > 0 and height > 0:
        bmi = weight / (height * height)
        # Use columns for side-by-side display
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Your BMI:")
            st.write(round(bmi, 2))
        with col2:
            st.subheader("")
            if bmi < 18.5:
                st.write("Underweight")
            elif 18.5 <= bmi < 24.9:
                st.write("Normal weight")
            elif 25 <= bmi < 29.9:
                st.write("Overweight")
            else:
                st.write("Obesity")
    else:
        st.write("Please enter valid values for weight and height.")





if selected == "Data Analysis":
    # Corrected URL for the raw CSV file
    url = 'https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/main/Framingham%20Dataset.csv'
    #allow all the columns to be visible
    pd.set_option('display.max_columns', None)
    # Read the CSV file from the URL
    df = pd.read_csv(url)
    #selection of relevant rows and columns for research question, put into new dataset
    df_rq=df[['BMI', 'AGE', 'SEX', 'TOTCHOL', 'SYSBP', 'DIABP', 'CURSMOKE','DIABETES', 'BPMEDS', 'HEARTRTE', 'GLUCOSE','ANYCHD','PERIOD']]
    st.write("### Summary Statistics of relevant rows")
    st.dataframe(df_rq.describe())





if selected == "Conclusion":
    # Corrected URL for the raw CSV file
    url = 'https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/main/Framingham%20Dataset.csv'
    #allow all the columns to be visible
    pd.set_option('display.max_columns', None)
    # Read the CSV file from the URL
    df = pd.read_csv(url)
    #selection of relevant rows and columns for research question, put into new dataset
    df_rq=df[['BMI', 'AGE', 'SEX', 'TOTCHOL', 'SYSBP', 'DIABP', 'CURSMOKE','DIABETES', 'BPMEDS', 'HEARTRTE', 'GLUCOSE','ANYCHD','PERIOD']]
    st.write("### Summary Statistics of relevant rows")
    st.dataframe(df_rq.describe())



## Check the initial distribution of GLUCOSE
#plt.figure(figsize=(12, 5))

#plt.subplot(1, 2, 1)
#sns.histplot(df_rq['GLUCOSE'], bins=30, kde=True)
#plt.title('Distribution of GLUCOSE Before Imputation')
#plt.xlabel('GLUCOSE')
#plt.ylabel('Frequency')
#    st.header("Handling the outliers")