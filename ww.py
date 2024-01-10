import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import category_encoders as ce
import sweetviz as sv

# Set a default value for data
data = pd.DataFrame()

# Sidebar
st.sidebar.title("Data Cleaner App - 'Wired Whisperer'")
st.sidebar.image('https://drive.google.com/uc?export=view&id=1MfJeiu8IDE_D1_jYzGCSwPPot8lcuVv2', use_column_width=True)
st.sidebar.markdown(
    """
    **Sarthak Nagpal**
    
    Aspiring Data Scientist
    
    I am a passionate data science enthusiast with a strong foundation in statistics,
    machine learning, and data analysis. Currently seeking opportunities to apply my
    skills and contribute to real-world projects.

    [GitHub](www.Github.com/SarthakNagpal)
    """
)

# Function to generate Sweetviz report
@st.cache(allow_output_mutation=True)
def generate_eda_report(data):
    report = sv.analyze(data)
    return report

# Function to display Sweetviz report in a separate HTML popup
def display_sweetviz_report(report):
    st.subheader("Exploratory Data Analysis (EDA)")
    st.write("Performing EDA...")

    # Save the report to a temporary HTML file
    report_file_path = "eda_report.html"
    report.show_html(report_file_path, open_browser=False)

    # Display a link to open the report in a new tab
    st.markdown(f'<a href="file:///{report_file_path}" target="_blank">Open Sweetviz Report</a>', unsafe_allow_html=True)

# Function to handle file upload and EDA report generation
def handle_uploaded_file(file):
    global data
    data = pd.read_csv(file)

    st.subheader("DataFrame uploaded:")
    st.write(data)

    # Button to generate and display EDA report
    if st.button("Generate EDA Report"):
        eda_report = generate_eda_report(data)
        display_sweetviz_report(eda_report)

# Main app logic
st.title("Data Preprocessing App")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

# Handle file upload and EDA report generation
if uploaded_file is not None:
    handle_uploaded_file(uploaded_file)

# Rest of the preprocessing steps...
# Unique Values and Value Counts
st.subheader("Unique Values and Value Counts")
if not data.empty:
    column_to_explore = st.selectbox("Select a column:", data.columns)
    st.write("Unique Values:")
    st.write(data[column_to_explore].unique())

    st.write("Value Counts:")
    st.write(data[column_to_explore].value_counts())

# Null Values and Duplicates
st.subheader("Null Values and Duplicates")
if not data.empty:
    st.write("Number of Null Values in Each Column:")
    st.write(data.isnull().sum())

    st.write("Number of Duplicate Entries:")
    st.write(data.duplicated().sum())

# Handle Null Values
st.subheader("Handle Null Values")

columns_with_nulls = data.columns[data.isnull().any()].tolist()
drop_columns = st.multiselect("Select columns to drop null values", columns_with_nulls)
fill_mean_columns = st.multiselect("Select columns to fill with mean", columns_with_nulls)
fill_mode_columns = st.multiselect("Select columns to fill with mode", columns_with_nulls)
fill_median_columns = st.multiselect("Select columns to fill with median", columns_with_nulls)
fill_custom_columns = st.multiselect("Select columns to fill with custom value", columns_with_nulls)

if st.button("Apply Null Value Handling"):
    data.dropna(subset=drop_columns, inplace=True)

    for col in fill_mean_columns:
        data[col].fillna(data[col].mean(), inplace=True)

    for col in fill_mode_columns:
        data[col].fillna(data[col].mode().iloc[0], inplace=True)

    for col in fill_median_columns:
        data[col].fillna(data[col].median(), inplace=True)

    cust_val = st.text_input("Enter custom value:")
    for col in fill_custom_columns:
        data[col].fillna(cust_val, inplace=True)

    st.subheader("Updated DataFrame after Handling Null Values")
    st.write(data)

    # Download Dataset after Handling Null Values
    download_null_btn = st.button("Download CSV (After Null Handling)")
    if download_null_btn:
        st.write("Downloading...")
        data.to_csv("after_null_handling_data.csv", index=False)
        st.success("Download completed!")
        with open("after_null_handling_data.csv", "rb") as file:
            st.download_button(
                label="Download CSV (After Null Handling)",
                data=file,
                key="after_null_handling_data.csv",
                help="Click here to download the CSV file after handling null values."
            )

# Descriptive Statistics
st.subheader("Descriptive Statistics")
if not data.empty:
    st.write("Summary Statistics:")
    st.write(data.describe())

# Correlation Heatmap
st.subheader("Correlation Heatmap")
if not data.empty:
    corr_data = data.corr(numeric_only=True)

    st.write("Top 10 Correlated Columns:")
    top_corr_columns = corr_data.abs().unstack().sort_values(ascending=False).drop_duplicates().head(10)
    st.write(top_corr_columns)

    corr_style = corr_data.style.background_gradient()
    st.write(corr_style)

# Outlier Detection
st.subheader("Outlier Detection")

if not data.empty:
    column_for_outliers = st.selectbox("Select a column for outlier detection:", data.select_dtypes(include='number').columns)

    st.write("Boxplot for Outlier Detection:")
    fig, ax = plt.subplots()
    sns.boxplot(x=data[column_for_outliers], ax=ax)
    st.pyplot(fig)

    z_scores = zscore(data[column_for_outliers])
    outliers = (z_scores > 3) | (z_scores < -3)

    st.write("Number of Outliers:", outliers.sum())
    st.write("Outliers:")
    st.write(data[outliers])

    numeric_columns = data.select_dtypes(include='number').columns
    remove_columns = st.multiselect("Select columns to remove", numeric_columns)
    replace_median_columns = st.multiselect("Select columns to replace with median", numeric_columns)
    replace_custom_columns = st.multiselect("Select columns to replace with custom value", numeric_columns)

    if st.button("Apply Outlier Handling"):
        for column_for_outliers in numeric_columns:
            z_scores = zscore(data[column_for_outliers])
            outliers = (z_scores > 3) | (z_scores < -3)

            data.dropna(subset=remove_columns, inplace=True)

            for col in replace_median_columns:
                median_val = data[col].median()
                data.loc[outliers, col] = median_val

            for col in replace_custom_columns:
                custom_val = st.text_input(f"Enter custom value for {col}:", key=f"custom_value_{col}")
                data.loc[outliers, col] = custom_val

        st.subheader("Updated DataFrame after Handling Outliers")
        st.write(data)

        # Download Dataset after Handling Outliers
        download_outlier_btn = st.button("Download CSV (After Outlier Handling)")
        if download_outlier_btn:
            st.write("Downloading...")
            data.to_csv("after_outlier_handling_data.csv", index=False)
            st.success("Download completed!")
            with open("after_outlier_handling_data.csv", "rb") as file:
                st.download_button(
                    label="Download CSV (After Outlier Handling)",
                    data=file,
                    key="after_outlier_handling_data.csv",
                    help="Click here to download the CSV file after handling outliers."
                )

# Encoding and Data Scaling
st.subheader("Encoding and Data Scaling")

# Encoding
st.subheader("Encoding Methods")
encoding_methods = ['Label Encoding', 'One-Hot Encoding', 'Frequency Encoding', 'Binary Encoding']
selected_encodings = {method: st.multiselect(f"Columns for {method}:", data.columns) for method in encoding_methods}

# Scaling
st.subheader("Scaling Methods")
scaling_methods = ['Standard Scaling', 'Min-Max Scaling']
scale_cols = st.multiselect("Select columns for scaling:", data.select_dtypes(include='number').columns)
selected_scaling = st.multiselect("Select Scaling Methods:", scaling_methods)

if st.button("Apply Encoding and Scaling"):
    for method, cols in selected_encodings.items():
        if method == 'Label Encoding':
            label_encoder = LabelEncoder()
            for col in cols:
                data[col] = label_encoder.fit_transform(data[col])

        elif method == 'One-Hot Encoding':
            data = pd.get_dummies(data, columns=cols)

        elif method == 'Frequency Encoding':
            for col in cols:
                encoding_map = data[col].value_counts(normalize=True).to_dict()
                data[col] = data[col].map(encoding_map)

        elif method == 'Binary Encoding':
            encoder = ce.BinaryEncoder(cols=cols)
            data = encoder.fit_transform(data)

    for method in selected_scaling:
        if method == 'Standard Scaling':
            scaler = StandardScaler()
            for col in scale_cols:
                data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))

        elif method == 'Min-Max Scaling':
            minmax_scaler = MinMaxScaler()
            for col in scale_cols:
                data[col] = minmax_scaler.fit_transform(data[col].values.reshape(-1, 1))

    st.subheader("Updated DataFrame after Encoding and Scaling")
    st.write(data)

    # Download Dataset after Encoding and Scaling
    download_encoding_scaling_btn = st.button("Download CSV (After Encoding and Scaling)")
    if download_encoding_scaling_btn:
        st.write("Downloading...")
        data.to_csv("after_encoding_scaling_data.csv", index=False)
        st.success("Download completed!")
        with open("after_encoding_scaling_data.csv", "rb") as file:
            st.download_button(
                label="Download CSV (After Encoding and Scaling)",
                data=file,
                key="after_encoding_scaling_data.csv",
                help="Click here to download the CSV file after encoding and scaling."
            )
