
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="RFM Customer Segmentation",
    page_icon="📊",
    layout="wide"
)
st.title("RFM Customer Segmentation")


uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
else:
    # Use default file from repo
    data = pd.read_csv("data (1).csv", encoding="ISO-8859-1") 
    st.write("Data is already loaded. Click on 'Run RFM' to proceed.")
    st.subheader("Data Preview")
    st.dataframe(data.head())
    
    if st.button("Run RFM Analysis"):
    
        data = data.dropna()
        data['CustomerID'] = data['CustomerID'].astype(str)
        data["Amount"] = data['Quantity'] * data['UnitPrice']
        rfm_m = data.groupby('CustomerID')['Amount'].sum().reset_index()
        
        rfm_f = data.groupby("CustomerID")["InvoiceNo"].count().reset_index()
        rfm_f.columns = ['CustomerID', 'Frequency']
        
        rfm = pd.merge(rfm_m, rfm_f, on="CustomerID", how="inner")
        
        data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], errors='coerce', dayfirst=True)
        max_date = data['InvoiceDate'].max()
        data['diff'] = max_date - data['InvoiceDate']
        rfm_p = data.groupby('CustomerID')['diff'].min().reset_index()
        rfm_p['Recency'] = rfm_p['diff'].apply(lambda x: x.days if pd.notnull(x) else np.nan)
        rfm_p = rfm_p.drop('diff', axis=1)
        rfm = pd.merge(rfm, rfm_p, on="CustomerID", how="inner")
        rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']
        

        for col in ['Amount', 'Frequency', 'Recency']:
            Q1 = rfm[col].quantile(0.05)
            Q3 = rfm[col].quantile(0.95)
            IQR = Q3 - Q1
            rfm = rfm[(rfm[col] >= Q1 - 1.5*IQR) & (rfm[col] <= Q3 + 1.5*IQR)]
        
        rfm_df = rfm[['Amount', 'Frequency', 'Recency']]
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_df)
        rfm_scaled = pd.DataFrame(rfm_scaled, columns=['Amount', 'Frequency', 'Recency'])
        
        kmeans = KMeans(n_clusters=3, max_iter=300, random_state=42)
        rfm['Cluster_ID'] = kmeans.fit_predict(rfm_scaled)
        
        # Visualizations
        # st.subheader("Outliers distribution")
        # fig0, ax0 = plt.subplots(figsize=(10,6))
        # sns.boxplot(data=rfm[['Amount','Frequency','Recency']], orient="v", palette="Set2", whis=1.5, ax=ax0)
        # st.pyplot(fig0)
        with st.container():
         col1, col2, col3 = st.columns(3)

         with col1:
             st.subheader("Amount by Cluster")
             fig1, ax1 = plt.subplots()                                                                                 
             sns.stripplot(x="Cluster_ID", y="Amount", data=rfm, ax=ax1)
             st.pyplot(fig1)

         with col2:
             st.subheader("Frequency by Cluster")
             fig2, ax2 = plt.subplots()
             sns.stripplot(x="Cluster_ID", y="Frequency", data=rfm, ax=ax2)
             st.pyplot(fig2)

         with col3:
             st.subheader("Recency by Cluster")
             fig3, ax3 = plt.subplots()
             sns.stripplot(x="Cluster_ID", y="Recency", data=rfm, ax=ax3)
             st.pyplot(fig3)
        
         st.success("RFM clustering completed!")

