import pandas as pd
import plotly.express as px
import streamlit as st
import gdown

# Download Superstore CSV from Google Drive
file_id = "17rc8ezNVar3eBL0uAixrQXQnqNIlXScD"
url = f"https://drive.google.com/uc?id={file_id}"
output = "Sample-Superstore.csv"
gdown.download(url, output, quiet=True)

# Load CSV
df = pd.read_csv(output, encoding='latin1')
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Month_Year'] = df['Order Date'].dt.to_period('M').astype(str)

st.title("Superstore Sales Dashboard")

# Sales trend plot
sales_trend = df.groupby('Month_Year')['Sales'].sum().reset_index()
fig = px.line(sales_trend, x='Month_Year', y='Sales', markers=True, title="Sales Trend Over Time")
fig.update_xaxes(tickangle=45)
st.plotly_chart(fig)

import streamlit as st
import pandas as pd
import plotly.express as px

# Load CSV
df = pd.read_csv("/Users/helly0311/Desktop/Sample-Superstore.csv", encoding='latin1')
st.sidebar.header("Filters")
region_filter = st.sidebar.selectbox("Select Region", ["All"] + list(df['Region'].unique()))
category_filter = st.sidebar.selectbox("Select Category", ["All"] + list(df['Category'].unique()))
subcat_filter = st.sidebar.selectbox("Select Sub-Category", ["All"] + list(df['Sub-Category'].unique()))

# Filter DataFrame safely
filtered_df = df.copy()
if region_filter != "All":
    filtered_df = filtered_df[filtered_df['Region'] == region_filter].copy()
if category_filter != "All":
    filtered_df = filtered_df[filtered_df['Category'] == category_filter].copy()
if subcat_filter != "All":
    filtered_df = filtered_df[filtered_df['Sub-Category'] == subcat_filter].copy()


st.subheader("Key Metrics")
total_sales = filtered_df['Sales'].sum()
total_profit = filtered_df['Profit'].sum()
total_orders = filtered_df['Order ID'].nunique()

col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", f"${total_sales:,.2f}")
col2.metric("Total Profit", f"${total_profit:,.2f}")
col3.metric("Total Orders", total_orders)
st.subheader("Sales by Category")
category_sales = filtered_df.groupby('Category')['Sales'].sum().reset_index()
fig1 = px.bar(category_sales, x='Category', y='Sales', title="Sales by Category", height=400)
st.plotly_chart(fig1, width='stretch')

st.subheader("Sales by Sub-Category")
subcat_sales = filtered_df.groupby('Sub-Category')['Sales'].sum().reset_index()
fig2 = px.bar(subcat_sales, x='Sub-Category', y='Sales', title="Sales by Sub-Category", height=400)
st.plotly_chart(fig2, width='stretch')

st.subheader("Sales Trend Over Time")
filtered_df['Month'] = pd.to_datetime(filtered_df['Order Date']).dt.to_period('M')
sales_time = filtered_df.groupby('Month')['Sales'].sum().reset_index()
sales_time['Month'] = sales_time['Month'].dt.to_timestamp()

fig3 = px.line(sales_time, x='Month', y='Sales', title="Monthly Sales Trend", height=400)
st.plotly_chart(fig3, width='stretch')

st.subheader("Sales vs Profit")
fig4 = px.scatter(filtered_df, x='Sales', y='Profit', color='Category', 
                  hover_data=['Sub-Category', 'Region'], height=400)
st.plotly_chart(fig4, width='stretch')


st.subheader("Profit Heatmap (Category vs Sub-Category)")

profit_pivot = filtered_df.pivot_table(
    index='Category',
    columns='Sub-Category',
    values='Profit',
    aggfunc='sum'
)

fig_heat = px.imshow(
    profit_pivot,
    color_continuous_scale='RdBu',
    aspect='auto',
    title="Profit Heatmap: Category vs Sub-Category"
)

st.plotly_chart(fig_heat, use_container_width=True)


st.subheader("Price vs Rating Scatter Plot")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df['price'], df['rating'], alpha=0.6)
ax.set_xlabel("Price ($)")
ax.set_ylabel("Rating")
ax.set_title("Relationship Between Price & Rating")

st.pyplot(fig)


# Decision Tree Model

st.subheader("🌳 Decision Tree Price Prediction Model")

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Select features
features = ['rating', 'number_of_reviews']
df_tree = df.dropna(subset=features + ['price'])

X = df_tree[features]
y = df_tree['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)

# Predict + evaluate
y_pred_tree = tree_model.predict(X_test)
mae_tree = mean_absolute_error(y_test, y_pred_tree)

st.write("### 📌 Decision Tree Model Performance")
st.write(f"- **MAE:** ${mae_tree:.2f}")

# User Inputs
st.write("### 🔍 Predict Price Using Decision Tree")
rating_input_tree = st.slider("Rating", 1.0, 5.0, 4.5)
reviews_input_tree = st.number_input("Number of Reviews", 0, 500, 50)

predicted_price_tree = tree_model.predict([[rating_input_tree, reviews_input_tree]])[0]

st.success(f"Predicted Price: **${predicted_price_tree:.2f}**")
