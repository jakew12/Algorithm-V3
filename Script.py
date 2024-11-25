import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import numpy as np

data = pd.read_csv(r'C:\Users\Jake\Downloads\Final Regression Sheet REDACTED(in).csv')

data['Click Rate'] = data['Click Rate'].str.rstrip('%').astype('float') / 100.0
data['Revenue'] = data['Revenue'].fillna(0)

label_encoder_day = LabelEncoder()
data['Day of Week Encoded'] = label_encoder_day.fit_transform(data['Day of Week'])

data['Hour'] = pd.to_datetime(data['Time of Day'], format='%H:%M:%S').dt.hour

tags_columns = ['REDACTED']
time_factors = ['Day of Week Encoded', 'Hour']
features = data[tags_columns + time_factors]

click_rate_target = data['Click Rate']
revenue_target = data['Revenue']

rf_click_model = RandomForestRegressor(random_state=42)
rf_revenue_model = RandomForestRegressor(random_state=42)

rf_click_model.fit(features, click_rate_target)
rf_revenue_model.fit(features, revenue_target)

click_feature_importances = rf_click_model.feature_importances_
revenue_feature_importances = rf_revenue_model.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': tags_columns + time_factors,
    'Click Rate Importance': click_feature_importances,
    'Revenue Importance': revenue_feature_importances
})

click_top_features = feature_importance_df.sort_values(by='Click Rate Importance', ascending=False).reset_index(drop=True)
revenue_top_features = feature_importance_df.sort_values(by='Revenue Importance', ascending=False).reset_index(drop=True)

print("Top Features for Click Rate:")
print(click_top_features)
print("\nTop Features for Revenue:")
print(revenue_top_features)

day_group = data.groupby('Day of Week').agg({'Click Rate': 'mean', 'Revenue': 'mean'}).sort_values(by='Click Rate', ascending=False)
day_group['Rank for Click Rate'] = day_group['Click Rate'].rank(ascending=False)
day_group['Rank for Revenue'] = day_group['Revenue'].rank(ascending=False)

print("\nAverage Performance by Day of Week:")
print(day_group)

hour_group = data.groupby('Hour').agg({'Click Rate': 'mean', 'Revenue': 'mean'}).sort_values(by='Click Rate', ascending=False)
top_hours_click_rate = hour_group.sort_values(by='Click Rate', ascending=False).head(10)
top_hours_revenue = hour_group.sort_values(by='Revenue', ascending=False).head(10)

print("\nTop 10 Hours for Click Rate:")
print(top_hours_click_rate)

print("\nTop 10 Hours for Revenue:")
print(top_hours_revenue)