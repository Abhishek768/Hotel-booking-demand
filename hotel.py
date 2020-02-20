import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snp
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix

hotel_data = pd.read_csv('hotel_bookings.csv')

print(hotel_data.head())

hotel_data['total_stay'] = hotel_data['stays_in_weekend_nights'] + hotel_data['stays_in_week_nights']
hotel_data['family_members_count'] = hotel_data['adults'] + hotel_data['children'] + hotel_data['babies']
hotel_data.drop(['company','agent','stays_in_weekend_nights','stays_in_week_nights','adults','children','babies'], axis=1, inplace=True)
hotel_data.dropna(inplace=True)

fig,axes = plt.subplots(nrows=2, ncols=2, figsize=(11,7))
snp.countplot(x='is_canceled', data=hotel_data, hue='hotel', ax=axes[0][0])
snp.countplot(x='arrival_date_month', data=hotel_data, hue='hotel', ax=axes[0][1])
snp.countplot('total_stay', data=hotel_data, ax=axes[1][0])
snp.countplot('family_members_count', data=hotel_data, hue='hotel', ax=axes[1][1])
snp.jointplot(x='family_members_count', y='total_stay', data=hotel_data)



plt.show()