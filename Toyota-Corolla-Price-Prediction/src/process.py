import numpy as np
import pandas as pd 

df = pd.read_csv("Machine-Learning-Projects/Toyota-Corolla-Price-Prediction/data/raw/ToyotaCorolla.csv")
df = df.drop(["Id", "Age_08_04", "Mfg_Month", "Metallic_Rim", "Radio_cassette", "Tow_Bar", "Backseat_Divider", "Mistlamps",
              "Radio", "Power_Steering", "Central_Lock", "Powered_Windows", "Boardcomputer", "CD_Player", "Automatic_airco",
              "Airco", "Met_Color", "Mfr_Guarantee", "BOVAG_Guarantee", "Guarantee_Period"], axis=1)

df["Model"] = df["Model"].str.replace("?", "")
df["Model"] = df["Model"].str.replace(r"\d+/\d+-Doors", "", regex=True).str.strip()

df["Fuel_Type"] = df["Fuel_Type"].map({"Petrol" : 0, "Diesel" : 1, "CNG" : 2})

df["Total_Airbags"] = df["Airbag_1"] + df["Airbag_2"]
df = df.drop(["Airbag_1", "Airbag_2"], axis=1)

def remove_outliers(df, column):
    q1, q3 = df[column].quantile(0.25), df[column].quantile(0.75)
    iqr = q3 - q1 
    return df[(df[column] >= q1 - (1.5 * iqr)) & (df[column] <= q3 + (1.5 * iqr))]

df = remove_outliers(df, "Price")
df = remove_outliers(df, "KM")

df.to_csv("Machine-Learning-Projects/Toyota-Corolla-Price-Prediction/data/processed/ToyotaCorolla_processed.csv", index=False)
