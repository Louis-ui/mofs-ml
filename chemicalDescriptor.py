import pandas as pd

chemicDescriptorData = pd.read_csv("database\output.csv")

for index, row in chemicDescriptorData.iterrows():
    count = 0
    if(row["Zn"] != 0):
        count = count + 1
    if(row["Cu"] != 0):
        count = count + 1
    if(row["Cd"] != 0):
        count = count + 1
    if(row["Co"] != 0):
        count = count + 1
    if(row["Mn"] != 0):
        count = count + 1
    if(row["Ag"] != 0):
        count = count + 1
    if(row["Ni"] != 0):
        count = count + 1
    if(row["Fe"] != 0):
        count = count + 1
    if(row["Mo"] != 0):
        count = count + 1
    if(row["In"] != 0):
        count = count + 1
    row["metal type"] = count

chemicDescriptorData_prepare = chemicDescriptorData
