import os
import csv
from firebase_Obj import GeoPull
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Replace 4 with the number of logical cores on your system

geo = GeoPull()
data = []

with open('./24-4-25/24-4-25-DataPull.csv', 'r', newline='', encoding="utf-8-sig") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        data.append(row)
data_dict = {index-1: dict(zip(data[0], row)) for index, row in enumerate(data[1:], start=1)}
for k, i in data_dict.items():
    i['quizResults'] = eval(i['quizResults'])
geo.dataRaw = data_dict

# geo.writeMap(test='europe', name='./24-4-25/Austria/24-4-25-Austria-EUMAP', w=True, filter='cFilter',cFilter='Austria',randomState=2)
# geo.writeMap(test='us', name='./24-4-25/Austria/24-4-25-Austria-USMAP', w=True, filter='cFilter',cFilter='Austria',randomState=2)
# geo.PCAOutput(test='europe', name='./24-4-25/Austria/24-4-25-PCA-Austria-EUMAP', w=True, filter='cFilter',cFilter='Austria',kMeansNum=8, showPlot=False,randomState=2)
# geo.PCAOutput(test='us', name='./24-4-25/Austria/24-4-25-PCA-Austria-USMAP', w=True, filter='cFilter',cFilter='Austria',kMeansNum=8, showPlot=False,randomState=2)

# geo.writeMap(test='europe', name='./24-4-25/Combined/24-4-25-US-EUMAP', w=True, filter='filter-us',randomState=2)
# geo.writeMap(test='europe', name='./24-4-25/Combined/24-4-25-EU-EUMAP', w=True, filter='filter-europe',randomState=2)
# geo.writeMap(test='us', name='./24-4-25/Combined/24-4-25-US-USMAP', w=True, filter='filter-us',randomState=2)
# geo.writeMap(test='us', name='./24-4-25/Combined/24-4-25-EU-USMAP', w=True, filter='filter-europe',randomState=2)
geo.writeMap(test='us', name='./24-4-25/Combined/24-4-25-Combined-USMAP', w=True,randomState=2)
geo.writeMap(test='europe', name='./24-4-25/Combined/24-4-25-Combined-EUMAP', w=True,randomState=2)


# geo.PCAOutput(test='europe', name='./24-4-25/Combined/24-4-25-PCA-US-EUMAP', w=True, filter='filter-us',kMeansNum=8, showPlot=False,randomState=2)
# geo.PCAOutput(test='europe', name='./24-4-25/Combined/24-4-25-PCA-EU-EUMAP', w=True, filter='filter-europe',kMeansNum=8, showPlot=False,randomState=2)
# geo.PCAOutput(test='us', name='./24-4-25/Combined/24-4-25-PCA-US-USMAP', w=True, filter='filter-us',kMeansNum=8, showPlot=False,randomState=2)
# geo.PCAOutput(test='us', name='./24-4-25/Combined/24-4-25-PCA-EU-USMAP', w=True, filter='filter-europe',kMeansNum=8, showPlot=False,randomState=2)