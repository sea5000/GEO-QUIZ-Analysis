import os
import csv
from firebase_Obj import GeoPull
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Replace 4 with the number of logical cores on your system

geo = GeoPull()
data = []

with open('4-4-25-DataPull-Cleaned-1.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        data.append(row)
data_dict = {index-1: dict(zip(data[0], row)) for index, row in enumerate(data[1:], start=1)}
for k, i in data_dict.items():
    i['quizResults'] = eval(i['quizResults'])
geo.dataRaw = data_dict

# geo.PCAOutput(test='europe', name='4-4-25-PCA-US-EUMAP', w=True, filter='filter-us',kMeansNum=8, showPlot=False)
# geo.PCAOutput(test='europe', name='4-4-25-PCA-EU-EUMAP', w=True, filter='filter-europe',kMeansNum=8, showPlot=False)
# geo.PCAOutput(test='us', name='4-4-25-PCA-US-USMAP', w=True, filter='filter-us',kMeansNum=8, showPlot=False)
# geo.PCAOutput(test='us', name='4-4-25-PCA-EU-USMAP', w=True, filter='filter-europe',kMeansNum=8, showPlot=False)

# geo.PCAOutput(test='europe', name='4-4-25-PCA-US-EUMAP', w=True, filter='filter-us',kMeansNum=8, showPlot=False)
# geo.PCAOutput(test='europe', name='4-4-25-PCA-EU-EUMAP', w=True, filter='filter-europe',kMeansNum=8, showPlot=False)
# geo.PCAOutput(test='us', name='4-4-25-PCA-US-USMAP', w=True, filter='filter-us',kMeansNum=8, showPlot=False)
geo.PCAOutput(test='us', name='4-4-25-PCA-EU-USMAP', w=True, filter='filter-europe',kMeansNum=8, showPlot=True)