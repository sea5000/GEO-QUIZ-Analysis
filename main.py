import os
import csv
from firebase_Obj import GeoPull
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Replace 4 with the number of logical cores on your system

geo = GeoPull()
data = []
date = "5-5-25"
with open(f'./{date} Data/{date}-DataPull.csv', 'r', newline='', encoding="utf-8-sig") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        data.append(row)
data_dict = {index-1: dict(zip(data[0], row)) for index, row in enumerate(data[1:], start=1)}
for k, i in data_dict.items():
    i['quizResults'] = eval(i['quizResults'])
geo.dataRaw = data_dict

# geo.writeMap(test='europe', name=f'./{date} Data/Austria/{date}-Austria-EUMAP', w=True, filter='cFilter',cFilter='Austria',randomState=2)
# geo.writeMap(test='us', name=f'./{date} Data/Austria/{date}-Austria-USMAP', w=True, filter='cFilter',cFilter='Austria',randomState=2)
# geo.PCAOutput(test='europe', name=f'./{date} Data/Austria/{date}-PCA-Austria-EUMAP', w=True, filter='cFilter',cFilter='Austria',kMeansNum=8, showPlot=False,randomState=2)
# geo.PCAOutput(test='us', name=f'./{date} Data/Austria/{date}-PCA-Austria-USMAP', w=True, filter='cFilter',cFilter='Austria',kMeansNum=8, showPlot=False,randomState=2)

geo.writeMap(test='europe', name=f'./{date} Data/Correctness/{date}-US-EUMAP', w=True, filter='filter-us',randomState=2)
geo.writeMap(test='europe', name=f'./{date} Data/Correctness/{date}-EU-EUMAP', w=True, filter='filter-europe',randomState=2)
geo.writeMap(test='us', name=f'./{date} Data/Correctness/{date}-US-USMAP', w=True, filter='filter-us',randomState=2)
geo.writeMap(test='us', name=f'./{date} Data/Correctness/{date}-EU-USMAP', w=True, filter='filter-europe',randomState=2)
geo.writeMap(test='us', name=f'./{date} Data/Combined/{date}-Combined-USMAP', w=True,randomState=2)
geo.writeMap(test='europe', name=f'./{date} Data/Combined/{date}-Combined-EUMAP', w=True,randomState=2)


geo.PCAOutput(test='europe', name=f'./{date} Data/PCA/{date}-PCA-US-EUMAP', w=True, filter='filter-us',kMeansNum=8, showPlot=False,randomState=2)
geo.PCAOutput(test='europe', name=f'./{date} Data/PCA/{date}-PCA-EU-EUMAP', w=True, filter='filter-europe',kMeansNum=8, showPlot=False,randomState=2)
geo.PCAOutput(test='us', name=f'./{date} Data/PCA/{date}-PCA-US-USMAP', w=True, filter='filter-us',kMeansNum=8, showPlot=False,randomState=2)
geo.PCAOutput(test='us', name=f'./{date} Data/PCA/{date}-PCA-EU-USMAP', w=True, filter='filter-europe',kMeansNum=8, showPlot=False,randomState=2)

geo.writeMap(test='europe', name=f'./{date}-Combined-EUMAP', w=True,randomState=2)
geo.PCAOutput(test='europe', name=f'./{date} Data/PCA/{date}-PCA-COMBINED-EUMAP', w=True, kMeansNum=8, showPlot=False,randomState=2)
geo.PCAOutput(test='us', name=f'./{date} Data/PCA/{date}-PCA-COMBINED-USMAP', w=True, kMeansNum=8, showPlot=False,randomState=2)