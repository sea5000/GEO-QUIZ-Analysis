import firebase_admin
from firebase_admin import credentials, firestore
import csv
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import xml.etree.ElementTree as ET
import datetime as dt
import sys
from copy import deepcopy
from matplotlib.widgets import Cursor
from matplotlib.backend_bases import MouseEvent

class GeoPull:
    dataRaw = []
    cList = ['Andorra', 'Albania', 'Austria', 'Bosnia and Herzegovina', 'Belgium', 'Bulgaria', 'Belarus', 'Switzerland', 'Cyprus', 'Czechia', 'Germany', 'Denmark', 'Estonia', 'Spain', 'Finland', 'France', 'Greece', 'Croatia', 'Hungary', 'Ireland', 'Iceland', 'Italy', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Latvia', 'Malta', 'Monaco', 'Moldova', 'Montenegro', 'Macedonia', 'Netherlands', 'Norway', 'Poland', 'Portugal', 'Serbia', 'Russia', 'Romania', 'Sweden', 'Slovenia', 'Slovakia', 'San Marino', 'The Vatican', 'Türkiye / Turkey', 'United Kingdom', 'Ukraine']
    sList = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']
    USD = None
    europeD = None
    kMeansV = None
    FreqTable = {}

    def __init__(self, cred=None):
        """
        Initialize the FirestoreObject with a document ID and data.
        :param document_id: The ID of the Firestore document.
        :param data: A dictionary containing the document's data.
        """
        print("Initilizing v3.0 of GEOPULL")
        try:
            cred = credentials.Certificate("../geo-quiz-7ad38-ea4fdf4f6b42.json")
        except:
            print("No Credentials... Shall continue but without a connection to the database.")
        
        try:
            firebase_admin.initialize_app(cred)
            self.cred = cred
        except:
            print("Unable to initialize Firebase... Shall continue but without a connection to the database.")
    def initCheck(self, test=None, filter=None, cFilter:str=None):
        if filter != "cFilter":
            if self.FreqTable.get(filter) == None or self.FreqTable.get(test) == {}:
                self.initData(test=test, filter=filter)
        # elif self.FreqTable.get(test) == None or self.FreqTable.get(test) == {}:
            # self.initData(test=test, filter=filter)
        elif filter == None:
            if self.FreqTable.get('combined') == None or self.FreqTable.get('combined') == {}:
                self.initData(test=test, filter=filter)
        elif filter == 'cFilter':
            if self.FreqTable.get(filter) == None:# or self.FreqTable.get(filter).get(test) == {}:
                self.initData(test=test, filter=filter, cFilter=cFilter)
            elif self.FreqTable[filter]['us'] == {}:
                self.initData(test=test, filter=filter, cFilter=cFilter)
        else:
            self.initData(test=test, filter=filter, cFilter=cFilter)
    def pull(self, w=False, type='csv', name=None):
        """
        Pull data from Firestore and populate the object.
        """
        if name == None and w:
            raise ValueError("Name cannot be None.")
        db = firestore.client() # Establishes firestore

        users_ref = db.collection('scores-combined') # Establishes collection Instance

        docs = users_ref.stream() # pulls all documents in the collection
        for doc in docs:
            self.dataRaw.append(doc.to_dict())
        for k, i in enumerate(self.dataRaw):
            self.dataRaw[k]['time'] = str(dt.datetime.fromtimestamp(i['time']/1000.0).strftime('%Y-%m-%d %H:%M:%S'))
            if len(i) == 4:
                scoreEurope = 0
                scoreUS = 0
                for qN in i['quizResults'].keys():
                    if int(qN) <= 19:
                        if i['quizResults'][qN]['correct'] == True:
                            scoreEurope += 1
                    if int(qN) > 19:
                        if i['quizResults'][qN]['correct'] == True:
                            scoreUS += 1
                self.dataRaw[k]['score'] = [scoreEurope,scoreUS]

        #return self.dataRaw # returns the data pulled from firestore
        if w and name==None:
            raise TypeError("Name cannot be None.")
        elif w:
            self.write(type=type,name=name)
        self.initCheck()
        print(f"Data pulled from Firestore and written to file: {name}.{type}")

    def write(self, type=None, name=None):
        """
        Write the data stored in the object to a file.
        :param type: The type of file to write to ('csv' or 'json').
        :param name: The name of the file to write to (without extension).
        """
        

        if (type != "csv" and type != "json") or type == None:
                raise ValueError("Invalid type. Must be 'csv' or 'json'.")
        if name == None:
            raise ValueError("Name cannot be None.")
        if type == "csv":
            with open(f'{name}.csv', 'w', newline='', encoding='utf-8-sig') as csv_file:
                csv_writer = csv.writer(csv_file)
                
                # Write the header
                header = ["name","origin","time","score-europe","score-us","QCount","quizResults"]
                csv_writer.writerow(header)
                
                # Write the rows
                for row in self.dataRaw:
                    if len(row.values())==4:
                        print("Wrote Scores")
                        scoreEurope = 0
                        scoreUS = 0
                        for qN in row['quizResults'].keys():
                            if int(qN) <= 19:
                                if row['quizResults'][qN]['correct'] == True:
                                    scoreEurope += 1
                            if int(qN) > 19:
                                if row['quizResults'][qN]['correct'] == True:
                                    scoreUS += 1
                            row['score-europe'] = scoreEurope
                            row['score-us'] = scoreUS
                    else:
                        scoreEurope, scoreUS = row['score']
                        row['score-europe'] = scoreEurope
                        row['score-us'] = scoreUS
                for row in self.dataRaw:
                    writeRow = []
                    writeRow.append(row['name'])
                    writeRow.append(row['origin'])
                    writeRow.append(row['time'])
                    writeRow.append(row['score-europe'])
                    writeRow.append(row['score-us'])
                    writeRow.append(len(row['quizResults']))
                    writeRow.append(row['quizResults'])

                    csv_writer.writerow(writeRow)
        elif type == "json":
            data_dict = {index: dict(zip(self.dataRaw[0], row)) for index, row in enumerate(self.dataRaw[1:], start=1)}
            with open(f'{name}.json', 'w') as jsonfile:
                json.dump(data_dict, jsonfile, indent=4)

    def drop(self, verbose=True):
        if self.dataRaw == None:
            raise KeyError('No data found. Try running pull() first.')
        else:
            if verbose:
                for k, i in self.dataRaw.items():
                    print(k,i['name'],"-",i['origin'],"-",i['time'],"-",i['score-europe'],"-",i['score-us'])
                    sys.stdout.flush()

            var = input("Rows to delete: ")
            print(var)
            var = [int(i) for i in var.split(",")]
            for i in var:
                self.dataRaw.pop(i)
            self.initCheck()
     
    def frequencyTable(self, test='europe', filter=None, cFilter:str=None):
        validOptions= {'europe','us'}
        filterOptions = {'filter-us','filter-europe','cFilter'}
        if test not in validOptions:
            raise ValueError(f"Invalid test type. Must be one of {validOptions}.")
        if test == 'europe':
            sList = self.cList
        elif test == 'us':
            sList = self.sList
        if filter in filterOptions:
            if cFilter and filter=="cFilter":
                # print("cFilter: ",cFilter)
                filteredData = {}
                newKey = 0
                for k, i in self.dataRaw.items():
                    if i['origin'] == cFilter:# and i['origin'] != "-----------":
                        filteredData[newKey] = i
                        newKey += 1
                data = filteredData
            elif filter == 'filter-europe' :
                filteredData = {}
                newKey = 0
                for k, i in self.dataRaw.items():
                    if i['origin'] != 'United States of America' and i['origin'] != "-----------":
                        filteredData[newKey] = i
                        newKey += 1
                data = filteredData
            elif filter == 'filter-us':
                filteredData = {}
                newKey = 0
                for k, i in self.dataRaw.items():
                    if i['origin'] == 'United States of America':
                        filteredData[newKey] = i
                        newKey += 1
                data = filteredData
        else:
            data = self.dataRaw
        outputData = []
        qList = len(sList)
        answers = {}
        for state in sList: # Creates a blank output table for each state
            blank = [0]
            blank*= qList
            answers[state] = blank

        dataCopy = data
        for key in range(len(dataCopy)): 
            qResults = data[key]['quizResults']
            for qN in qResults.keys():
                if qResults[qN]['test'] == test:
                    answer = qResults[qN]['answer']
                    guess = qResults[qN]['guess']
                    count = answers[answer][sList.index(guess)]
                    count += 1
                    answers[answer][sList.index(guess)] = count
        key = 0
        for i in answers.keys():
            row = []
            row.append(sList[key])
            row+=answers[i]
            rowSum = sum(answers[i])
            correct  = answers[i][sList.index(i)]
            if rowSum == 0:
                percent = 0.0
            else:
                percent = round(correct/rowSum,2)
            row.append(correct)
            row.append(rowSum)
            row.append(percent)
            outputData.append(row)
            key +=1
        return outputData
    
    def initData(self,test=None, filter=None,cFilter:str=None):
        if self.FreqTable.get('filter-europe') == None:
            self.FreqTable['filter-europe'] = {}
        if self.FreqTable.get('filter-us') == None:
            self.FreqTable['filter-us'] = {}
        if self.FreqTable.get('europe') == None:
            self.FreqTable['europe'] = {}
        if self.FreqTable.get('us') == None:
            self.FreqTable['us'] = {}
        if self.FreqTable.get('combined') == None:
            self.FreqTable['combined'] = {}
                
        if cFilter != None and filter=='cFilter':
            if self.FreqTable.get('cFilter') == None:
                self.FreqTable['cFilter'] = {'us':{},'europe':{}}
        
        for test in ["us","europe"]:
            if filter:
                if filter == 'filter-us':
                    self.FreqTable['filter-us'][test] = self.frequencyTable(test,filter=filter)
                elif filter == 'filter-europe':
                    self.FreqTable['filter-europe'][test] = self.frequencyTable(test,filter=filter)
                elif cFilter != None and filter == 'cFilter':
                    self.FreqTable['cFilter'][test][cFilter] = self.frequencyTable(test,filter=filter,cFilter=cFilter)
            elif filter == None:
                self.FreqTable['combined'][test] = self.frequencyTable(test)

    def writeFreqTable(self, test='europe', name='Output', filter:str=None, cFilter:str=None):
        validOptions= {'europe','us'}
        filterOptions = {'filter-us','filter-europe','cFilter'}
        if test not in validOptions:
            raise ValueError(f"Invalid test type. Must be one of {validOptions}.")
        elif test == 'europe':
            sList = self.cList
            printTitle = "Country"
        elif test == 'us':
            sList = self.sList
            printTitle = "State"
        elif filter not in filterOptions:
            raise ValueError(f"Invalid filter type. Must be one of {filterOptions}.")
        else:
            raise ValueError(f"Invalid test type. Must be one of {validOptions}.")
        self.initCheck(test, filter=filter, cFilter=cFilter)
        if filter in filterOptions:
            if filter == "cFilter":
                # if self.FreqTable.get('cFilter').get(test).get(cFilter) == None:
                #     self.FreqTable['cFilter'][test][cFilter] = self.frequencyTable(test=test, filter=filter, cFilter=cFilter)
                output = self.FreqTable['cFilter'][test][cFilter]
            elif self.FreqTable.get(filter) == None:
                #if self.FreqTable.get(filter).get(test) == None:
                    # self.FreqTable[filter][test] = self.frequencyTable(test=test, filter=filter)
                output = self.FreqTable[filter][test]
            else:
                output = self.FreqTable[filter][test]
        else:
            if filter == None and self.FreqTable.get('combined') == None:
                self.FreqTable['combined'][test] = self.frequencyTable(test=test)
                output = self.FreqTable['combined'][test]
            elif filter == None and self.FreqTable.get('combined') != None:
                output = self.FreqTable['combined'][test]
            elif self.FreqTable.get(test) == None:
                self.FreqTable[test] = self.frequencyTable(test=test)
                output = self.FreqTable[test]
            else:
                output = self.FreqTable[test]
        with open(f'{name}.csv', 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([printTitle]+sList+["Correct","Total","Percent"])
            for k,i in enumerate(output):
                csv_writer.writerow(i)
    def PCAOutput(self, test='europe', name='Output', numComp=2, kMeansNum=8, w=True, randomState=0, filter:str=None, cFilter:str=None,transpose=False,showPlot=True):
        validOptions= {'europe','us'}
        filterOptions = {'filter-us','filter-europe','cFilter'}
        if test not in validOptions:
            raise ValueError(f"Invalid test type. Must be one of {validOptions}.")
        elif test == 'europe':
            sList = self.cList
            printTitle = "Country"
        elif test == 'us':
            sList = self.sList
            printTitle = "State"
        elif filter not in filterOptions:
            raise ValueError(f"Invalid filter type. Must be one of {filterOptions}.")
        else:
            raise ValueError(f"Invalid test type. Must be one of {validOptions}.")
        self.initCheck(test=test,filter=filter, cFilter=cFilter)
        if filter in filterOptions:
            if filter == "cFilter":
                outputData = self.FreqTable['cFilter'][test][cFilter]
            else:
                outputData = self.FreqTable[filter][test]
        else:
            outputData = self.FreqTable[test]
        df = pd.DataFrame(outputData,index=None)#, columns=outputData[0])
        df.drop(df.columns[0], axis=1, inplace=True)  # Drop the first column
        df.drop(df.columns[-3:], axis=1, inplace=True)

        if transpose:
            df = df.transpose()
        data_array = df.to_numpy()

        scaler = StandardScaler()

        # Fit the scaler to the data and transform it (center and normalize)
        scaled_frequency_table = scaler.fit_transform(data_array)

        pca = PCA(n_components=numComp,whiten=True,)
        pca_result = pca.fit_transform(scaled_frequency_table)

        self.kMeansV = KMeans(random_state=randomState,n_clusters=kMeansNum).fit(pca_result)
        plt.figure(figsize=(8, 6))
        
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=self.kMeansV.labels_, cmap='viridis', alpha=0.7)

        #Plot Title Logic
        if test == 'us':
            if filter in filterOptions:
                if filter == 'filter-us':
                    self.title = 'American PCA Results in the US'
                elif filter == 'filter-europe':
                    self.title = 'European PCA Results in the US'
                elif filter == 'cFilter':
                    self.title = f'PCA Results of {cFilter} in the US'
            else:
                self.title = 'Combined PCA Results in the US'
        elif test == 'europe':
            if filter in filterOptions:
                if filter == 'filter-us':
                    self.title = 'American PCA Results in Europe'
                elif filter == 'filter-europe':
                    self.title = 'European PCA Results in Europe'
                elif filter == 'cFilter':
                    self.title = f'PCA Results of {cFilter} in Europe'
            else:
                self.title = 'Combined PCA Results in the US'
        # Add a legend for the clusters
        legend_labels = [f'Cluster {i+1}' for i in range(len(set(self.kMeansV.labels_)))]
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=10) for i in range(len(set(self.kMeansV.labels_)))]
        plt.legend(handles, legend_labels, title="Clusters", loc="best")
        # Add titles and labels
        plt.title(self.title)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        # Add labels to the points

        for i, label in enumerate(sList):
            plt.text(pca_result[i, 0], pca_result[i, 1], label, fontsize=8, ha='right', va='bottom')

        if transpose:
            df = df.transpose()
        df = df.reset_index(drop=True)
        df.columns = sList
        rowTitles = pd.DataFrame(sList)#[i][0] for i in outputData])
        lastThree = pd.DataFrame([outputData[i][-3:] for i in range(0,len(outputData))],columns=['Correct','Sum','Percent']).reset_index(drop=True)

        df_with_clusters = pd.concat([rowTitles,df,lastThree,pd.Series(self.kMeansV.labels_, name='Cluster').reset_index(drop=True)], axis=1)#[outputData[i][0] for i in range(len(outputData))]), pd.Series(self.kMeansV.labels_, name='Cluster')], axis=1)

        if w:
            df_with_clusters.to_csv(f'{name}.csv', index=False)
            plt.savefig(f'{name}.png', dpi=300, bbox_inches='tight')
            self.writeMap(test=test, name=name, w=w, filter=filter, cFilter=cFilter, kMeans=True, numComp=numComp, kMeansNum=kMeansNum, randomState=randomState)
        if showPlot:
            plt.grid()
            plt.show()
            plt.close()
        plt.clf()
    def writeMap(self, test='europe', name='Output', numComp=2, kMeansNum=8, w=True, filter:str=None, cFilter:str=None, kMeans=False, relative=False,randomState=0):
        validOptions= {'europe','us'}
        filterOptions = {'filter-us','filter-europe','cFilter'}
        if test not in validOptions:
            raise ValueError(f"Invalid test type. Must be one of {validOptions}.")
        elif test == 'europe':
            sList = self.cList
            printTitle = "Country"
        elif test == 'us':
            sList = self.sList
            printTitle = "State"
        elif filter not in filterOptions:
            raise ValueError(f"Invalid filter type. Must be one of {filterOptions}.")
        else:
            raise ValueError(f"Invalid test type. Must be one of {validOptions}.")
        self.initCheck(test=test,filter=filter, cFilter=cFilter)
        
        if filter in filterOptions:
            if filter == "cFilter":
                data = self.FreqTable['cFilter'][test][cFilter]
            elif self.FreqTable.get(filter) == None:
                if self.FreqTable.get(filter).get(test) == None:
                    self.FreqTable[filter][test] = self.frequencyTable(test=test, filter=filter)
                    data = self.FreqTable[filter][test]
            else:
                data = self.FreqTable[filter][test]
        elif filter == None:
            if self.FreqTable.get('combined') == None:
                self.FreqTable['combined'][test] = self.frequencyTable(test=test)
                data = self.FreqTable['combined'][test]
            else:
                data = self.FreqTable['combined'][test]
        else:
            if self.FreqTable.get(test) == None:
                self.FreqTable[test] = self.frequencyTable(test=test)
                data = self.FreqTable[test]
            else:
                data = self.FreqTable[test]

        ET.register_namespace("", "http://www.w3.org/2000/svg")
        TitleB1 = None
        TitleB2 = None
        if kMeans:
            if test =="europe":
                if filter == "filter-us":
                    TitleB1 = ET.Element("tspan", x="50%", y='2%', id='TitleB1', style="fill:black; alignment-baseline: central; text-anchor: middle;")
                    TitleB1.text = "PCA Results of American Knowledge"
                    TitleB2 = ET.Element("tspan", x="50%", y='6%', id='TitleB2', style="fill:black; alignment-baseline: central; text-anchor: middle;")
                    TitleB2.text = "on European Geography"
                    #self.title = r'<ns0:tspan xmlns:ns0="http://www.w3.org/2000/svg" style="" x="50%" y="2%" id="TitleB1"></ns0:tspan><ns0:tspan xmlns:ns0="http://www.w3.org/2000/svg" style="fill:black; alignment-baseloe: central; text-anchor: middle;" x="50%" y="6%" id="TitleB1"></ns0:tspan>'
                elif filter=="filter-europe":
                    TitleB1 = ET.Element("tspan", x="50%", y='2%', id='TitleB1', style="fill:black; alignment-baseline: central; text-anchor: middle;")
                    TitleB1.text = "PCA Results of American Knowledge"
                    TitleB2 = ET.Element("tspan", x="50%", y='6%', id='TitleB2', style="fill:black; alignment-baseline: central; text-anchor: middle;")
                    TitleB2.text = "on European Gography"
                elif filter=="cFilter":
                    TitleB1 = ET.Element("tspan", x="50%", y='2%', id='TitleB1', style="fill:black; alignment-baseline: central; text-anchor: middle;")
                    TitleB1.text = f"PCA Results of {cFilter} Knowledge"
                    TitleB2 = ET.Element("tspan", x="50%", y='6%', id='TitleB2', style="fill:black; alignment-baseline: central; text-anchor: middle;")
                    TitleB2.text = "on European Gography"
            elif test == "us":
                if filter == "filter-europe":
                    TitleB1 = ET.Element("tspan", x="50%", y='7%', id='TitleB1', style="fill:black; alignment-baseline: central; text-anchor: middle;")
                    TitleB1.text = "PCA Results of Eurpean Knowledge"
                    TitleB2 = ET.Element("tspan", x="50%", y='12%', id='TitleB2', style="fill:black; alignment-baseline: central; text-anchor: middle;")
                    TitleB2.text = "on American Gography"
                elif filter == "filter-us":
                    TitleB1 = ET.Element("tspan", x="50%", y='7%', id='TitleB1', style="fill:black; alignment-baseline: central; text-anchor: middle;")
                    TitleB1.text = "PCA Results of American Knowledge"
                    TitleB2 = ET.Element("tspan", x="50%", y='12%', id='TitleB2', style="fill:black; alignment-baseline: central; text-anchor: middle;")
                    TitleB2.text = "on American Geography"
                elif filter == "cFilter":
                    TitleB1 = ET.Element("tspan", x="50%", y='7%', id='TitleB1', style="fill:black; alignment-baseline: central; text-anchor: middle;")
                    TitleB1.text = f"PCA Results of {cFilter} Knowledge"
                    TitleB2 = ET.Element("tspan", x="50%", y='12%', id='TitleB2', style="fill:black; alignment-baseline: central; text-anchor: middle;")
                    TitleB2.text = "on American Geography"
            if self.kMeansV == None:
                raise ValueError("kMeansV == NONE -sea")
            num_clusters = len(set(self.kMeansV.labels_))
            cmap = plt.cm.get_cmap('viridis', num_clusters)
            state_cluster_map = {state: int(cluster) for state, cluster in zip(sList, self.kMeansV.labels_)}
            cluster_colors={}

            cluster_colors = {cluster: to_hex(cmap(cluster)[:3]) for cluster in range(num_clusters)}

            # Map states to their corresponding colors
            state_color_map = {state: cluster_colors[cluster] for state, cluster in state_cluster_map.items()}
        else:
            if test =="europe":
                if filter == "filter-us":
                    TitleB1 = ET.Element("tspan", x="50%", y='2%', id='TitleB1', style="fill:black; alignment-baseline: central; text-anchor: middle;")
                    TitleB1.text = "How Often Americans"
                    TitleB2 = ET.Element("tspan", x="50%", y='6%', id='TitleB2', style="fill:black; alignment-baseline: central; text-anchor: middle;")
                    TitleB2.text = "are Correct on European Geography"
                elif filter=="filter-europe":
                    TitleB1 = ET.Element("tspan", x="50%", y='2%', id='TitleB1', style="fill:black; alignment-baseline: central; text-anchor: middle;")
                    TitleB1.text = "How Often Europeans"
                    TitleB2 = ET.Element("tspan", x="50%", y='6%', id='TitleB2', style="fill:black; alignment-baseline: central; text-anchor: middle;")
                    TitleB2.text = "are Correct on European Geography"
                elif filter=="cFilter":
                    TitleB1 = ET.Element("tspan", x="50%", y='2%', id='TitleB1', style="fill:black; alignment-baseline: central; text-anchor: middle;")
                    TitleB1.text = f"How Often People From {cFilter}"
                    TitleB2 = ET.Element("tspan", x="50%", y='6%', id='TitleB2', style="fill:black; alignment-baseline: central; text-anchor: middle;")
                    TitleB2.text = "are Correct on European Geography"
                elif filter==None:
                    TitleB1 = ET.Element("tspan", x="50%", y='2%', id='TitleB1', style="fill:black; alignment-baseline: central; text-anchor: middle;")
                    TitleB1.text = f"How Often People"
                    TitleB2 = ET.Element("tspan", x="50%", y='6%', id='TitleB2', style="fill:black; alignment-baseline: central; text-anchor: middle;")
                    TitleB2.text = "are Correct on European Geography"
            elif test == "us":
                if filter == "filter-europe":
                    TitleB1 = ET.Element("tspan", x="50%", y='7%', id='TitleB1', style="fill:black; alignment-baseline: central; text-anchor: middle;")
                    TitleB1.text = "How Often Europeans"
                    TitleB2 = ET.Element("tspan", x="50%", y='12%', id='TitleB2', style="fill:black; alignment-baseline: central; text-anchor: middle;")
                    TitleB2.text = "are Correct on American Geography"
                elif filter == "filter-us":
                    TitleB1 = ET.Element("tspan", x="50%", y='7%', id='TitleB1', style="fill:black; alignment-baseline: central; text-anchor: middle;")
                    TitleB1.text = "How Often Americans"
                    TitleB2 = ET.Element("tspan", x="50%", y='12%', id='TitleB2', style="fill:black; alignment-baseline: central; text-anchor: middle;")
                    TitleB2.text = "are Correct on American Geography"
                elif filter=="cFilter":
                    TitleB1 = ET.Element("tspan", x="50%", y='2%', id='TitleB1', style="fill:black; alignment-baseline: central; text-anchor: middle;")
                    TitleB1.text = f"How Often People From {cFilter}"
                    TitleB2 = ET.Element("tspan", x="50%", y='6%', id='TitleB2', style="fill:black; alignment-baseline: central; text-anchor: middle;")
                    TitleB2.text = "are Correct on American Geography"
                elif filter==None:
                    TitleB1 = ET.Element("tspan", x="50%", y='2%', id='TitleB1', style="fill:black; alignment-baseline: central; text-anchor: middle;")
                    TitleB1.text = f"How Often People"
                    TitleB2 = ET.Element("tspan", x="50%", y='6%', id='TitleB2', style="fill:black; alignment-baseline: central; text-anchor: middle;")
                    TitleB2.text = "are Correct on American Geography"
            # Define a custom red-yellow-green colormap
            colors = ['#FF0000', '#FFFF00', '#289800']  # Red, Yellow, Green
            if relative:
                clustersC = set()
                clustersD = []

                # Collect unique values and map them
                for i in range(len(data)):
                    value = (data[i][len(data[0]) - 1]*100)  # Access 4th-to-last column
                    clustersC.add(str(int(value)))
                    clustersD.append(str(int(value)))

                # Sort clustersC and ensure consistent mapping
                clustersC = sorted([int(i) for i in clustersC])  # Sort the unique values

                stateClusters = []
                for i in clustersD:
                    if int(i) in clustersC:
                        stateClusters.append(str(clustersC.index(int(i))))
                    else:
                        print(f"Warning: {i} not found in clustersC")

                # Create the colormap
                num_clusters = len(clustersC)
                cmap = mcolors.LinearSegmentedColormap.from_list('RedYellowGreen', colors, N=num_clusters+1)

                # Map clusters to colors
                cluster_colors = {cluster: to_hex(cmap(cluster / (num_clusters - 1))[:3]) for cluster in range(num_clusters)}
                state_cluster_map = {state: int(cluster) for state, cluster in zip(sList, stateClusters)}
                state_color_map = {state: cluster_colors[cluster] for state, cluster in state_cluster_map.items()}

                minimum = min([int(i) for i in clustersC])
                maximum = max([int(i) for i in clustersC])
            else:
                clustersC = set()
                clustersD = []
                # print(data)
                for i in range(len(data[0])-4):
                    clustersC.add(data[i][len(data[0])-1]*100)
                    clustersD.append(str(int(data[i][len(data[0])-1]*100)))
                num_clusters = 101
                cmap = mcolors.LinearSegmentedColormap.from_list('RedYellowGreen', colors, N=100)
                state_cluster_map = {state: int(cluster) for state, cluster in zip(sList, [str(int(data[i][len(data[0])-1]*100)) for i in range(len(data[0])-4)])}
                cluster_colors={}
                cluster_colors = {cluster: to_hex(cmap(cluster)[:3]) for cluster in range(num_clusters)}

                # Map states to their corresponding colors
                state_color_map = {state: cluster_colors[cluster] for state, cluster in state_cluster_map.items()}

        if w:
            if test == 'us':
                if not kMeans:
                    tree = ET.parse('./maps/us-scale-text.svg')
                    self.writeFreqTable(test=test, name=name, filter=filter, cFilter=cFilter)
                else:
                    tree = ET.parse('./maps/us-text.svg')
            elif test == 'europe':
                if not kMeans:
                    tree = ET.parse('./maps/europe-scale-text.svg')
                    self.writeFreqTable(test=test, name=name, filter=filter, cFilter=cFilter)
                else:
                    tree = ET.parse('./maps/europe-text.svg')
            root = tree.getroot()

            id_to_color = state_color_map
            # Iterate through all elements in the SVG
            for element in root.iter():
                element_id = element.get('id')
                if str(element_id) == 'TitleA':
                    element.append(TitleB1)
                    element.append(TitleB2)
                if str(element_id) == 'min' and relative:
                    element.text = str(minimum)
                if str(element_id) == 'max' and relative:
                    element.text = str(maximum)
                if str(element_id) in id_to_color.keys():
                    # Update the fill color
                    style = element.get('style', '')
                    style_dict = dict(item.split(':') for item in style.split(';') if item)
                    style_dict['fill'] = id_to_color[element_id]
                    element.set('style', ';'.join(f'{k}:{v}' for k, v in style_dict.items()))
            tree.write(f'{name}.svg')

        # cairosvg.svg2png(url=f'{name}.svg', write_to=f'{name}-MAP.png')


        print(f'Outputted map "{name}" with param test: "{test}", filter: "{filter}"')
        print("================================================================")


    def __repr__(self):
        """
        String representation of the FirestoreObject.
        :return: A string representation of the object.
        """
        return f"FirestoreObject(document_id={self.document_id}, data={self.data})"