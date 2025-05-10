# import pandas as pd

# df = pd.read_csv('5-5-25-DataPull.csv', encoding='utf-8-sig')
# print(df.head())
# cList = ['Andorra', 'Albania', 'Austria', 'Bosnia and Herzegovina', 'Belgium', 'Bulgaria', 'Belarus', 'Switzerland', 'Cyprus', 'Czechia', 'Germany', 'Denmark', 'Estonia', 'Spain', 'Finland', 'France', 'Greece', 'Croatia', 'Hungary', 'Ireland', 'Iceland', 'Italy', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Latvia', 'Malta', 'Monaco', 'Moldova', 'Montenegro', 'Macedonia', 'Netherlands', 'Norway', 'Poland', 'Portugal', 'Serbia', 'Russia', 'Romania', 'Sweden', 'Slovenia', 'Slovakia', 'San Marino', 'The Vatican', 'Turkey', 'United Kingdom', 'Ukraine']
# EuropeansP = df[df['origin'].isin(cList)]
# AmericanP = df[df['origin'] == "United States of America"]
# combined = df[(df['origin'] == "United States of America") | (df['origin'].isin(cList))]
# # AmericanP.to_csv('AmericanP.csv', index=False)
# # EuropeansP.to_csv('EuropeansP.csv', index=False)
# print(len(EuropeansP), len(AmericanP), "SUM:", len(EuropeansP) + len(AmericanP))
# print(combined['origin'].value_counts())
# combined['origin'].value_counts().to_csv('Nationality Count.csv')
l = ['0']*10
# print(len(l))
for i in range(1,len(l)):
    print(i)
