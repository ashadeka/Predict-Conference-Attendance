#add the reluctance and desirability to each city
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from geopy.distance import vincenty as vc
import numpy.random as rnd


details = pd.read_csv("Conference location and fees.csv")
split_it = details["Year/Site/Hotel"]
conference_city = []
conference_year = details["Year"]
conference_attendance = details["Attendance"]
adjusted_fee = details["adjusted fee"]

for info in split_it:
    split = info.split("/")
    city = split[1]
    conference_city.append(city)
conference_city = pd.Series(conference_city, name = "City")

df = pd.concat([conference_year, conference_city, adjusted_fee, conference_attendance], axis = 1)
df.set_value(20, "City", "Washington")
df.set_value(4, "City", "Saint Louis")
df.set_value(16, "City", "Saint Louis")
df.set_value(6, "City", "Pittsburg")


df.set_index(keys = ["City"], inplace = True)

preference = pd.read_csv("ConferenceData.csv", index_col= ["City"],usecols = ["City","Desirability"])
preference.dropna(inplace = True)


members = pd.read_csv("2014 Accredited Inst Members.csv", encoding = 'cp1252', index_col =["Institution"], usecols=["Institution"])
geodata = pd.read_csv("merged_2013_PP.csv", encoding = 'cp1252', index_col = ["INSTNM"], 
                      usecols = ["INSTNM","LATITUDE","LONGITUDE"])    



inst = members.join(geodata, how = "left")
inst.dropna(inplace = True)



cities = pd.read_csv("cities.csv",usecols=["city","lat","lng"])
cities.drop_duplicates(cols = ["city"], keep = "last", inplace = True)
cities.set_index(["city"], inplace = True)


final_members = df.join(cities,how = "left")
final_members = final_members.join(preference, how = "left")
final_members.dropna(inplace = True)


#CHICAGO

inst["Chicago-lat"] = final_members.get_value("Chicago","lat")[0]
inst["Chicago-lng"] = final_members.get_value("Chicago","lng")[0]

serie = []
for college in inst.iterrows():
    serie.append(college[1])
    

distances = []
for uni in serie:
    ls = uni.tolist()
    distances.append(vc(ls[:2],ls[2:]).miles)

inst.drop(["Chicago-lat","Chicago-lng"], axis = 1,inplace = True)  
chicago_mean_distance =  sum(distances)/len(distances)

#BOSTON

inst["lat2"] = final_members.get_value("Boston","lat")
inst["lng2"] = final_members.get_value("Boston","lng")

boston_serie= []
for college in inst.iterrows():
    boston_serie.append(college[1])
    
boston_distances = []

for uni in boston_serie:
    ls = uni.tolist()
    boston_distances.append(vc(ls[:2],ls[2:]).miles)

inst.drop(["lat2","lng2"], axis = 1,inplace = True)    
boston_mean_distance =  sum(boston_distances)/len(boston_distances)


#DENVER
inst["lat2"] = final_members.get_value("Denver","lat")[0]
inst["lng2"] = final_members.get_value("Denver","lng")[0]

denver_serie= []
for college in inst.iterrows():
    denver_serie.append(college[1])
    
denver_distances = []

for uni in denver_serie:
    ls = uni.tolist()
    denver_distances.append(vc(ls[:2],ls[2:]).miles)

inst.drop(["lat2","lng2"], axis = 1,inplace = True)    
denver_mean_distance =  sum(denver_distances)/len(denver_distances)

#NEW ORLEANS
inst["lat2"] = final_members.get_value("New Orleans","lat")[0]
inst["lng2"] = final_members.get_value("New Orleans","lng")[0]

no_serie= []
for college in inst.iterrows():
    no_serie.append(college[1])
    
no_distances = []

for uni in no_serie:
    ls = uni.tolist()
    no_distances.append(vc(ls[:2],ls[2:]).miles)
    
inst.drop(["lat2","lng2"], axis = 1,inplace = True)    
new_orleans_mean_distance =  sum(no_distances)/len(no_distances)

#ATLANTA

inst["lat2"] = final_members.get_value("Atlanta","lat")
inst["lng2"] = final_members.get_value("Atlanta","lng")

atl_serie= []
for college in inst.iterrows():
    atl_serie.append(college[1])
    
atl_distances = []

for uni in atl_serie:
    ls = uni.tolist()
    atl_distances.append(vc(ls[:2],ls[2:]).miles)

inst.drop(["lat2","lng2"], axis = 1,inplace = True)    
atl_mean_distance =  sum(atl_distances)/len(atl_distances)

#WASHINGTON

inst["lat2"] = final_members.get_value("Washington","lat")[0]
inst["lng2"] = final_members.get_value("Washington","lng")[0]

wst_serie= []
for college in inst.iterrows():
    wst_serie.append(college[1])
    
wst_distances = []

for uni in wst_serie:
    ls = uni.tolist()
    wst_distances.append(vc(ls[:2],ls[2:]).miles)

inst.drop(["lat2","lng2"], axis = 1,inplace = True)    
wst_mean_distance =  sum(wst_distances)/len(wst_distances)

#KANSAS CITY

inst["lat2"] = final_members.get_value("Kansas City","lat")
inst["lng2"] = final_members.get_value("Kansas City","lng")

ksc_serie= []
for college in inst.iterrows():
    ksc_serie.append(college[1])
    
ksc_distances = []

for uni in ksc_serie:
    ls = uni.tolist()
    ksc_distances.append(vc(ls[:2],ls[2:]).miles)

inst.drop(["lat2","lng2"], axis = 1,inplace = True)    
ksc_mean_distance =  sum(ksc_distances)/len(ksc_distances)

#LOS ANGELES
inst["lat2"] = final_members.get_value("Los Angeles","lat")
inst["lng2"] = final_members.get_value("Los Angeles","lng")

la_serie= []
for college in inst.iterrows():
    la_serie.append(college[1])
    
la_distances = []

for uni in la_serie:
    ls = uni.tolist()
    la_distances.append(vc(ls[:2],ls[2:]).miles)

inst.drop(["lat2","lng2"], axis = 1,inplace = True)    
la_mean_distance =  sum(la_distances)/len(la_distances)

#ORLANDO
inst["lat2"] = final_members.get_value("Orlando","lat")
inst["lng2"] = final_members.get_value("Orlando","lng")

ord_serie= []
for college in inst.iterrows():
    ord_serie.append(college[1])
    
ord_distances = []

for uni in ord_serie:
    ls = uni.tolist()
    ord_distances.append(vc(ls[:2],ls[2:]).miles)

inst.drop(["lat2","lng2"], axis = 1,inplace = True)    
ord_mean_distance =  sum(ord_distances)/len(ord_distances)

#PHILLY
inst["lat2"] = final_members.get_value("Philadelphia","lat")
inst["lng2"] = final_members.get_value("Philadelphia","lng")

philadelphia_serie= []
for college in inst.iterrows():
    philadelphia_serie.append(college[1])
    
philadelphia_distances = []

for uni in philadelphia_serie:
    ls = uni.tolist()
    philadelphia_distances.append(vc(ls[:2],ls[2:]).miles)

inst.drop(["lat2","lng2"], axis = 1,inplace = True)    
philadelphia_mean_distance =  sum(philadelphia_distances)/len(philadelphia_distances)

#phoenix


inst["lat2"] = final_members.get_value("Phoenix","lat")
inst["lng2"] = final_members.get_value("Phoenix","lng")

phoenix_serie= []
for college in inst.iterrows():
    phoenix_serie.append(college[1])
    
phoenix_distances = []

for uni in phoenix_serie:
    ls = uni.tolist()
    phoenix_distances.append(vc(ls[:2],ls[2:]).miles)

inst.drop(["lat2","lng2"], axis = 1,inplace = True)    
phoenix_mean_distance =  sum(phoenix_distances)/len(phoenix_distances)

#SAN ANTONIO
inst["lat2"] = final_members.get_value("San Antonio","lat")[0]
inst["lng2"] = final_members.get_value("San Antonio","lng")[0]


sa_serie= []
for college in inst.iterrows():
    sa_serie.append(college[1])
    
sa_distances = []

for uni in sa_serie:
    ls = uni.tolist()
    sa_distances.append(vc(ls[:2],ls[2:]).miles)

inst.drop(["lat2","lng2"], axis = 1,inplace = True)    
sa_mean_distance =  sum(sa_distances)/len(sa_distances)

#SAN FRANCISCO

inst["lat2"] = final_members.get_value("San Francisco","lat")
inst["lng2"] = final_members.get_value("San Francisco","lng")


sf_serie= []
for college in inst.iterrows():
    sf_serie.append(college[1])
    
sf_distances = []

for uni in sf_serie:
    ls = uni.tolist()
    sf_distances.append(vc(ls[:2],ls[2:]).miles)
    

inst.drop(["lat2","lng2"], axis = 1,inplace = True)    
sf_mean_distance =  sum(sf_distances)/len(sf_distances)

#SALT LAKE CITY

inst["lat2"] = final_members.get_value("Salt Lake City","lat")
inst["lng2"] = final_members.get_value("Salt Lake City","lng")


slc_serie= []
for college in inst.iterrows():
    slc_serie.append(college[1])
    
slc_distances = []

for uni in slc_serie:
    ls = uni.tolist()
    slc_distances.append(vc(ls[:2],ls[2:]).miles)

inst.drop(["lat2","lng2"], axis = 1,inplace = True)    
slc_mean_distance =  sum(slc_distances)/len(slc_distances)

#SEATTLE
inst["lat2"] = cities.get_value("Seattle","lat")
inst["lng2"] = cities.get_value("Seattle","lng")


stl_serie= []
for college in inst.iterrows():
    stl_serie.append(college[1])
    
stl_distances = []

for uni in stl_serie:
    ls = uni.tolist()
    stl_distances.append(vc(ls[:2],ls[2:]).miles)

inst.drop(["lat2","lng2"], axis = 1,inplace = True)    
stl_mean_distance =  sum(stl_distances)/len(stl_distances)

inst.drop(["LATITUDE","LONGITUDE"],axis = 1, inplace = True)

inst["distToChicago"] = distances  
inst["disttoSaltLake"] = slc_distances
inst["disttoSanFrancisco"] = sf_distances
inst["sttoSanAntonio"] = sa_distances
inst["disttoWashinton"] = wst_distances
inst["disttoOrlando"] = ord_distances
inst["disttoPhilly"] = philadelphia_distances
inst["distoDenver"] = denver_distances
inst["disttoWashington"] = wst_distances
inst["disttoKansasCity"] = ksc_distances
inst["disttoLosAngeles"] = la_distances
inst["disttoPhoenix"] = phoenix_distances
inst["disttoSeattle"] = stl_distances



final_members["average_distance"] = 0
final_members.set_value("Atlanta","average_distance",atl_mean_distance)
final_members.set_value("Boston","average_distance",boston_mean_distance)
final_members.set_value("Chicago","average_distance",chicago_mean_distance)
final_members.set_value("Denver","average_distance",denver_mean_distance)
final_members.set_value("Kansas City","average_distance",ksc_mean_distance)
final_members.set_value("Los Angeles","average_distance",la_mean_distance)
final_members.set_value("New Orleans","average_distance",new_orleans_mean_distance)
final_members.set_value("Orlando","average_distance",ord_mean_distance)
final_members.set_value("Philadelphia","average_distance",philadelphia_mean_distance)
final_members.set_value("Phoenix","average_distance",phoenix_mean_distance)
final_members.set_value("Salt Lake City","average_distance",slc_mean_distance)
final_members.set_value("San Antonio","average_distance",sa_mean_distance)
final_members.set_value("San Francisco","average_distance",sf_mean_distance) 
final_members.set_value("Washington","average_distance",wst_mean_distance)

final_members.drop(["lat","lng","Year"],axis =1, inplace = True)

final_members.to_csv("final_members.csv")

#inst.to_csv("distances.csv")


# The machine learning 
predicting = final_members[['adjusted fee', 
                            'average_distance','Desirability', 'Attendance']]
selection = rnd.binomial(1, 0.7, size=len(predicting)).astype(bool)
training = predicting[selection]
testing = predicting[~selection]
rfc = RandomForestClassifier()
rfc.fit(training[['adjusted fee', 'average_distance','Desirability']], training['Attendance'])
predicted = rfc.predict(testing[['adjusted fee',  'average_distance','Desirability']])

train_err = training['Attendance'] ^ rfc.predict(training[['adjusted fee', 'average_distance','Desirability']])
test_err = testing['Attendance'] ^ rfc.predict(testing[['adjusted fee', 'average_distance', 'Desirability']])
train_acc = sum(train_err) / len(train_err)
test_acc = sum(test_err) / len(test_err) 
