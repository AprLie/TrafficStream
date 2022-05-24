import pickle as pk 

for year in range(2011, 2018):
    need_station = pk.load(open(str(year)+'.pkl', 'rb'))
    maps = {v['now'] :k for k,v in need_station.items()}
    print(maps)

