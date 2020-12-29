names = ['Bulbasaur','Charmander','Squirtle']
hps = [45, 39, 44]

combined = []

for i,j in enumerate(names):
    combined.append( (j,hps[i]) )

print(combined)


combined_zip = zip(names,hps)

combined_zip_list = [ * combined_zip ]