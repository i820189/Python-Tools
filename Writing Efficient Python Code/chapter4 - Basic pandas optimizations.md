# Intro to pandas DataFrame iteration

### Pandas Recap
- See pandas overview in Intermediate Python
- Library used for data analysis
- Main data structure is the DataFrame
  - Tabular data with labeled rows and columns
  - Built on top of the NumPy array structure
- Chapter Objective:
  - Best practice for iterating over a pandas DataFrame

---

### Baseball stats

```python
import pandas as pd

baseball_df = pd.read_csv('baseball_stats.csv')

print(baseball_df.head())

> Team League Year RS RA W G Playoffs
> 0 ARI NL 2012 734 688 81 162 0
> 1 ATL NL 2012 700 600 94 162 1
> 2 BAL AL 2012 712 705 93 162 1
> 3 BOS AL 2012 734 806 69 162 0
> 4 CHC NL 2012 613 759 61 162 0
```

---

### **Calculating win percentage**

```python
import numpy as np

def calc_win_perc(wins, games_played):
    win_perc = wins / games_played
    return np.round(win_perc,2)

win_perc = calc_win_perc(50, 100)

print(win_perc)

> 0.5
```

### **Adding win percentage to DataFrame**

```python
win_perc_list = []

for i in range(len(baseball_df)):
    row = baseball_df.iloc[i]
    wins = row['W']
    games_played = row['G']
    win_perc = calc_win_perc(wins, games_played)
    win_perc_list.append(win_perc)

baseball_df['WP'] = win_perc_list

print(baseball_df.head())

> Team League Year RS RA W G Playoffs WP
> 0 ARI NL 2012 734 688 81 162 0 0.50
> 1 ATL NL 2012 700 600 94 162 1 0.58
> 2 BAL AL 2012 712 705 93 162 1 0.57
> 3 BOS AL 2012 734 806 69 162 0 0.43
> 4 CHC NL 2012 613 759 61 162 0 0.38
```