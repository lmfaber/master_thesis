import seaborn as seaborn




import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set(style='whitegrid', font_scale=2,rc={'figure.figsize':(11.7,8.27)})

data = pd.read_csv('timeline.csv', header=1)

data = data[data['Year'].between(2008,2020, inclusive=False)]
# print(sns.load_dataset('titanic'))
x = data['Year']
y = data['No. of publications']

plt.xticks(rotation=45, ha='center')		# Rotate the xaxis at 45 degree

sns.barplot(y=y, x=x, color='tab:blue')
plt.tight_layout()


plt.savefig('rna_seq.svg', format='svg')
