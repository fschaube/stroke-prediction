import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import data_preparation

df = data_preparation.final_df

'''The following diagrams visualize the distribution of the variables.'''

# Plot distribution of genders.
ax = sns.histplot(data=df, x='gender', hue='gender')
ax.grid(False)
result = df.groupby(['gender']).count()
ax.text(x=1, y=2930, s=result['age'][0], color='black', ha="center")
ax.text(x=0, y=2010, s=result['age'][1], color='black', ha="center")
ax.text(x=2, y=30, s=result['age'][2], color='black', ha="center")
plt.title('Genders')
plt.show()
plt.close()

# Delete 'Other' from genders.
df = df[df['gender'] != 'Other']

# Plot distribution of ages.
range_ages = [i for i in range(0, 110, 10)]
result = df.groupby(pd.cut(df['age'], range_ages, right=False)).count()
ax = sns.histplot(data=df, x='age', color='orange', bins=range_ages)
ax.grid(False)
plt.xticks(range_ages)
i = 5
for k, v in result['age'].iteritems():
    ax.text(x=i, y=v + 10, s=v, color='black', ha="center")
    i += 10
plt.title('Ages')
plt.show()
plt.close()

# Plot distribution of hypertension.
ax = sns.histplot(data=df, x='hypertension', hue='hypertension', bins=2)
ax.grid(False)
ax.set_xticks([0.25, 0.75])
ax.set_xticklabels([0, 1])
result_tension = df.groupby(['hypertension']).count()
ax.text(x=0.25, y=result_tension['gender'][0] + 40, s=result_tension['gender'][0], color='black', ha="center")
ax.text(x=0.75, y=result_tension['gender'][1] + 40, s=result_tension['gender'][1], color='black', ha="center")
plt.title('Hypertension')
plt.show()
plt.close()

# Plot distribution of heart_disease.
ax = sns.histplot(data=df, x='heart_disease', hue='heart_disease', bins=2)
ax.grid(False)
ax.set_xticks([0.25, 0.75])
result_disease = df.groupby(df['heart_disease']).count()
ax.text(x=0.25, y=result_disease['gender'][0] + 40, s=result_disease['gender'][0], color='black', ha="center")
ax.text(x=0.75, y=result_disease['gender'][1] + 40, s=result_disease['gender'][1], color='black', ha="center")
plt.title('heart disease')
ax.set_xticklabels([0, 1])
plt.show()
plt.close()

# Plot distribution of ever_married.
ax = sns.histplot(data=df, x='ever_married', hue='ever_married')
ax.grid(False)
result = df.groupby(df['ever_married']).count()
ax.text(x='Yes', y=result['gender'][1] + 40, s=result['gender'][1], color='black', ha="center")
ax.text(x='No', y=result['gender'][0] + 40, s=result['gender'][0], color='black', ha="center")
plt.title('ever married')
plt.show()
plt.close()

# Plot distribution of work_type.
ax = sns.histplot(data=df, x='work_type', hue='work_type')
ax.grid(False)
result = df.groupby(df['work_type']).count()
result = result.sort_values(by=['gender'], ascending=False)
for k, v in result['gender'].iteritems():
    ax.text(x=k, y=v + 35, s=v, color='black', ha="center")
plt.title('work type')
plt.show()
plt.close()

# Plot distribution of Residence_type.
ax = sns.histplot(data=df, x='Residence_type', hue='Residence_type', legend=False)
ax.grid(False)
result = df.groupby(df['Residence_type']).count().sort_values(by=['gender'], ascending=False)
for k, v in result['gender'].iteritems():
    ax.text(x=k, y=v + 25, s=v, color='black', ha="center")
plt.title('Residence_type')
plt.show()
plt.close()

# Plot distribution of avg_glucose_level.
range_glucose = [i for i in range(0, 280, 20)]
result = df.groupby(pd.cut(df['avg_glucose_level'], range_glucose, right=False)).count()
ax = sns.histplot(data=df, x='avg_glucose_level', color='yellow', bins=range_glucose)
ax.grid(False)
plt.xticks(range_glucose)
i = 10
for k, v in result['avg_glucose_level'].iteritems():
    ax.text(x=i, y=v + 10, s=v, color='black', ha="center")
    i += 20
plt.title('avg glucose level')
plt.show()
plt.close()

# Plot distribution of bmi.
range_bmi = [i for i in range(0, 110, 10)]
result = df.groupby(pd.cut(df['bmi'], range_bmi, right=False)).count()
ax = sns.histplot(data=df, x='bmi', color='orange', bins=range_bmi)
ax.grid(False)
plt.xticks(range_bmi)
i = 4
for k, v in result['bmi'].iteritems():
    ax.text(x=i, y=v + 10, s=v, color='black', ha="center")
    i += 10
plt.title('bmi')
plt.show()
plt.close()

# Plot distribution of smoking_status.
ax = sns.histplot(data=df, x='smoking_status', hue='smoking_status', legend=False)
ax.grid(False)
result = df.groupby(df['smoking_status']).count().sort_index(key=lambda x: x.str.lower())
for k, v in result['gender'].iteritems():
    ax.text(x=k, y=v + 15, s=v, color='black', ha="center")
plt.title('smoking status')
plt.show()
plt.close()

# Plot distribution of stroke.
ax = sns.histplot(data=df, x='stroke', hue='stroke', bins=2)
ax.grid(False)
ax.set_xticks([0.25, 0.75])
resultStroke = df.groupby(df['stroke']).count()
ax.text(x=0.25, y=resultStroke['gender'][0] + 40, s=resultStroke['gender'][0], color='black', ha="center")
ax.text(x=0.75, y=resultStroke['gender'][1] + 40, s=resultStroke['gender'][1], color='black', ha="center")
plt.title('stroke')
ax.set_xticklabels([0.0, 1.0])
plt.show()
plt.close()

# correlation matrix.
ax = sns.heatmap(df.corr(method='pearson'), annot=True)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()
plt.close()
