import pandas as pd
from matplotlib import pyplot as plt

import data_preparation

df = data_preparation.final_df

'''The following diagrams visualize the relations between the target variable and the other variables.'''

# Stroke analyse preparation.
stroke_married = df[['ever_married', 'stroke']].groupby(df['ever_married']).mean().sort_values(by='stroke',
                                                                                               ascending=False)
stroke_work = df[['work_type', 'stroke']].groupby(df['work_type']).mean().sort_values(by='stroke', ascending=False)
stroke_age = df[['age', 'stroke']].groupby(pd.cut(df['age'], [i for i in range(0, 110, 10)],
                                                  right=False)).mean().sort_values(by='stroke', ascending=False)

stroke_residence = df[['Residence_type', 'stroke']].groupby(df['Residence_type']).mean().sort_values(by='stroke',
                                                                                                     ascending=False)
stroke_avg = df[['avg_glucose_level', 'stroke']].groupby(
    pd.cut(df['avg_glucose_level'], [i for i in range(0, 280, 20)], right=False)).mean().sort_values(by='stroke',
                                                                                                     ascending=False)
stroke_bmi = df[['bmi', 'stroke']].groupby(
    pd.cut(df['bmi'], [i for i in range(0, 110, 10)], right=False)).mean().sort_values(
    by='stroke', ascending=False)

stroke_smoke = df[['smoking_status', 'stroke']].groupby(df['smoking_status']).mean().sort_values(by='stroke',
                                                                                                 ascending=False)

# Stroke probability for non-/married.
fig, ax = plt.subplots()
ax.grid(False)
ax.bar(0.25, stroke_married['stroke'][0], 0.5, label='Yes')
ax.bar(0.75, stroke_married['stroke'][1], 0.5, label='No')
ax.text(x=0.25, y=stroke_married['stroke'][0] / 2, s=round(stroke_married['stroke'][0], 2), color='white', ha="center")
ax.text(x=0.75, y=stroke_married['stroke'][1] / 2, s=round(stroke_married['stroke'][1], 2), color='white', ha="center")
ax.set_ylabel('stroke probability')
ax.set_title('stroke probability for non-/married')
ax.set_xticks([0.25, 0.75])
ax.set_xticklabels(['Yes', 'No'])
ax.legend()
fig.tight_layout()
#plt.savefig('stroke_married.png')
plt.show()
plt.close()

# Stroke probability per work_type.
fig, ax = plt.subplots()
ax.grid(False)
indezes = stroke_work.index.tolist()
ax.set_xticks([0.10, 0.30, 0.50, 0.70])
ax.bar(0.10, stroke_work['stroke'][0], 0.2)
ax.bar(0.30, stroke_work['stroke'][1], 0.2)
ax.bar(0.50, stroke_work['stroke'][2], 0.2)
ax.bar(0.70, stroke_work['stroke'][3], 0.2)
ax.text(x=0.10, y=stroke_work['stroke'][0] / 2, s=round(stroke_work['stroke'][0], 2), color='white', ha="center")
ax.text(x=0.30, y=stroke_work['stroke'][1] / 2, s=round(stroke_work['stroke'][1], 2), color='white', ha="center")
ax.text(x=0.50, y=stroke_work['stroke'][2] / 2, s=round(stroke_work['stroke'][2], 2), color='white', ha="center")
ax.text(x=0.70, y=stroke_work['stroke'][3], s=round(stroke_work['stroke'][3], 5), color='black', ha="center")
ax.set_xticklabels(indezes[:-1])
ax.set_ylabel('stroke work_type')
ax.set_title('stroke probability per work_type')
plt.show()
plt.close()

# Stroke probability of ages.
fig, ax = plt.subplots()
ax.plot([
    stroke_age['stroke'][8],
    stroke_age['stroke'][7],
    stroke_age['stroke'][6],
    stroke_age['stroke'][5],
    stroke_age['stroke'][4],
    stroke_age['stroke'][3],
    stroke_age['stroke'][2],
    stroke_age['stroke'][1],
    stroke_age['stroke'][0]],
    linestyle='-')
ax.set_xticks(range(9))
ax.set_xticklabels(['10', '20', '30', '40', '50', '60', '70', '80', '90'])
plt.title("stroke probability of ages")
plt.ylabel('stroke probability')
plt.show()
plt.close()

# Stroke probability per Residence_type.
fig, ax = plt.subplots()
ax.grid(False)
ax.bar(0.25, stroke_residence['stroke'][0], 0.5, label='Urban')
ax.bar(0.75, stroke_residence['stroke'][1], 0.5, label='Rural')
ax.text(x=0.25, y=stroke_residence['stroke'][0] / 2, s=round(stroke_residence['stroke'][0], 3), color='white',
        ha="center")
ax.text(x=0.75, y=stroke_residence['stroke'][1] / 2, s=round(stroke_residence['stroke'][1], 3), color='white',
        ha="center")
ax.set_ylabel('stroke probability')
ax.set_title('stroke probability per Residence_type')
ax.set_xticks([0.25, 0.75])
ax.set_xticklabels(['Urban', 'Rural'])
ax.legend()
fig.tight_layout()
plt.show()
plt.close()

# Stroke probability per avg_glucose_level.
fig, ax = plt.subplots()
ax.plot([
    stroke_avg['stroke'][7],
    stroke_avg['stroke'][5],
    stroke_avg['stroke'][9],
    stroke_avg['stroke'][6],
    stroke_avg['stroke'][8],
    stroke_avg['stroke'][10],
    stroke_avg['stroke'][4],
    stroke_avg['stroke'][1],
    stroke_avg['stroke'][3],
    stroke_avg['stroke'][2],
    stroke_avg['stroke'][0]],
    linestyle='-')
ax.set_xticks(range(11))
ax.set_xticklabels(['60', '80', '100', '120', '140', '160', '180', '200', '220', '240', '260'])
plt.title("stroke probability per avg_glucose_level")
plt.ylabel('stroke probability')
plt.show()
plt.close()

# Stroke probability per bmi.
fig, ax = plt.subplots()
ax.plot([
    stroke_bmi['stroke'][4],
    stroke_bmi['stroke'][2],
    stroke_bmi['stroke'][1],
    stroke_bmi['stroke'][0],
    stroke_bmi['stroke'][3],
    stroke_bmi['stroke'][5],
    stroke_bmi['stroke'][6]],
    linestyle='-')
ax.set_xticks(range(7))
ax.set_xticklabels(['20', '30', '40', '50', '60', '70', '80'])
plt.title("stroke probability per bmi")
plt.ylabel('stroke probability')
plt.show()
plt.close()