
import warnings
import pandas as pd

# Ignore warnings.
warnings.filterwarnings('ignore')

# Read data into dataframe.
df = pd.read_csv(filepath_or_buffer='data.csv', header=0)

'''################# DATA PREPARATION ##################'''

# Drop id column because it is not necessary.
df = df.drop(columns=['id'], axis=1)

# Delete empty values.
df = df.dropna(axis=0, how='any')
final_df = df

# Print duplicates.
ids = df.duplicated()
for id in ids:
    if id:
        print('Duplicates detected!')
        print('Exit program...')
        exit()

'''There should not be a baby that smokes.'''
if not df[(df['age'] == 0) & (df['smoking_status'] == 'smokes')].empty:
    print("Abort program!")
    exit()

'''##################-START-MIN AND MAX VALUES-###########################'''

# Age values.
print('Max age: %d' % (df['age'].max()))
print('Min age: %d' % (df['age'].min()))
print('Mean age: %d' % (df['age'].mean()))
print('Median age: %d' % (df['age'].median()))

# BMI values.
print('Max bmi: %d' % (df['bmi'].max()))
print('Min bmi: %d' % (df['bmi'].min()))
print('Mean bmi: %d' % (df['bmi'].mean()))
print('Median bmi: %d' % (df['bmi'].median()))

# AVG glucose values.
print('Max avg_glucose_level: %d' % (df['avg_glucose_level'].max()))
print('Min avg_glucose_level: %d' % (df['avg_glucose_level'].min()))

'''##################-END-MIN AND MAX VALUES-###########################'''
