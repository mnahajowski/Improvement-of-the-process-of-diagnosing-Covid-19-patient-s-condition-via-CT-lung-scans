import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from get_config import config


CLASSES = config.content['CLASSES']
NUM_CLASSES = len(CLASSES)


def characterize_data(data_files):
    count_covid = len([filename for filename in data_files if '\\COVID-19\\' in filename])
    print("COVID-19 images count : " + str(count_covid))

    count_non_covid = len([filename for filename in data_files if "\\Non-COVID-19\\" in filename])
    print("Non-COVID images count : " + str(count_non_covid))

    data = {'Cases': ['COVID-19', 'non-COVID'],
            'Cases_count': [count_covid, count_non_covid]
            }

    draw_data_characteristics(data=data)
    return {CLASSES[0]: count_covid, CLASSES[1]: count_non_covid}


def draw_data_characteristics(data):
    df = pd.DataFrame(data)
    sns.set(style="darkgrid")
    plt.figure(figsize=(10,8))
    sns.barplot(x=df.index, y= df['Cases_count'].values)
    plt.title('Number of All the Data', fontsize=14)
    plt.xlabel('Case type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(range(len(df.index)), ['COVID-19', 'non-COVID'])
    plt.show()

    print(df)
