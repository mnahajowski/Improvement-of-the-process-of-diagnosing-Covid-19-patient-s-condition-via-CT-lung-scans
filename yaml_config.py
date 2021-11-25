import yaml
import io

model_info = {
    'CLASSES': ['COVID-19', 'Non-COVID-19'],
    'IMAGE_SIZE': [224, 224],
    'input_shape': [224, 224, 3],
    'custom_covid_data_path': 'COVID-19-second/COVID-19/',
    'custom_non_covid_data_path': 'COVID-19-second/Non-COVID-19/'
}
with io.open('config.yaml', 'w', encoding='utf8') as outfile:
    yaml.dump(model_info, outfile, default_flow_style=False, allow_unicode=True)
