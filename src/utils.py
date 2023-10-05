import os
import pandas as pd


def print_diseases():
    plant_diseases = os.listdir("data/train")
    plants = {plant.split('___')[0] for plant in plant_diseases}
    diseases = {plant.split('___')[1] for plant in plant_diseases if plant.split('___')[1] != 'healthy'}

    print("Unique plants: \n{}\nTotal number: {}".format(plants, len(plants)))
    print("Unique diseases: \n{}\nTotal number: {}".format(diseases, len(diseases)))


def print_data_frame():
    path = "data/train"
    plant_diseases = os.listdir(path)
    data = []
    for disease in plant_diseases:
        num_images = len(os.listdir(os.path.join(path, disease)))
        data.append({"Disease": disease, "No. of Images": num_images})

    df = pd.DataFrame(data)
    print(df)
