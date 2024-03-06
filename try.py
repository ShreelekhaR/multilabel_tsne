import numpy as np
from sklearn.manifold import TSNE
from random import randint
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import base64
import os
import ipywidgets as widgets

# Encode images to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

testing = 'graft_sentinel'
# Load your image features data
image_features = np.load('sentinel_graft_feats.npy', allow_pickle=True)

# Load your multilabel data
ground_truth_labels = np.load('sentinel_100_test_labels.npy', allow_pickle=True)


# class names
classes = ['tennis', 'skate', 'amfootball', 'swimming', 'cemetery', 'garage', 'golf', 'roundabout', 'parkinglot', 'supermarket',
                'school', 'marina', 'baseball', 'fall', 'pond', 'airport', 'beach', 'bridge', 'religious', 'residential', 'warehouse',
                'office', 'farmland', 'university', 'forest', 'lake', 'naturereserve', 'park', 'sand', 'soccer', 'equestrian', 'shooting', 
                'icerink', 'commercialarea', 'garden', 'dam', 'railroad', 'highway', 'river', 'wetland']

class_indices = [i for i in range(len(classes))]


print(image_features.shape)
print(len(ground_truth_labels))

# multi-label tsne

# Create a list of labels for each image
label_list = []
for i in range(0, len(ground_truth_labels)):
    local = str([classes[x] for x in ground_truth_labels[i]])
    label_list.append(local)

# Calculate the t-SNE of the image features
perplexity = 70
n_iter = 10000
init = 'random'
tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity, n_iter=n_iter, init=init)
image_features_tsne = tsne.fit_transform(image_features)

# plot the t-SNE
plt.figure(figsize=(16, 16))
plt.scatter(image_features_tsne[:,0], image_features_tsne[:,1])
plt.savefig('outputs/KD_tsne.png')

# # Create an interactive plot of the t-SNE so we can hover over the points and see the labels

# import libraries for interactive plot
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly


colors = {}
for i in range(len(classes)):
    # create a color for each class hex
    color = "#{:06x}".format(randint(0, 0xFFFFFF))
    colors[classes[i]] = color

# read image paths from img_paths.txt
img_paths = []
with open('img_paths1.txt', 'r') as f:
    img_paths = f.readlines()
    img_paths = [x.strip() for x in img_paths]

# Encode images to base64
encoded_images = [image_to_base64(img_path) for img_path in img_paths]

df = pd.DataFrame({'x': image_features_tsne[:,0], 'y': image_features_tsne[:,1], 'label': label_list, 'img_path': img_paths})
# save the dataframe to a csv file
df.to_csv('KD_tsne.csv', index=False)
exit()

fig = go.Figure()


# # plot the t-SNE and use label list to show the labels when hovering over the points
fig = px.scatter(x=image_features_tsne[:,0], y=image_features_tsne[:,1], hover_name=label_list, labels={'hover_name':'Labels'}, title='t-SNE of Sentinel Images using KD Features',width=1200, height=800)

# add a dropdown to shade the points by class when class is selected
# prepend All to the list of classes
classes = ['All'] + classes

def highlight_class(class_name):
    if class_name == 'All':
        return ['#000000'] * len(label_list)
    result = []

    for labels in label_list:
        if class_name in labels:
            result.append(colors[class_name])
        else:
            # add black for points that are not in the class in hex
            result.append('#000000')
        
    return result

fig.update_layout(
    updatemenus=[
        {
            "buttons": [
                {
                    "label": class_name,
                    "method": "update",
                    "args": [{"marker.color": [highlight_class(class_name)]}],

                }
                for class_name in classes
            ]
        }
    ],
    margin=dict(l=0, r=0, b=0, t=25),

)




# offline plot to html
# plotly.offline.plot(fig, filename='trial.html', auto_open=True)
pio.write_html(fig, file='trial.html', auto_open=True)

