import numpy as np
import os
from utils import read_subsamples
# from build_plots import run_build_plots
from build_model import run_build_model
from algorithm import run_greedy
from algorithm import compare_with
from build_model import dump_bad_subsamples
from sklearn.manifold import TSNE
from matplotlib.ticker import NullFormatter
import matplotlib.pyplot as plt



def __write_pre(html, known_features, good_features):
    html.write("<!DOCTYPE html>")
    html.write("<head>")
    html.write("<meta charset=\"UTF-8\">")
    html.write("<title>Title</title>")
    html.write("<style>")
    html.write(".tableHeader, .headerElement { padding: 3px; border: 1px solid black;}")
    html.write(".mainTable { border-collapse: collapse; width: 900px; }")
    html.write(".topElement { list-style-type: none; }")
    html.write("</style>")
    html.write("</head>")
    headers = ["dataset", "baseline features", "semi-supervised features", "baseline score", "semi-supervised score"]
    html.write("<body>")
    html.write("<div>")
    html.write("<div>Known important features:</div>")
    html.write("<ul>")
    for i in known_features:
        html.write("<li>" + str(i + 1) + "</li>")
    html.write("</ul>")
    html.write("</div>")
    html.write("<div>")
    html.write("<div>Good important features:</div>")
    html.write("<ul>")
    for i in good_features:
        html.write("<li>" + str(i + 1) + "</li>")
    html.write("</ul>")
    html.write("</div>")
    html.write("<table class = \"mainTable\">")
    html.write("<tr class=\"tableHeader\">")
    for i in headers:
        html.write("<th class = \"headerElement\">" + i + "</th>")
    html.write("</tr>")


def __write_row(html, dataset, baseline_features, semi_features, accuracy_semi, fullness_semi, accuracy_baseline, fullness_baseline):
    html.write("<tr class = \"tableRow\">")
    html.write("<td class = \"tableHeader\">")
    html.write("<div>" + dataset + "</div>")
    html.write("</td>")
    html.write("<td class = \"tableHeader\">")
    html.write("<ul class = \"topList\">")
    for i in baseline_features:
        html.write("<li class = \"topElement\">" + str(i) + "</li>")
    html.write("</ul>")
    html.write("</td>")
    html.write("<td class = \"tableHeader\">")
    html.write("<ul class = \"topList\">")
    for i in semi_features:
        html.write("<li class = \"topElement\">" + str(i) + "</li>")
    html.write("</ul>")
    html.write("</td>")
    html.write("<td class = \"tableHeader\">")
    html.write("<ul class = \"topList\">")
    html.write("<li class = \"topElement\"> acc = " + str(accuracy_baseline) + "</li>")
    html.write("<li class = \"topElement\"> full = " + str(fullness_baseline) + "</li>")
    html.write("</ul>")
    html.write("</td>")
    html.write("<td class = \"tableHeader\">")
    html.write("<ul class = \"topList\">")
    html.write("<li class = \"topElement\"> acc = " + str(accuracy_semi) + "</li>")
    html.write("<li class = \"topElement\"> full = " + str(fullness_semi) + "</li>")
    html.write("</ul>")
    html.write("</td>")
    html.write("</tr>")
    

def __write_post(html):
    html.write("</table>")
    html.write("</body>")
    html.write("</html>")       



directory_name_datasets = '4TablesPlots/subsamples'
single_kernel = "sigmoid 7.19 0.82 auto"
directory_name = '4TablesPlots'
html = open(directory_name + '/comparison.html', 'w')
# known_features = np.array([934, 389, 677])
# good_features = np.array([130, 643, 323, 837, 510, 392, 462, 209, 727, 669, 93, 418, 804, 168, 939, 109, 174, 111, 880, 178, 754, 758, 502, 121, 190, 574, 767])
__write_pre(html, known_features, good_features)        
for file_name in os.listdir(directory_name_datasets): # open directory with datasets
    if '.csv' not in file_name: # skip datasets not in csv format
            continue
    with open(directory_name_datasets + '/' + file_name, 'r') as fd: # open each file 
        #run_build_plots(x, y) # build all the plots for datasets
        if int(file_name.strip('.csv')) > 10:
            continue
        print("started processing : " + file_name) # logging the start of building procedure
        x, y = read_subsamples(fd) 

        # 4set
        res = compare_with(x, y, known_features, good_features, single_kernel)
        baseline_features, semi_features, scores = res
        accuracy_semi, fullness_semi, accuracy_baseline, fullness_baseline = scores
        __write_row(html, file_name, baseline_features, semi_features, accuracy_semi, fullness_semi, accuracy_baseline, fullness_baseline)
        
        known_features -= 1
        good_features -= 1
        (fig, subplots) = plt.subplots(5, 1, figsize=(15, 8))
        perplexities = [3, 4, 5, 6, 7]
        other_features = [i for i in range(0, x.shape[1]) if i not in known_features and i not in good_features]
        
        x_t = x.T

        for i, perplexity in enumerate(perplexities):
            ax = subplots[i]

            tsne = TSNE(n_components=2, init='random',
                                 random_state=0, perplexity=perplexity)
            Y = tsne.fit_transform(x_t)
            ax.set_title("Perplexity=%d" % perplexity)
            ax.scatter(Y[other_features, 0], Y[other_features, 1], c="b")
            ax.scatter(Y[good_features, 0], Y[good_features, 1], c="r")
            ax.scatter(Y[known_features, 0], Y[known_features, 1], c="g")
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
            ax.axis('tight')
        plt.savefig('4TablesPlots/main' + file_name.strip('.csv') + 'features.png', padinches = 0.1)
        plt.close()
        baseline_features -= 1
        semi_features -= 1
        
        good_selected = [f for f in semi_features if f in good_features]
        good_not_selected = [f for f in good_features if f not in good_selected]
        bad_selected = [f for f in semi_features if f not in good_features]
        other_features = [i for i in range(0, x.shape[1]) if i not in known_features and i not in good_features and i not in bad_selected]
        
        (fig, subplots) = plt.subplots(5, 1, figsize=(15, 8))
        perplexities = [3, 4, 5, 6, 7]
        
        for i, perplexity in enumerate(perplexities):
            ax = subplots[i]

            tsne = TSNE(n_components=2, init='random', random_state=0, perplexity=perplexity)
            Y = tsne.fit_transform(x_t)
            ax.set_title("Perplexity=%d" % perplexity)
            ax.scatter(Y[other_features, 0], Y[other_features, 1], c="b")
            ax.scatter(Y[bad_selected, 0], Y[bad_selected, 1], c="pink")
            ax.scatter(Y[good_not_selected, 0], Y[good_not_selected, 1], c="yellow")
            ax.scatter(Y[good_selected, 0], Y[good_selected, 1], c="r")
            ax.scatter(Y[known_features, 0], Y[known_features, 1], c="g")
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
            ax.axis('tight')
        plt.savefig('4TablesPlots/filtered' + file_name.strip('.csv') + 'features.png', padinches = 0.1)
        plt.close()
        known_features += 1
        good_features += 1
        

        # 3set
        # known_features = np.array([65, 337, 339])
        # good_features = np.array([256, 260, 324, 6, 454, 200, 11, 206, 279, 153, 473, 411, 286, 287, 414, 222, 482, 227, 297, 494, 497, 50, 244, 56, 378, 443, 445])
        # 2set
        # known_features = np.array([41, 37, 73])
        # good_features = np.array([64, 71, 72, 74, 75, 76, 78, 15, 79, 80, 82, 36, 38, 39, 40, 42, 43, 44, 50, 53, 54, 55, 56, 58, 60, 62, 63])
        
        # run_greedy(x, y, known_features, good_features) # run and compare proposed algorithm with baseline

__write_post(html)
    # dump to html
html.close()

























