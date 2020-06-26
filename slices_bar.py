import matplotlib.pyplot as plt
import numpy as np
import re

def collect_number_times(slice_):
    slice_split = re.findall('<td.+?>.+?</td>', slice_)
    size = int(re.search('\d+', slice_split[0])[0])
    lines = re.findall('<li.+?>.+?</li>', slice_split[-2])
    feat_number = []
    feat_times = []
    for line in lines:
        feat = re.search(r'\d+\(\d+\)', line)[0]
        number = int(feat.split('(')[0])
        times = int(feat.split('(')[1].strip(')'))
        feat_number.append(number)
        feat_times.append(times)
    return size, feat_number, feat_times

labels_list = []
times_list = []
size_list = []

for number in range(1, 7):
    with open(str(number) + 'TablesPlots/HtmlTable.html', 'r') as hd:
        html_raw = hd.read()
        html_lines = re.findall('<tr.+?>.+?</tr>', html_raw)
        html_lines.pop(0)
        first_slice = html_lines[0]
        last_slice = html_lines[-1]
        first_size, first_numbers, first_times = collect_number_times(first_slice)
        last_size, last_numbers, last_times = collect_number_times(last_slice)
        
        to_delete = list(set(first_numbers).difference(set(last_numbers)))

        # print(sorted(first_numbers))
        # print(sorted(to_delete))
        for num in to_delete:
            # print('first', num)
            index = first_numbers.index(num)
            first_numbers.pop(index)
            first_times.pop(index)

        to_delete = list(set(last_numbers).difference(set(first_numbers)))
        # print(sorted(last_numbers))
        for num in to_delete:    
            # print('last', num)
            index = last_numbers.index(num)
            last_numbers.pop(index)
            last_times.pop(index)
        
        # print(len(first_numbers))
        # print(len(last_numbers))
        labels = first_numbers

        labels_list.append(labels)
        times_list.append((first_times, last_times))
        size_list.append((first_size, last_size))

        x = np.arange(len(labels))  # the label locations
        width = 0.25  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width, first_times, width, label='Срез размера ' + str(first_size))
        rects2 = ax.bar(x + width, last_times, width, label='Срез размера ' + str(last_size))

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Частота отбора')
        ax.set_title('Номер признака')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()


        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')


        autolabel(rects1)
        autolabel(rects2)

        fig.tight_layout()

        plt.savefig(str(number) + 'TablesPlots/frequency.png')
        plt.close()

fig, axes = plt.subplots(6, figsize=(12, 10), dpi=300)
for i, ax in enumerate(axes.flatten()):
    labels = labels_list[i]
    first_times, last_times = times_list[i]
    first_size, last_size = size_list[i]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    
    rects1 = ax.bar(x - width/2, first_times, width, label='Срез размера ' + str(first_size))
    rects2 = ax.bar(x + width/2, last_times, width, label='Срез размера ' + str(last_size))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Частота отбора')
    ax.set_xlabel('Индексы признаков в исходном наборе данных')
    ax.set_title('Набор данных номер ' + str(i))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 25, 50])
    # ax.set_yticklabels([0, 25, 50])
    ax.legend()


    # def autolabel(rects):
    #     """Attach a text label above each bar in *rects*, displaying its height."""
    #     for rect in rects:
    #         height = rect.get_height()
    #         ax.annotate('{}'.format(height),
    #                     xy=(rect.get_x() + rect.get_width() / 2, height),
    #                     xytext=(0, 3),  # 3 points vertical offset
    #                     textcoords="offset points",
    #                     ha='center', va='bottom')


    # autolabel(rects1)
    # autolabel(rects2)

fig.tight_layout()

plt.savefig('allfrequencies.png')
plt.close()
