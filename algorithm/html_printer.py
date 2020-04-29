
def __write_pre(html, known_features):
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
    headers = ["feature number", "baseline fs", "semi-supervised fs"]
    html.write("<body>")
    html.write("<div>")
    html.write("<div>Known important features:</div>")
    html.write("<ul>")
    for i in known_features:
        html.write("<li>" + str(i + 1) + "</li>")
    html.write("</ul>")
    html.write("</div>")
    html.write("<table class = \"mainTable\">")
    html.write("<tr class=\"tableHeader\">")
    for i in headers:
        html.write("<th class = \"headerElement\">" + i + "</th>")
    html.write("</tr>")


def __write_row(html, sorted_stacked):
    html.write("<tr class = \"tableRow\">")
    html.write("<td class = \"tableHeader\">")
    html.write("<ul class = \"topList\">")
    for i in sorted_stacked[:, 0]:
        html.write("<li class = \"topElement\">" + str(int(i)) + "</li>")
    html.write("</ul>")
    html.write("</td>")
    html.write("<td class = \"tableHeader\">")
    html.write("<ul class = \"topList\">")
    for i in sorted_stacked[:, 1]:
        html.write("<li class = \"topElement\">" + str(i) + "</li>")
    html.write("</ul>")
    html.write("</td>")
    html.write("<td class = \"tableHeader\">")
    html.write("<ul class = \"topList\">")
    for i in sorted_stacked[:, 2]:
        html.write("<li class = \"topElement\">" + str(i) + "</li>")
    html.write("</ul>")
    html.write("</td>")
    html.write("</tr>")
    

def __write_post(html):
    html.write("</table>")
    html.write("</body>")
    html.write("</html>")       

def html_print(html, sorted_stacked_list, known_features):
	__write_pre(html, known_features)
	[__write_row(html, sorted_stacked) for sorted_stacked in sorted_stacked_list]
	__write_post(html)

