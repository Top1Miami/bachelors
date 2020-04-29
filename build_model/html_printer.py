

def __write_pre(html):
    html.write("<!DOCTYPE html>")
    html.write("<head>")
    html.write("<meta charset=\"UTF-8\">")
    html.write("<title>Title</title>")
    html.write("<style>")
    html.write(".tableHeader, .headerElement { padding: 3px; border: 1px solid black; }")
    html.write(".mainTable { border-collapse: collapse; width: 900px; }")
    html.write(".topElement { list-style-type: none; }")
    html.write("</style>")
    html.write("</head>")
    headers = ["slice size", "top3", "top10", "top30", "confidence"]
    html.write("<body>")
    html.write("<table class = \"mainTable\">")
    html.write("<tr class=\"tableHeader\">")
    for i in headers:
        html.write("<th class = \"headerElement\">" + i + "</th>")
    html.write("</tr>")

def __write_row(html, top_3, top_10, top_30, confidence_by_slice, slice_size):
    html.write("<tr class = \"tableRow\">")
    html.write("<td class = \"tableHeader\">" + str(slice_size) + "</td>")
    html.write("<td class = \"tableHeader\">")
    html.write("<ul class = \"topList\">")
    for i in range(3):
        html.write("<li class = \"topElement\">" + str(top_3[i][0]) + "(" + str(top_3[i][1]) + ")</li>")
    html.write("</ul>")
    html.write("</td>")
    html.write("<td class = \"tableHeader\">")
    html.write("<ul class = \"topList\">")
    for i in range(10):
        html.write("<li class = \"topElement\">" + str(top_10[i][0]) + "(" + str(top_10[i][1]) + ")</li>")
    html.write("</ul>")
    html.write("</td>")
    html.write("<td class = \"tableHeader\">")
    html.write("<ul class = \"topList\">")
    for i in range(30):
        html.write("<li class = \"topElement\">" + str(top_30[i][0]) + "(" + str(top_30[i][1]) + ")</li>")
    html.write("</ul>")
    html.write("</td>")
    html.write("<td class = \"tableHeader\">")
    html.write("<ul class = \"topList\">")
    for i in range(30):
        html.write("<li class = \"topElement\">" + str(confidence_by_slice[i][0]) + "(" + str(round(confidence_by_slice[i][1], 3)) + ")</li>")
    html.write("</ul>")
    html.write("</td>")
    html.write("</tr>")

def __write_post(html):
    html.write("</table>")
    html.write("</body>")
    html.write("</html>")       

def html_print(html, top_3, top_10, top_30, confidence_by_slice, range_list):
	__write_pre(html)
	[__write_row(html, top_3[i], top_10[i], top_30[i], confidence_by_slice[i], range_list[i]) for i in range(len(top_3))]
	__write_post(html)

