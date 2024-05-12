import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# 读取数据文件
def read_data(file_path):
    return pd.read_excel(file_path)

# 创建动态折线图
def create_line_chart(x_data, y_data, title, x_title, y_title):
    fig = go.Figure()
    for y in y_data:
        fig.add_trace(go.Scatter(x=x_data, y=y, mode='lines', name=y.name))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
    return fig

# 创建比较表格
def create_comparison_table(data):
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(data.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[data[col] for col in data.columns],
                   fill_color='lavender',
                   align='left'))
    ])
    return fig

# 示例数据文件路径
accuracy_file = "accuracy.xlsx"
# loss_file = "loss_data.xlsx"
# map_file = "map_data.xlsx"
# comparison_file = "comparison_data.xlsx"

# 读取数据
accuracy_data = read_data(accuracy_file)
# loss_data = read_data(loss_file)
# map_data = read_data(map_file)
# comparison_data = read_data(comparison_file)

# 创建动态折线图
accuracy_chart = create_line_chart(accuracy_data['Iter'], [accuracy_data['Model']],
                                   "Accuracy Trend", "Epoch", "Accuracy")
#loss_chart = create_line_chart(loss_data['Epoch'], [loss_data['Model_1'], loss_data['Model_2']],
#                                "Loss Trend", "Epoch", "Loss")
# #map_chart = create_line_chart(map_data['Epoch'], [map_data['Model_1'], map_data['Model_2']],
#                               "mAP Trend", "Epoch", "mAP")

# 创建比较表格
#comparison_table = create_comparison_table(comparison_data)

# 将图表和表格保存为 HTML 文件
accuracy_chart.write_html("accuracy_chart.html")
# loss_chart.write_html("loss_chart.html")
# map_chart.write_html("map_chart.html")
# comparison_table.write_html("comparison_table.html")
