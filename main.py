from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st
import time
from detect import detect
import os
import sys
import argparse
from PIL import Image
from draw import read_data
from draw import create_line_chart
import plotly.graph_objects as go
from draw import create_comparison_table
import plotly.express as px
import plotly.io as pio


def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result


def get_detection_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)

def accurary():
    accuracy_file = "metrics.xlsx"
    accuracy_data = read_data(accuracy_file)
    accuracy_chart = create_line_chart(accuracy_data['Iteration'], [accuracy_data['Accuracy']],
                                       "准确率变化曲线", "Iteration", "Accuracy")

    return accuracy_chart
def loss():
    accuracy_file = "metrics.xlsx"
    accuracy_data = read_data(accuracy_file)
    # 创建动态折线图
    accuracy_chart = create_line_chart(accuracy_data['Iteration'][121:], [accuracy_data['TotalLoss'][121:]],
                                       "损失值变化曲线", "Iteration", "TotalLoss")
    return accuracy_chart
def mAP():
    accuracy_file = "mAP.xlsx"
    # 读取数据
    accuracy_data = read_data(accuracy_file)
    # 创建动态折线图
    accuracy_chart = create_line_chart(accuracy_data['x'], [accuracy_data['MT'],
                                                            accuracy_data['MT+T'],accuracy_data['MT+T+SS']],
                                       "mAP变化曲线", "Iteration", "mAP")
    return accuracy_chart


def mAPl():
    accuracy_file = "mAPl.xlsx"
    # 读取数据
    accuracy_data = read_data(accuracy_file)
    # 创建动态折线图
    accuracy_chart = create_line_chart(accuracy_data['x'], [accuracy_data['MT'],
                                                            accuracy_data['MT+T'], accuracy_data['MT+T+SS']],
                                       "mAPl变化曲线", "Iteration", "mAP")
    return accuracy_chart


def mAPs():
    accuracy_file = "mAPs.xlsx"
    # 读取数据
    accuracy_data = read_data(accuracy_file)
    # 创建动态折线图
    accuracy_chart = create_line_chart(accuracy_data['x'], [accuracy_data['MT'],
                                                            accuracy_data['MT+T'], accuracy_data['MT+T+SS']],
                                       "mAPs变化曲线", "Iteration", "mAP")
    return accuracy_chart


def bijiao():
    chart=read_data("bijiao.xlsx")
    # chart=create_comparison_table(file)
    return chart
def countcls():
    df = read_data('class.xlsx')
    # 统计每个class的总数
    class_counts = df['class1'].value_counts()
    # 统计不同image种类数量
    image_counts = df['image1'].nunique()
    if image_counts <= 3:
        # 如果image总数的种类小于等于3，则以每个图像的每个class总数为纵坐标
        grouped_data = df.groupby(['class1', 'image1']).size().unstack(fill_value=0)
        long_format = grouped_data.reset_index().melt(id_vars='class1', var_name='image1', value_name='count')
        # 使用Plotly绘制柱状图
        fig = px.bar(long_format, x='class1', y='count', color='image1', barmode='group')
        fig.update_layout(title='统计类别', xaxis_title='类别', yaxis_title='数量')
    else:
        # 如果image总数的种类大于3，则统计所有图像的每个class总数为纵坐标
        fig = px.bar(x=class_counts.index, y=class_counts.values)
        fig.update_layout(title='统计类别', xaxis_title='类别', yaxis_title='数量')

    return fig


if __name__ == '__main__':

    st.title('行人车辆检测')
    st.subheader('Faster-RCNN | Mean Teacher')
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='weights/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str,
                        default='data/images', help='source')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.35, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true',
                        help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    source = ("行人车辆检测", "模型性能分析")
    source_index = st.sidebar.selectbox("模块选择", range(
        len(source)), format_func=lambda x: source[x])

    if source_index == 0:
        uploaded_files = st.sidebar.file_uploader(
            "上传图片", type=['png', 'jpeg', 'jpg'], accept_multiple_files=True)
        if uploaded_files:
            is_valid = True
            image_paths = [f"图像 {i + 1} 路径: {file.name}" for i,file in enumerate(uploaded_files)]
            selected_image = st.sidebar.selectbox("选择要显示的图像路径", image_paths)
            st.sidebar.write(selected_image)
            if selected_image:
                is_select=True
                selected_file = next(file for file in uploaded_files if file.name == selected_image.split("路径: ")[1])
                if selected_file:
                    st.sidebar.image(selected_file)
                    picture = Image.open(selected_file)
                    picture = picture.save(f'data/images/{selected_file.name}')
                    opt.source = f'data/images/{selected_file.name}'
            else:
                picture = Image.open(uploaded_files[0])
                picture = picture.save(f'data/images/{uploaded_files[0].name}')
                opt.source = f'data/images/{uploaded_files[0].name}'
        else:
            is_valid = False
    else:
        is_valid = False
        #TODO：模型性能分析
        source2=("准确率曲线","损失值曲线","mAP曲线","不同模型mAP比较")
        source2_index = st.sidebar.selectbox("性能指标", range(
            len(source2)), format_func=lambda x: source2[x])
        if source2_index==0:
            fig=accurary()
            st.plotly_chart(fig)

            # 添加导出按钮
            if st.button('导出'):
                # 将图表导出为图片
                pio.write_image(fig, 'data/images/chart.png')
                st.success("图表已导出为图片：chart.png")
            # 添加下载按钮
            if os.path.exists("data/images/chart.png"):
                st.download_button(label="点击此处下载图片", data=open("data/images/chart.png", "rb"), file_name="accuracy.png",
                                   mime="image/png")
        elif source2_index==1:
            fig=loss()
            st.plotly_chart(fig)
            if st.button('导出'):
                # 将图表导出为图片
                pio.write_image(fig, 'data/images/chart.png')
                st.success("图表已导出为图片：chart.png")
        elif source2_index==2:
            fig1=mAP()
            fig2=mAPl()
            fig3=mAPs()
            st.plotly_chart(fig1)
            if st.button('导出mAP曲线'):
                # 将图表导出为图片
                pio.write_image(fig1, 'data/images/chart.png')
                st.success("图表已导出为图片：chart.png")
            st.plotly_chart(fig2)
            if st.button('导出mAPl曲线'):
                # 将图表导出为图片
                pio.write_image(fig1, 'data/images/chart.png')
                st.success("图表已导出为图片：chart.png")
            st.plotly_chart(fig3)
            if st.button('导出mAPs曲线'):
                # 将图表导出为图片
                pio.write_image(fig1, 'data/images/chart.png')
                st.success("图表已导出为图片：chart.png")
        else :
            fig=bijiao()
            # st.write(fig,width=800, height=700)
            # 将 DataFrame 转换为 HTML 表格，并设置样式
            # 设置样式
            styled_df = fig.style.applymap(
                lambda x: f'color: {"blue" if not isinstance(x, str) else "black"}; font-size: 16px',
                # 设置非字符串为蓝色字体，字号16px
                subset=pd.IndexSlice[:, ['B', 'C']]  # 设置 B 列和 C 列的样式
            ).applymap(
                lambda x: 'color: black; font-weight: bold; font-size: 25px',  # 设置表头为黑色字体加粗，字号25px
                subset=pd.IndexSlice[0, :]  # 设置第一行的样式
            ).applymap(
                lambda x: 'color: black; font-weight: bold; font-size: 20px',  # 设置第一列的样式
                subset=pd.IndexSlice[:, ['A']]  # 设置 A 列的样式
            ).format('{:.1f}', subset=pd.IndexSlice[:, ['B', 'C']])  # 设置 B 列和 C 列数字小数点后保留一位

            # 显示 DataFrame
            st.write(styled_df, unsafe_allow_html=True)
    st.session_state.setdefault('is_detect',False)
    if is_valid:
        print('valid')
        if st.button('开始检测'):
            st.session_state['is_detect'] = True
            detect(opt)
        if st.session_state['is_detect']:
            with st.spinner(text='Preparing Images'):
                for img in os.listdir(get_detection_folder()):
                    st.image(str(Path(f'{get_detection_folder()}') / img))
        if st.button('统计类别'):
                if st.session_state['is_detect']:
                    fig1=countcls()
                    st.plotly_chart(fig1)
                    if st.button('导出'):
                        # 将图表导出为图片
                        pio.write_image(fig1, 'data/images/chart.png')
                        st.success("图表已导出为图片：chart.png")
            #st.balloons()
    else:
        pass
