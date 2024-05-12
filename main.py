from io import StringIO
from pathlib import Path
import streamlit as st
import time
from detect import detect
import os
import sys
import argparse
from PIL import Image
from draw import read_data
from draw import create_line_chart
from draw import create_comparison_table


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
    # 示例数据文件路径
    accuracy_file = "metrics.xlsx"
    # loss_file = "loss_data.xlsx"
    # map_file = "map_data.xlsx"
    # comparison_file = "comparison_data.xlsx"

    # 读取数据
    accuracy_data = read_data(accuracy_file)
    # loss_data = read_data(loss_file)
    # map_data = read_data(map_file)
    # comparison_data = read_data(comparison_file)

    # 创建动态折线图
    accuracy_chart = create_line_chart(accuracy_data['Iteration'], [accuracy_data['Accuracy']],
                                       "准确率变化曲线", "Iteration", "Accuracy")
    return accuracy_chart

def loss():
    accuracy_file = "metrics.xlsx"
    # loss_file = "loss_data.xlsx"
    # map_file = "map_data.xlsx"
    # comparison_file = "comparison_data.xlsx"

    # 读取数据
    accuracy_data = read_data(accuracy_file)
    # loss_data = read_data(loss_file)
    # map_data = read_data(map_file)
    # comparison_data = read_data(comparison_file)

    # 创建动态折线图
    accuracy_chart = create_line_chart(accuracy_data['Iteration'][121:], [accuracy_data['TotalLoss'][121:]],
                                       "准确率变化曲线", "Iteration", "TotalLoss")
    return accuracy_chart


def mAP():
    accuracy_file = "mAP.xlsx"
    # loss_file = "loss_data.xlsx"
    # map_file = "map_data.xlsx"
    # comparison_file = "comparison_data.xlsx"

    # 读取数据
    accuracy_data = read_data(accuracy_file)
    # loss_data = read_data(loss_file)
    # map_data = read_data(map_file)
    # comparison_data = read_data(comparison_file)

    # 创建动态折线图
    accuracy_chart = create_line_chart(accuracy_data['x'], [accuracy_data['MT'],
                                                            accuracy_data['MT+T'],accuracy_data['MT+T+SS']],
                                       "mAP变化曲线", "Iteration", "mAP")
    return accuracy_chart


def mAPl():
    pass


def mAPs():
    pass


def bijiao():
    chart=read_data("bijiao.xlsx")
    # chart=create_comparison_table(file)
    return chart


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
        uploaded_file = st.sidebar.file_uploader(
            "上传图片", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='资源加载中...'):
                st.sidebar.image(uploaded_file)
                picture = Image.open(uploaded_file)
                picture = picture.save(f'data/images/{uploaded_file.name}')
                opt.source = f'data/images/{uploaded_file.name}'
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
            # 展示 accuracy_chart.html
            ## st.markdown("## Accuracy Chart")
            ## st.markdown("<iframe src='accuracy_chart.html' width='1000' height='600'></iframe>", unsafe_allow_html=True)
            st.plotly_chart(fig)
        elif source2_index==1:
            fig=loss()
            st.plotly_chart(fig)

        elif source2_index==2:
            fig=mAP()
            mAPl()
            mAPs()
            st.plotly_chart(fig)
        else :
            fig=bijiao()
            st.write(fig)
            
    if is_valid:
        print('valid')
        if st.button('开始检测'):

            detect(opt)

            if source_index == 0:
                with st.spinner(text='Preparing Images'):
                    for img in os.listdir(get_detection_folder()):
                        st.image(str(Path(f'{get_detection_folder()}') / img))

                    #st.balloons()
            else:
                # TODO：模型性能分析
                pass
    else:
        pass
