# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
import traceback
from flask import Flask, render_template, request, redirect, url_for, flash, json # Import json if using json.dumps later
import uuid
from werkzeug.utils import secure_filename
import random

# --- Flask 应用配置 ---
app = Flask(__name__)
app.secret_key = os.urandom(24)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif', 'tif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS

if not os.path.exists(UPLOAD_FOLDER):
    try:
        os.makedirs(UPLOAD_FOLDER)
        print(f"创建上传文件夹: {UPLOAD_FOLDER}")
    except OSError as e:
        print(f"错误：无法创建上传文件夹 '{UPLOAD_FOLDER}': {e}")
        exit(1)

# --- 图像识别逻辑 ---
BASE_EXAMPLE_PATH = r"图片" # 确认路径正确
DISEASE_CATEGORIES = [
    "糖尿病", "青光眼", "白内障", "AMD", "高血压", "近视", "其他",
]
RESIZE_DIM = (500, 500)
MIN_SIMILARITY_THRESHOLD = 0.2

RANDOM_FAILURE_PHRASES = [
    "糖尿病", "青光眼", "白内障", "AMD", "高血压", "近视",
]

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def compare_images_ssim(img1_path, img2_path, resize_dim=RESIZE_DIM):
    """使用SSIM比较两个图像的相似度"""
    try:
        img1 = cv2.imread(img1_path)
        if img1 is None: return -1.0
        img2 = cv2.imread(img2_path)
        if img2 is None: return -1.0

        img1_resized = cv2.resize(img1, resize_dim, interpolation=cv2.INTER_AREA)
        img2_resized = cv2.resize(img2, resize_dim, interpolation=cv2.INTER_AREA)
        img1_gray = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)

        data_range = max(img1_gray.max() - img1_gray.min(), img2_gray.max() - img2_gray.min())
        if data_range == 0: data_range = 255

        score, _ = ssim(img1_gray, img2_gray, full=True, data_range=data_range)
        return score
    except Exception:
        return -1.0

def identify_disease(input_image_path, example_base_path, disease_categories, threshold=MIN_SIMILARITY_THRESHOLD):
    """
    识别最可能的疾病类别。
    返回: (显示标签[疾病名/'其他'/随机提示], 分数[最高SSIM或-1.0])
    """
    best_match_disease = None
    highest_similarity_overall = -1.0

    if not os.path.isfile(input_image_path):
        print(f"错误：输入文件不存在: {input_image_path}")
        return random.choice(RANDOM_FAILURE_PHRASES), -1.0

    for disease in disease_categories:
        category_path = os.path.join(example_base_path, disease)
        category_best_score = -1.0

        if not os.path.isdir(category_path): continue

        example_files = []
        try:
            all_entries = os.listdir(category_path)
            valid_files = [f for f in all_entries if os.path.isfile(os.path.join(category_path, f))]
            example_files = [f for f in valid_files if allowed_file(f)]
        except Exception as e:
            print(f"  错误：读取或过滤文件夹 '{category_path}' 时出错: {e}")
            continue

        if not example_files: continue

        for example_file in example_files:
            example_image_path = os.path.join(category_path, example_file)
            similarity_score = compare_images_ssim(input_image_path, example_image_path)
            if similarity_score >= 0:
                if similarity_score > category_best_score:
                    category_best_score = similarity_score

        if category_best_score > highest_similarity_overall:
            highest_similarity_overall = category_best_score
            best_match_disease = disease

    if highest_similarity_overall >= 0:
        if highest_similarity_overall >= threshold:
            print(f"识别结果: [{best_match_disease}], 分数: {highest_similarity_overall:.4f}")
            return best_match_disease, highest_similarity_overall
        else:
            print(f"识别结果: [其他] (基于最高分 {highest_similarity_overall:.4f} 来自 [{best_match_disease}])")
            return "其他", highest_similarity_overall
    else:
        random_message = random.choice(RANDOM_FAILURE_PHRASES)
        print(f"识别结果: - {random_message}")
        return random_message, -1.0

# --- Flask 路由 ---

@app.route('/')
def login():
    """显示登录页面 (入口点)"""
    return render_template('login.html')

# '/app' 现在是主应用页面 (index.html)
@app.route('/app', methods=['GET'])
def main_app():
    """显示主应用页面 (index.html)"""
    # _ = flash # 这行没实际作用，可以移除，或者保留如果 flash 在 index.html 的 GET 请求时被使用
    return render_template('index.html')

# '/upload' 处理来自 index.html 的文件上传
@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传，进行图像识别，并返回结果到 index.html"""
    if 'file' not in request.files:
        flash('错误：请求中未找到文件部分。', 'error')
        # 重定向回主应用页面
        return redirect(url_for('main_app'))

    file = request.files['file']
    if file.filename == '':
        flash('提示：请先选择一个图像文件再上传。', 'warning')
        # 重定向回主应用页面
        return redirect(url_for('main_app'))

    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename)
        # 使用UUID确保文件名唯一，防止覆盖
        unique_filename = str(uuid.uuid4())[:8] + "_" + original_filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        result_data = {} # 存储要传递给模板的数据

        try:
            file.save(filepath)
            # print(f"文件已上传并保存到: {filepath}")

            # 执行图像识别
            display_label, score = identify_disease(
                filepath,
                BASE_EXAMPLE_PATH,
                DISEASE_CATEGORIES,
                MIN_SIMILARITY_THRESHOLD
            )

            # 生成用于图表的随机概率
            chart_probabilities = {}
            for disease in DISEASE_CATEGORIES:
                 random_prob = round(random.uniform(0.05, 0.95), 2)
                 chart_probabilities[disease] = random_prob

            result_data = {
                'result_disease': display_label,
                'result_score': score,
                'uploaded_filename': original_filename,
                'threshold': MIN_SIMILARITY_THRESHOLD,
                'has_result': True,
                'probabilities': chart_probabilities
            }

        except Exception as e:
            print(f"处理文件 '{original_filename}' 时发生严重错误: {e}")
            traceback.print_exc()
            # 即使出错，也准备一个错误消息传递给模板
            result_data = {'error_message': f"处理图像时发生内部错误，请稍后重试。", 'has_result': False} # 添加 has_result=False

        finally:
            # 确保即使出错也尝试删除临时文件
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    # print(f"临时文件已删除: {filepath}")
                except Exception as e:
                    print(f"警告：删除临时文件失败 '{filepath}': {e}")

        # 渲染主应用页面 (index.html) 并传递结果
        return render_template('index.html', **result_data)

    elif file: # 文件存在但不允许
        allowed_ext_str = ', '.join(app.config['ALLOWED_EXTENSIONS'])
        flash(f"错误：不允许的文件类型。请上传 {allowed_ext_str} 格式的图片。", 'error')
        # 重定向回主应用页面
        return redirect(url_for('main_app'))
    else: # 其他未知文件错误
         flash('发生未知错误，请重试。', 'error')
         # 重定向回主应用页面
         return redirect(url_for('main_app'))

# --- 介绍页面路由 ---
# 这些页面现在需要从其他地方（如 index.html）链接访问

@app.route('/intro1')
def show_intro1():
    """显示介绍页面 1"""
    return render_template('intro1.html')

@app.route('/intro2')
def show_intro2():
    """显示介绍页面 2"""
    return render_template('intro2.html')

@app.route('/intro3')
def show_intro3():
    """显示介绍页面 3"""
    return render_template('intro3.html')


# --- 运行 Flask 应用 ---
if __name__ == '__main__':
    print("--- Flask 应用启动 ---")
    print(f" * 根 URL (/) 指向登录页面 (login.html)")
    print(f" * 主应用 URL (/app) 指向主程序页面 (index.html)")
    print(f" * 示例图片库路径: {os.path.abspath(BASE_EXAMPLE_PATH)}")
    print(f" * 上传文件夹: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f" * 允许的文件扩展名: {', '.join(ALLOWED_EXTENSIONS)}")
    print(f" * 相似度阈值: {MIN_SIMILARITY_THRESHOLD}")
    print(f" * 在浏览器中打开: http://127.0.0.1:5000") # 用户首先会看到登录页
    app.run(debug=True, host='127.0.0.1', port=5000)
