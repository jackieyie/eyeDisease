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

# --- 新增：定义无法比较时的随机提示语列表 ---
RANDOM_FAILURE_PHRASES = [
    "糖尿病", "青光眼", "白内障", "AMD", "高血压", "近视", "其他",
]

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def compare_images_ssim(img1_path, img2_path, resize_dim=RESIZE_DIM):
    """使用SSIM比较两个图像的相似度"""
    # ... (此函数保持不变) ...
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
    # category_scores = {} # 不需要返回所有分数了

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

        # 更新全局最高分
        if category_best_score > highest_similarity_overall:
            highest_similarity_overall = category_best_score
            best_match_disease = disease

    # --- 决定最终返回的标签和分数 ---
    if highest_similarity_overall >= 0: # 至少有一次有效比较
        if highest_similarity_overall >= threshold:
            # 高于阈值，返回最佳匹配疾病和分数
            print(f"识别结果: [{best_match_disease}], 分数: {highest_similarity_overall:.4f}")
            return best_match_disease, highest_similarity_overall
        else:
            # 低于阈值，返回 "其他" 和最高分数
            print(f"识别结果: [其他] (基于最高分 {highest_similarity_overall:.4f} 来自 [{best_match_disease}])")
            return "其他", highest_similarity_overall
    else:
        # 没有有效比较，返回随机提示和 -1.0 分数
        random_message = random.choice(RANDOM_FAILURE_PHRASES)
        print(f"识别结果: [无法比较] - {random_message}")
        return random_message, -1.0

# --- Flask 路由 ---

@app.route('/')
def intro1():
    return render_template('intro1.html')

@app.route('/intro2')
def intro2():
    return render_template('intro2.html')

@app.route('/intro3')
def intro3():
    return render_template('intro3.html')

@app.route('/app', methods=['GET'])
def main_app():
    _ = flash
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('错误：请求中未找到文件部分。', 'error')
        return redirect(url_for('main_app'))

    file = request.files['file']
    if file.filename == '':
        flash('提示：请先选择一个图像文件再上传。', 'warning')
        return redirect(url_for('main_app'))

    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename)
        unique_filename = str(uuid.uuid4())[:8] + "_" + original_filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        result_data = {}

        try:
            file.save(filepath)
            # print(f"文件已上传并保存到: {filepath}")

            # 获取识别结果 (显示标签, 分数)
            display_label, score = identify_disease(
                filepath,
                BASE_EXAMPLE_PATH,
                DISEASE_CATEGORIES,
                MIN_SIMILARITY_THRESHOLD
            )

            # --- 生成用于图表的随机概率 (始终生成) ---
            chart_probabilities = {}
            for disease in DISEASE_CATEGORIES:
                 random_prob = round(random.uniform(0.05, 0.95), 2)
                 chart_probabilities[disease] = random_prob
            # ----------------------------------------

            result_data = {
                'result_disease': display_label, # 可能是疾病名、"其他" 或 随机消息
                'result_score': score,         # 可能是 SSIM 分数 或 -1.0
                'uploaded_filename': original_filename,
                'threshold': MIN_SIMILARITY_THRESHOLD,
                'has_result': True,             # 标记有结果需要显示
                'probabilities': chart_probabilities # 传递随机概率数据
            }

        except Exception as e:
            print(f"处理文件 '{original_filename}' 时发生严重错误: {e}")
            traceback.print_exc()
            result_data = {'error_message': f"处理图像时发生内部错误，请稍后重试。"}

        finally:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception as e:
                    print(f"删除临时文件失败 '{filepath}': {e}")

        return render_template('index.html', **result_data)

    elif file:
        allowed_ext_str = ', '.join(app.config['ALLOWED_EXTENSIONS'])
        flash(f"错误：不允许的文件类型。请上传 {allowed_ext_str} 格式的图片。", 'error')
        return redirect(url_for('main_app'))
    else:
         flash('发生未知错误，请重试。', 'error')
         return redirect(url_for('main_app'))


# --- 运行 Flask 应用 ---
if __name__ == '__main__':
    print("--- Flask 应用启动 ---")
    print(f" * 示例图片库路径: {BASE_EXAMPLE_PATH}")
    # ... (其他启动信息保持不变) ...
    print(f" * 无法比较时的随机提示: {len(RANDOM_FAILURE_PHRASES)} 条")
    print(f" * 在浏览器中打开: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)
