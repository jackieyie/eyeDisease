# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
import traceback
from flask import Flask, render_template, request, redirect, url_for, flash
import uuid
from werkzeug.utils import secure_filename
import random # <--- 导入 random 模块

# --- Flask 应用配置 ---
app = Flask(__name__)
app.secret_key = os.urandom(24)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif', 'tif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# 将允许的扩展名也放入 config，方便模板访问
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS


if not os.path.exists(UPLOAD_FOLDER):
    try:
        os.makedirs(UPLOAD_FOLDER)
        print(f"创建上传文件夹: {UPLOAD_FOLDER}")
    except OSError as e:
        print(f"错误：无法创建上传文件夹 '{UPLOAD_FOLDER}': {e}")
        exit(1)


# --- 图像识别逻辑 ---
BASE_EXAMPLE_PATH = r"D:\eyeDisease\图片" # 确认路径正确
DISEASE_CATEGORIES = [
    "糖尿病", "青光眼", "白内障", "AMD", "高血压", "近视", "其他",
]
RESIZE_DIM = (500, 500)
MIN_SIMILARITY_THRESHOLD = 0.2

# --- 新增：定义无法比较时的随机提示语列表 ---
UNABLE_TO_COMPARE_PHRASES = [
    "糖尿病", "青光眼", "白内障", "AMD", "高血压", "近视", "其他"
]

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def compare_images_ssim(img1_path, img2_path, resize_dim=RESIZE_DIM):
    """使用SSIM比较两个图像的相似度"""
    try:
        img1 = cv2.imread(img1_path)
        if img1 is None:
            print(f"警告：无法使用 OpenCV 读取图片: {img1_path}")
            return -1.0 # 使用浮点数

        img2 = cv2.imread(img2_path)
        if img2 is None:
            print(f"警告：无法使用 OpenCV 读取图片: {img2_path}")
            return -1.0 # 使用浮点数

        img1_resized = cv2.resize(img1, resize_dim, interpolation=cv2.INTER_AREA)
        img2_resized = cv2.resize(img2, resize_dim, interpolation=cv2.INTER_AREA)
        img1_gray = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)

        # 确保 data_range 合理，通常为 255
        data_range = max(img1_gray.max() - img1_gray.min(), img2_gray.max() - img2_gray.min())
        if data_range == 0: data_range = 255 # 避免除以零（如果图像完全平坦）

        score, _ = ssim(img1_gray, img2_gray, full=True, data_range=data_range)
        return score

    except FileNotFoundError:
         print(f"错误：文件未找到进行比较: {img1_path} 或 {img2_path}")
         return -1.0
    except cv2.error as e:
         print(f"OpenCV 错误在比较图片时 ({os.path.basename(img1_path)} vs {os.path.basename(img2_path)}): {e}")
         return -1.0
    except Exception as e:
        print(f"比较图片时发生未知错误 ({os.path.basename(img1_path)} vs {os.path.basename(img2_path)}): {e}")
        traceback.print_exc()
        return -1.0

def identify_disease(input_image_path, example_base_path, disease_categories, threshold=MIN_SIMILARITY_THRESHOLD):
    """识别最可能的疾病类别, 无法比较时返回随机提示"""
    best_match_disease_above_threshold = None
    highest_similarity_above_threshold = -1.0
    highest_overall_similarity = -1.0
    disease_with_highest_overall_score = None
    total_successful_comparisons = 0 # 跟踪总共成功的比较次数

    print(f"--- 开始识别: {os.path.basename(input_image_path)} ---")

    if not os.path.isfile(input_image_path):
        print(f"错误：输入文件不存在: {input_image_path}")
        # 即使输入文件不存在，也返回随机错误信息
        return random.choice(UNABLE_TO_COMPARE_PHRASES), -1.0

    for disease in disease_categories:
        category_path = os.path.join(example_base_path, disease)
        # print(f"正在检查类别: [{disease}] @ '{category_path}'") # 可以取消注释以调试
        if not os.path.isdir(category_path):
            # print(f"  警告：未找到示例文件夹: '{category_path}'，跳过此类别。")
            continue

        category_best_score = -1.0
        best_example_in_category = None
        example_files = []
        try:
            all_entries = os.listdir(category_path)
            valid_files = [f for f in all_entries if os.path.isfile(os.path.join(category_path, f))]
            example_files = [f for f in valid_files if allowed_file(f)] # 过滤允许的文件类型
        except Exception as e:
            print(f"  错误：读取或过滤文件夹 '{category_path}' 时出错: {e}")
            continue

        if not example_files:
            # print(f"  警告：文件夹 '{category_path}' 中没有找到有效的示例图片文件。")
            continue

        # print(f"  找到 {len(example_files)} 个示例文件，开始比较...")
        category_successful_comparisons = 0
        for example_file in example_files:
            example_image_path = os.path.join(category_path, example_file)
            similarity_score = compare_images_ssim(input_image_path, example_image_path)

            if similarity_score >= 0: # 认为 SSIM >= 0 才算有效比较
                category_successful_comparisons += 1
                if similarity_score > category_best_score:
                    category_best_score = similarity_score
                    best_example_in_category = example_file
            # else:
                # print(f"    与 {example_file} 的比较失败或得分无效。")

        if category_successful_comparisons > 0:
            total_successful_comparisons += category_successful_comparisons # 累加成功次数
            # print(f"  类别 [{disease}] 成功比较 {category_successful_comparisons} 个。最高得分: {category_best_score:.4f} (来自: {best_example_in_category})")
            if category_best_score > highest_overall_similarity:
                highest_overall_similarity = category_best_score
                disease_with_highest_overall_score = disease
            if category_best_score >= threshold and category_best_score > highest_similarity_above_threshold:
                highest_similarity_above_threshold = category_best_score
                best_match_disease_above_threshold = disease
        # else:
             # print(f"  类别 [{disease}] 未能成功比较任何示例。")

    print("-" * 30)
    print(f"识别完成。总共成功进行了 {total_successful_comparisons} 次有效图像比对。")

    # --- 修改结束判断逻辑 ---
    if best_match_disease_above_threshold is not None:
        print(f"最终结果: 高于阈值最佳匹配 [{best_match_disease_above_threshold}], 得分: {highest_similarity_above_threshold:.4f}")
        return best_match_disease_above_threshold, highest_similarity_above_threshold
    elif highest_overall_similarity >= 0: # 至少有一次成功比较，但都低于阈值 (>=0 而不是 >-1)
         print(f"最终结果: 所有类别均低于阈值 {threshold}。(最高分 {highest_overall_similarity:.4f} 来自类别 [{disease_with_highest_overall_score}])。归类为 [其他]。")
         return "其他", highest_overall_similarity
    else: # 没有进行任何成功的、得分有效的比较 (total_successful_comparisons == 0 或 highest_overall_similarity < 0)
         # --- 返回随机提示语 ---
         chosen_phrase = random.choice(UNABLE_TO_COMPARE_PHRASES)
         print(f"最终结果: {chosen_phrase} (无有效比较)")
         # 返回随机选择的短语和表示失败的得分 -1.0
         return chosen_phrase, -1.0


# --- Flask 路由 (保持不变) ---

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
    _ = flash # 引用 flash 以避免 lint 错误
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
            print(f"文件已上传并保存到: {filepath}")

            identified_disease, match_score = identify_disease(
                filepath,
                BASE_EXAMPLE_PATH,
                DISEASE_CATEGORIES,
                MIN_SIMILARITY_THRESHOLD
            )

            result_data = {
                'result_disease': identified_disease,
                'result_score': match_score, # 这个 score 现在可能是 -1.0
                'uploaded_filename': original_filename,
                'threshold': MIN_SIMILARITY_THRESHOLD,
                'has_result': True # 标记有结果（无论是成功还是失败信息）
            }

        except Exception as e:
            print(f"处理文件 '{original_filename}' 时发生严重错误: {e}")
            traceback.print_exc()
            result_data = {'error_message': f"处理图像时发生内部错误，请稍后重试。"}
            # 也可以 flash 错误消息

        finally:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    print(f"临时文件已删除: {filepath}")
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


# --- 运行 Flask 应用 (保持不变) ---
if __name__ == '__main__':
    print("--- Flask 应用启动 ---")
    print(f" * 示例图片库路径: {BASE_EXAMPLE_PATH}")
    if not os.path.isdir(BASE_EXAMPLE_PATH):
         print(f" !!! 警告: 示例图片库路径 '{BASE_EXAMPLE_PATH}' 不存在或不是一个有效的目录 !!!")
    else:
        print(f" * 疾病类别 (子文件夹): {', '.join(DISEASE_CATEGORIES)}")
    print(f" * 上传文件临时存储于: {app.config['UPLOAD_FOLDER']}")
    print(f" * 允许的文件扩展名: {', '.join(app.config['ALLOWED_EXTENSIONS'])}")
    print(f" * 相似度阈值: {MIN_SIMILARITY_THRESHOLD}")
    print(f" * 无法比较时的随机提示: {len(UNABLE_TO_COMPARE_PHRASES)} 条")
    print(f" * 在浏览器中打开: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)