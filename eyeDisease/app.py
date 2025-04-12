# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
import traceback
from flask import Flask, render_template, request, redirect, url_for, flash
import uuid # 用于生成唯一文件名
from werkzeug.utils import secure_filename # 用于安全处理文件名

# --- Flask 应用配置 ---
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/' # 用于 flash 消息，随便设置一个复杂的即可
# 设置上传文件临时存储的文件夹
UPLOAD_FOLDER = 'uploads'
# 允许上传的文件扩展名 (可以根据需要调整)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif', 'tif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 确保上传文件夹存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- 图像识别逻辑 (从你之前的代码复制并稍作修改) ---

# 1. 定义存储示例图片的根目录 (!!! 重要：确认这个路径对运行 app.py 的环境是正确的 !!!)
# 使用绝对路径可能更可靠
# BASE_EXAMPLE_PATH = r"C:\绝对路径\到\你的\disease_examples"
BASE_EXAMPLE_PATH = r"C:\Users\倪永杰\Auto_Loser\图片" # 使用你提供的路径 (确保 Flask 能访问)

# 2. 疾病类别列表 (必须与文件夹名称完全一致)
DISEASE_CATEGORIES = [
    "糖尿病", "青光眼", "白内障", "AMD", "高血压", "近视", "其他",
]

# 3. 图像比较参数
RESIZE_DIM = (500, 500)
MIN_SIMILARITY_THRESHOLD = 0.2 # 低于此阈值的结果将被视为“其他”

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def compare_images_ssim(img1_path, img2_path, resize_dim=RESIZE_DIM):
    """使用SSIM比较两个图像的相似度 (基本保持不变，增加了文件未找到的检查)"""
    try:
        # 使用健壮的方式读取图片
        img1_stream = open(img1_path, "rb")
        img1_bytes = bytearray(img1_stream.read())
        img1_np = np.asarray(img1_bytes, dtype=np.uint8)
        img1 = cv2.imdecode(img1_np, cv2.IMREAD_COLOR)
        img1_stream.close()

        img2_stream = open(img2_path, "rb")
        img2_bytes = bytearray(img2_stream.read())
        img2_np = np.asarray(img2_bytes, dtype=np.uint8)
        img2 = cv2.imdecode(img2_np, cv2.IMREAD_COLOR)
        img2_stream.close()

        if img1 is None or img2 is None:
            print(f"警告：无法解码图片: {img1_path if img1 is None else img2_path}")
            return -1

        img1_resized = cv2.resize(img1, resize_dim, interpolation=cv2.INTER_AREA)
        img2_resized = cv2.resize(img2, resize_dim, interpolation=cv2.INTER_AREA)
        img1_gray = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)

        score, diff = ssim(img1_gray, img2_gray, full=True, data_range=img1_gray.max() - img1_gray.min())
        return score

    except FileNotFoundError:
         print(f"错误：文件未找到: {img1_path} 或 {img2_path}")
         return -1 # 指示文件找不到
    except Exception as e:
        print(f"比较图片时出错 ({os.path.basename(img1_path)} vs {os.path.basename(img2_path)}): {e}")
        return -1

def identify_disease(input_image_path, example_base_path, disease_categories, threshold=MIN_SIMILARITY_THRESHOLD):
    """识别最可能的疾病类别 (基本保持不变)"""
    # 检查输入路径是否存在由调用者保证

    best_match_disease_above_threshold = None
    highest_similarity_above_threshold = -1
    highest_overall_similarity = -1
    disease_with_highest_overall_score = None

    print(f"--- 开始识别: {os.path.basename(input_image_path)} ---")

    for disease in disease_categories:
        category_path = os.path.join(example_base_path, disease)
        print(f"正在检查类别: [{disease}]")
        if not os.path.isdir(category_path):
            print(f"  警告：未找到文件夹: '{category_path}'")
            continue

        category_best_score = -1
        best_example_in_category = None
        example_files = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]

        if not example_files:
            print(f"  警告：文件夹 '{category_path}' 中无示例图片。")
            continue

        print(f"  比较 {len(example_files)} 个示例...")
        for example_file in example_files:
            example_image_path = os.path.join(category_path, example_file)
            similarity_score = compare_images_ssim(input_image_path, example_image_path)
            if similarity_score == -1: # 如果比较出错或文件找不到，跳过这个示例
                print(f"    跳过比较: {example_file} (比较失败)")
                continue
            if similarity_score > category_best_score:
                category_best_score = similarity_score
                best_example_in_category = example_file

        if category_best_score > -1: # 确保至少有一个成功的比较
            print(f"  类别 [{disease}] 最高得分: {category_best_score:.4f} (来自: {best_example_in_category})")
            if category_best_score > highest_overall_similarity:
                highest_overall_similarity = category_best_score
                disease_with_highest_overall_score = disease
            if category_best_score >= threshold and category_best_score > highest_similarity_above_threshold:
                highest_similarity_above_threshold = category_best_score
                best_match_disease_above_threshold = disease
        else:
             print(f"  类别 [{disease}] 未能成功比较任何示例。")


    print("-" * 30)
    print(f"识别完成。")
    if best_match_disease_above_threshold is not None:
        print(f"结果: 高于阈值最佳匹配 [{best_match_disease_above_threshold}], 得分: {highest_similarity_above_threshold:.4f}")
        return best_match_disease_above_threshold, highest_similarity_above_threshold
    elif highest_overall_similarity > -1 :
         print(f"结果: 所有类别低于阈值 (最高分 {highest_overall_similarity:.4f} 来自 [{disease_with_highest_overall_score}])。归类为 [其他]。")
         return "其他", highest_overall_similarity
    else:
         print(f"结果: 未能进行有效比较。归类为 [无法比较]。")
         return "无法比较", -1

# --- Flask 路由 ---

@app.route('/', methods=['GET'])
def index():
    """显示主页上传表单"""
    # 初始加载页面时，没有结果传递给模板
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传和识别"""
    if 'file' not in request.files:
        flash('未检测到上传的文件部分', 'error') # 使用 flash 显示错误给用户 (可选)
        return redirect(request.url) # 重定向回上传页面

    file = request.files['file']

    if file.filename == '':
        flash('未选择任何文件', 'error')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # 使用 secure_filename 防止恶意文件名
        # 为了避免重名，可以加上 UUID
        original_filename = secure_filename(file.filename)
        unique_filename = str(uuid.uuid4()) + "_" + original_filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        try:
            # 保存上传的文件
            file.save(filepath)
            print(f"文件已保存到: {filepath}")

            # --- 执行识别 ---
            identified_disease, match_score = identify_disease(
                filepath,
                BASE_EXAMPLE_PATH,
                DISEASE_CATEGORIES,
                MIN_SIMILARITY_THRESHOLD
            )

            # --- 准备结果传递给模板 ---
            # 注意：模板中可以直接使用这些变量名
            return render_template('index.html',
                                   result_disease=identified_disease,
                                   result_score=match_score,
                                   uploaded_filename=original_filename, # 显示原始文件名给用户
                                   threshold=MIN_SIMILARITY_THRESHOLD) # 把阈值也传给模板用于显示

        except Exception as e:
            print(f"处理文件时发生错误: {e}")
            traceback.print_exc()
            # 向模板传递错误信息
            return render_template('index.html', error_message=f"处理图像时发生错误: {e}")

        finally:
            # --- (可选) 清理上传的文件 ---
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    print(f"临时文件已删除: {filepath}")
                except Exception as e:
                    print(f"删除临时文件失败: {e}")

    else:
        # 如果文件类型不允许
        return render_template('index.html', error_message="不允许的文件类型。请上传图片文件 (png, jpg, jpeg, bmp, gif, tif)。")

    # 如果有未处理的情况，重定向回首页
    return redirect(url_for('index'))

# --- 运行 Flask 应用 ---
if __name__ == '__main__':
    print("--- Flask 应用启动 ---")
    print(f"示例图片库路径: {BASE_EXAMPLE_PATH}")
    if not os.path.isdir(BASE_EXAMPLE_PATH):
         print(f"!!! 警告: 示例图片库路径 '{BASE_EXAMPLE_PATH}' 不存在或不是一个目录 !!!")
    print(f"疾病类别: {', '.join(DISEASE_CATEGORIES)}")
    print(f"上传文件临时存储于: {app.config['UPLOAD_FOLDER']}")
    print("请在浏览器中打开 http://127.0.0.1:5000")
    # debug=True 会在代码更改时自动重载，方便开发，但生产环境不要用
    # host='0.0.0.0' 可以让局域网内其他设备访问，如果只需要本机访问用 '127.0.0.1'
    app.run(debug=True, host='127.0.0.1', port=5000)