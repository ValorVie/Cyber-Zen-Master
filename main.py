import os
import re

from analyze_agent import AnalyzeAgent
from debator_agent import DebatorAgent
from express_agent import ExpressAgent


def sanitize_folder_name(text):
    """將辯題轉換為合法的文件夾名稱"""
    sanitized = text.replace("/", "vs")
    sanitized = re.sub(r'[\\/*?:"<>|]', "", sanitized)
    sanitized = "_".join(sanitized.split())
    return sanitized


def setup_output_dirs(base_dir):
    """創建輸出目錄結構"""
    stages = ["stage1", "stage2", "stage3"]
    dirs = {}
    for stage in stages:
        stage_dir = os.path.join(base_dir, stage)
        os.makedirs(stage_dir, exist_ok=True)
        dirs[stage] = stage_dir
    return dirs


def run_answer_anything_system(full_topic):
    """運行完整的解答系統"""
    base_output_dir = os.path.join("output", sanitize_folder_name(full_topic))
    output_dirs = setup_output_dirs(base_output_dir)

    # 第一階段：分析
    print(f"\nStage 1: 分析階段 ({full_topic})")
    analyzer = AnalyzeAgent()
    analyzer.OUTPUT_DIR = output_dirs["stage1"]

    print("聯想分析:")
    analyzer.opposite_opinion_association(full_topic)
    print("\n批判性分析:")
    analyzer.analyze_critique()
    print("\n現實意義分析:")
    analyzer.analyze_significance(full_topic)

    # 第二階段：辯論
    print(f"\nStage 2: 立論階段 ({full_topic})")
    debater = DebatorAgent()
    debater.STAGE1_DIR = output_dirs["stage1"]
    debater.OUTPUT_DIR = output_dirs["stage2"]

    print("開始生成觀點")
    debater.generate_argument(full_topic)

    # 第三階段：表達
    print(f"\nStage 3: 表達階段 ({full_topic})")
    expresser = ExpressAgent()
    expresser.STAGE1_DIR = output_dirs["stage1"]
    expresser.STAGE2_DIR = output_dirs["stage2"]
    expresser.OUTPUT_DIR = output_dirs["stage3"]

    print("開始生成金句")
    expresser.build_golden_sentence(full_topic)
    print("\n開始生成自我表達")
    expresser.build_self_expression(full_topic)

    # 讀取最終回答並返回
    response_file = os.path.join(output_dirs["stage3"], "self_expression.txt")
    if os.path.exists(response_file):
        with open(response_file, "r", encoding="utf-8") as f:
            response = f.read().strip()
    else:
        response = "無回答"

    return full_topic, response


if __name__ == "__main__":
    topics = ["人活著的意義是什麼"]

    final_output_file = "output/final_answers.txt"
    os.makedirs("output", exist_ok=True)

    with open(final_output_file, "w", encoding="utf-8") as f:
        for topic in topics:
            print(f"\n開始處理問題: {topic}")
            print("=" * 50)
            question, answer = run_answer_anything_system(topic)
            f.write(f"問題: {question}\n回答: {answer}\n\n")
            print("=" * 50)

    print(f"所有問題及回答已保存至 {final_output_file}")
