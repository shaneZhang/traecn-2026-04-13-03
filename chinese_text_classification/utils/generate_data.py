# -*- coding: utf-8 -*-
"""
生成模拟中文文本分类数据集
类别：政治、科技、体育、娱乐、财经
"""

import random
import csv
import os

# 定义各类别的关键词和模板
CATEGORIES = {
    "政治": {
        "keywords": ["政府", "会议", "政策", "领导", "国家", "发展", "改革", "建设", "人民", "社会",
                     "制度", "治理", "法治", "民主", "外交", "安全", "稳定", "和谐", "进步", "成就"],
        "templates": [
            "{keyword1}召开重要会议，讨论{keyword2}相关议题，强调{keyword3}的重要性。",
            "{keyword1}发布新政策，旨在促进{keyword2}发展，推动{keyword3}建设。",
            "{keyword1}表示，{keyword2}是{keyword3}的重要保障，必须高度重视。",
            "{keyword1}与{keyword2}举行会谈，就{keyword3}问题深入交换意见。",
            "{keyword1}强调，要坚持{keyword2}，推进{keyword3}不断取得新进展。",
        ]
    },
    "科技": {
        "keywords": ["人工智能", "互联网", "大数据", "云计算", "芯片", "算法", "创新", "研发",
                     "技术", "科技", "智能", "数字", "网络", "软件", "硬件", "系统", "平台", "应用",
                     "突破", "专利"],
        "templates": [
            "{keyword1}技术取得重大突破，{keyword2}领域迎来新发展，{keyword3}应用前景广阔。",
            "某公司发布新一代{keyword1}，采用先进{keyword2}，提升{keyword3}性能。",
            "{keyword1}与{keyword2}深度融合，推动{keyword3}行业数字化转型。",
            "研究团队开发出新型{keyword1}，在{keyword2}方面表现优异，{keyword3}效率大幅提升。",
            "{keyword1}成为热点，{keyword2}企业加大投入，{keyword3}市场竞争激烈。",
        ]
    },
    "体育": {
        "keywords": ["比赛", "冠军", "球队", "运动员", "联赛", "决赛", "进球", "得分", "训练",
                     "教练", "体育", "运动", "竞技", "冠军", "金牌", "世界杯", "奥运会", "赛事",
                     "纪录", "胜利"],
        "templates": [
            "{keyword1}在{keyword2}中表现出色，获得{keyword3}，创造历史最好成绩。",
            "{keyword1}队与{keyword2}队展开激烈对决，最终{keyword3}赢得比赛。",
            "{keyword1}宣布退役，结束辉煌的{keyword2}生涯，{keyword3}界失去传奇。",
            "{keyword1}开始新赛季{keyword2}，目标直指{keyword3}，备战工作有序进行。",
            "{keyword1}在{keyword2}中打破{keyword3}，成为新的纪录保持者。",
        ]
    },
    "娱乐": {
        "keywords": ["电影", "电视剧", "明星", "歌手", "演唱会", "综艺", "票房", "上映", "导演",
                     "演员", "娱乐", "影视", "音乐", "作品", "角色", "剧情", "观众", "粉丝",
                     "获奖", "口碑"],
        "templates": [
            "{keyword1}主演的{keyword2}即将上映，{keyword3}备受期待，预售票房火爆。",
            "{keyword1}在{keyword2}中的精彩表现获得好评，{keyword3}演技获认可。",
            "{keyword1}举办{keyword2}，现场{keyword3}气氛热烈，粉丝热情高涨。",
            "{keyword1}凭借{keyword2}获得{keyword3}，成为当晚最大赢家。",
            "{keyword1}与{keyword2}合作新作品，{keyword3}题材引发关注，制作精良。",
        ]
    },
    "财经": {
        "keywords": ["股市", "经济", "投资", "市场", "企业", "利润", "增长", "股价", "金融",
                     "银行", "财经", "贸易", "消费", "产业", "公司", "上市", "财报", "营收",
                     "风险", "收益"],
        "templates": [
            "{keyword1}今日表现强劲，{keyword2}板块领涨，{keyword3}创新高。",
            "{keyword1}发布季度财报，{keyword2}同比增长，{keyword3}超出预期。",
            "{keyword1}宣布进军{keyword2}市场，{keyword3}战略布局引发关注。",
            "{keyword1}分析指出，{keyword2}将持续增长，{keyword3}前景看好。",
            "{keyword1}与{keyword2}达成合作协议，{keyword3}领域迎来新机遇。",
        ]
    }
}


def generate_text(category, num_samples=200):
    """为指定类别生成文本数据"""
    texts = []
    cat_data = CATEGORIES[category]
    keywords = cat_data["keywords"]
    templates = cat_data["templates"]
    
    for _ in range(num_samples):
        template = random.choice(templates)
        # 随机选择3个不同的关键词
        selected_keywords = random.sample(keywords, 3)
        text = template.format(
            keyword1=selected_keywords[0],
            keyword2=selected_keywords[1],
            keyword3=selected_keywords[2]
        )
        texts.append(text)
    
    return texts


def generate_dataset(output_path, samples_per_category=200):
    """生成完整的数据集并保存为CSV"""
    all_data = []
    
    for category in CATEGORIES.keys():
        texts = generate_text(category, samples_per_category)
        for text in texts:
            all_data.append({"text": text, "label": category})
    
    # 打乱数据
    random.shuffle(all_data)
    
    # 保存到CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'label'])
        writer.writeheader()
        writer.writerows(all_data)
    
    print(f"数据集已生成: {output_path}")
    print(f"总样本数: {len(all_data)}")
    for category in CATEGORIES.keys():
        count = sum(1 for d in all_data if d['label'] == category)
        print(f"  {category}: {count}条")


if __name__ == "__main__":
    # 生成1000条数据（每个类别200条）
    output_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "news_data.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    generate_dataset(output_file, samples_per_category=200)
