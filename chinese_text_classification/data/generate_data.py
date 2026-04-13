import json
import random
import os

random.seed(42)

templates = {
    "政治": [
        "国务院召开会议研究部署{}工作，强调要深入贯彻落实科学发展观",
        "全国人大常委会第{}次会议在京举行，审议多项重要法案",
        "外交部发言人就{}问题回答记者提问，表示中方立场",
        "国家领导人出席{}峰会并发表重要讲话",
        "中央经济工作会议在北京召开，部署明年{}工作重点",
        "全国政协第{}次会议在京开幕，各界委员齐聚一堂",
        "国务院发布关于{}的政策文件，推动相关领域发展",
        "全国人大代表团访问{}，加强国际交流合作",
        "中央政治局召开会议分析研究当前{}形势",
        "国务院总理主持召开{}座谈会，听取各方意见",
        "全国人大常委会通过关于{}的决定",
        "外交部就{}事件发表声明，敦促有关方面",
        "国家主席与{}总统通电话，就双边关系交换意见",
        "国务院印发{}发展规划纲要",
        "全国政协召开{}专题协商会",
    ],
    "科技": [
        "我国成功发射{}卫星，标志着航天事业取得新突破",
        "人工智能技术在{}领域取得重大进展",
        "国产芯片研发取得突破，{}性能达到国际先进水平",
        "科学家发现{}新方法，为相关研究开辟新途径",
        "我国自主研发的{}系统正式投入使用",
        "量子计算研究取得重要突破，{}成果发表在国际期刊",
        "新一代{}技术标准正式发布",
        "科技部公布{}重点研发计划项目",
        "我国在{}领域获得重要专利授权",
        "人工智能{}应用落地，助力产业升级",
        "国产{}软件正式发布，打破国外垄断",
        "科研团队攻克{}关键技术难题",
        "我国建成世界领先的{}实验室",
        "大数据技术在{}行业得到广泛应用",
        "5G网络覆盖{}城市，用户规模突破千万",
    ],
    "体育": [
        "中国队在{}比赛中获得冠军，创造历史最好成绩",
        "国足在{}比赛中以比分战胜对手，取得关键胜利",
        "中国选手在{}世锦赛上斩获金牌",
        "NBA常规赛{}队击败对手，取得连胜",
        "中国女排在{}比赛中展现顽强拼搏精神",
        "中超联赛第{}轮比赛结束，积分榜发生变化",
        "中国运动员在{}项目中打破世界纪录",
        "国际{}联合会在北京召开年度会议",
        "CBA联赛{}队主场战胜对手",
        "中国体操队在{}比赛中获得团体冠军",
        "世界杯预选赛{}赛区比赛结果揭晓",
        "中国游泳队在{}比赛中获得多项冠军",
        "田径世锦赛中国选手在{}项目中夺金",
        "乒乓球{}公开赛中国队包揽冠亚军",
        "羽毛球{}超级赛中国选手表现出色",
    ],
    "娱乐": [
        "电影{}票房突破十亿，成为年度黑马",
        "知名歌手{}新专辑发布，引发粉丝热议",
        "电视剧{}收视率创新高，成为现象级作品",
        "综艺节目{}引发观众热烈讨论",
        "明星{}出席某品牌发布会，造型惊艳全场",
        "导演{}新作入围国际电影节主竞赛单元",
        "演员{}凭借{}角色获得最佳男主角奖",
        "音乐节{}站圆满落幕，数万歌迷狂欢",
        "网剧{}播放量破亿，口碑持续走高",
        "综艺节目{}季正式开播，嘉宾阵容强大",
        "电影{}首映礼在北京举行，主创团队亮相",
        "歌手{}巡回演唱会{}站门票秒空",
        "电视剧{}大结局收视率破纪录",
        "明星夫妻{}宣布结婚喜讯",
        "演员{}新剧开机，搭档实力派演员",
    ],
    "财经": [
        "央行宣布下调存款准备金率{}个百分点",
        "A股市场{}板块集体上涨，沪指站上新点位",
        "国家统计局发布数据显示{}经济指标向好",
        "人民币对美元汇率{}走势分析",
        "上市公司{}发布年度财报，净利润增长",
        "国务院出台{}行业减税降费政策",
        "证监会就{}新规征求意见",
        "银行业{}业务规模持续扩大",
        "保险业{}产品创新不断涌现",
        "房地产市场{}城市成交量上升",
        "外贸进出口总额同比增长{}%",
        "创业板注册制改革稳步推进，{}企业成功上市",
        "央行开展{}亿元逆回购操作",
        "基金公司发行{}主题基金受到投资者追捧",
        "银保监会发布{}监管新规",
    ]
}

def generate_text(category):
    template = random.choice(templates[category])
    placeholders = template.count("{}")
    if placeholders > 0:
        filler_words = ["相关", "重要", "重点", "关键", "主要", "核心", "基础", "重大", "突出", "紧迫"]
        fillers = random.sample(filler_words * 3, placeholders)
        text = template.format(*fillers)
    else:
        text = template
    
    additional_sentences = [
        "这是本台记者报道的。",
        "详细情况请关注后续报道。",
        "相关部门正在积极处理中。",
        "专家表示此举意义重大。",
        "社会各界对此反响热烈。",
        "下一步将继续推进相关工作。",
        "这是记者发回的报道。",
        "具体情况正在进一步调查中。",
    ]
    
    if random.random() > 0.5:
        text += random.choice(additional_sentences)
    
    return text

def generate_dataset(num_samples=800):
    data = []
    categories = list(templates.keys())
    samples_per_category = num_samples // len(categories)
    
    for category in categories:
        for _ in range(samples_per_category):
            text = generate_text(category)
            data.append({
                "text": text,
                "label": category
            })
    
    random.shuffle(data)
    return data

def save_dataset(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    print("正在生成中文新闻分类数据集...")
    data = generate_dataset(800)
    
    random.shuffle(data)
    train_end = int(len(data) * 0.7)
    val_end = int(len(data) * 0.85)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dataset(train_data, os.path.join(base_dir, 'train.json'))
    save_dataset(val_data, os.path.join(base_dir, 'val.json'))
    save_dataset(test_data, os.path.join(base_dir, 'test.json'))
    
    print(f"数据集生成完成！")
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")
    print(f"测试集: {len(test_data)} 条")
    
    label_counts = {}
    for item in data:
        label_counts[item['label']] = label_counts.get(item['label'], 0) + 1
    print(f"\n各类别样本数量:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} 条")
