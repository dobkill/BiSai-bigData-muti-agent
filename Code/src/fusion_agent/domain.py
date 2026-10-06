"""赛题方向口径、平台别名与文本标准化，不把任何来源当作真值。"""
import re

REGIONS = {
    "东海": "东海 东海北部 东海中部 东海南部 钓鱼岛海域 赤尾屿海域 上海外海 长江口外海 舟山外海 宁波外海 温州外海 东海油气田海域 冲绳海槽 苏岩礁附近海域".split(),
    "南海": "南海 南海北部 南海中部 南海南部 北部湾 琼州海峡 海南岛东南海域 珠江口外海 东沙群岛海域 西沙群岛海域 中沙群岛海域 南沙群岛海域 黄岩岛海域 美济礁附近海域 仁爱礁附近海域 万安滩海域 礼乐滩海域 曾母暗沙海域 永暑礁附近海域 渚碧礁附近海域 北康暗沙海域 南康暗沙海域 纳土纳群岛北部海域".split(),
    "黄海": "黄海 北黄海 黄海中部 南黄海 青岛外海 成山头外海 海州湾外海 连云港外海 济州岛西部海域".split(),
    "其他": "台湾海峡 巴士海峡 巴林塘海峡 宫古水道 对马海峡 菲律宾海 阿拉伯海 安达曼海 孟加拉湾 印度洋 北大西洋 南大西洋 大西洋 北太平洋 南太平洋 西太平洋 西北太平洋 中太平洋 太平洋 地中海 黑海 红海 亚丁湾 莫桑比克海峡 塔斯曼海 白令海 巴伦支海 鄂霍次克海 波斯湾 渤海 关岛附近海空域 夏威夷附近海空域".split(),
}
REGION_DIRECTION = {r: d for d, rs in REGIONS.items() for r in rs}
REGION_ALIASES = {
    "钓鱼岛": "钓鱼岛海域", "赤尾屿": "赤尾屿海域", "东海油气田": "东海油气田海域",
    "苏岩礁": "苏岩礁附近海域", "海南岛东南": "海南岛东南海域",
    "东沙群岛": "东沙群岛海域", "西沙群岛": "西沙群岛海域", "中沙群岛": "中沙群岛海域",
    "南沙群岛": "南沙群岛海域", "黄岩岛": "黄岩岛海域", "美济礁": "美济礁附近海域",
    "仁爱礁": "仁爱礁附近海域", "万安滩": "万安滩海域", "礼乐滩": "礼乐滩海域",
    "曾母暗沙": "曾母暗沙海域", "永暑礁": "永暑礁附近海域", "渚碧礁": "渚碧礁附近海域",
    "北康暗沙": "北康暗沙海域", "南康暗沙": "南康暗沙海域", "纳土纳群岛北部": "纳土纳群岛北部海域",
    "东海北部海域": "东海北部", "东海中部海域": "东海中部", "东海南部海域": "东海南部",
    "南海北部海域": "南海北部", "南海中部海域": "南海中部", "南海南部海域": "南海南部",
    "南黄海海域": "南黄海", "北黄海海域": "北黄海", "关岛附近": "关岛附近海空域",
    "中太平洋海域": "中太平洋", "夏威夷附近海域": "夏威夷附近海空域",
}
COUNTRY_ALIASES = {
    "中国": ["中国", "解放军", "我军", "中方"], "美国": ["美国", "美军", "美海军", "美空军", "美方"],
    "俄罗斯": ["俄罗斯", "俄军", "俄海军", "俄空军", "俄方"], "日本": ["日本", "日军", "日方"],
    "韩国": ["韩国", "韩军", "韩方"], "台湾": ["台湾", "台军", "台方"],
    **{c: [c] for c in ["英国", "法国", "德国", "意大利", "西班牙", "印度", "澳大利亚", "加拿大", "新加坡", "马来西亚", "菲律宾", "越南", "新西兰", "印度尼西亚", "泰国"]},
}
TASKS = ["对海突击训练", "实弹射击训练", "远程轰炸训练", "电子对抗训练", "情报监视侦察", "海上侦察巡逻", "编队警戒部署", "两栖作战训练", "海上补给训练", "联合搜救训练", "反潜巡逻", "联合巡逻", "联合巡航", "战备警巡", "战备巡逻", "空中警戒", "电子侦察", "远海训练", "搜救训练", "海上巡逻", "自由航行", "例行部署", "友好访问", "海上护航", "海上监视侦察", "空中加油支援", "舰载机起降训练", "远程兵力投送", "防空反导训练", "预警指挥"]
TASK_ALIASES = {"战备巡航": "战备巡逻", "情报监视与侦察": "情报监视侦察", "海上护航行动": "海上护航", "海上护航": "海上护航"}
PLATFORM_ALIASES = {
    "阵风M": ["阵风M", "阵风”M", "阵风“M", "阵风 M"],
    "台风": ["台风"], "无敌级": ["无敌级", "无敌号"],
    "加尔各答级": ["加尔各答级", "加尔各答号"], "维多利亚级": ["维多利亚级"],
    "阿利·伯克级": ["阿利·伯克", "阿利伯克", "伯克级"], "摩耶级": ["摩耶级", "摩耶号"],
    "秋月级": ["秋月级", "秋月号"], "安扎克级": ["安扎克级", "安扎克号"],
    "守护级": ["守护级", "守护”级", "守护“级"], "金斯顿级": ["金斯顿级", "金斯顿号"],
    "山东号": ["山东号", "山东舰"], "辽宁号": ["辽宁号", "辽宁舰"],
    "罗斯福号": ["罗斯福号", "罗斯福”号", "罗斯福"], "里根号": ["里根号", "里根"],
    "尼米兹号": ["尼米兹号", "尼米兹"], "卡尔·文森号": ["卡尔·文森", "卡尔文森"],
    "维克兰特号": ["维克兰特", "维克兰特号"], "加富尔号": ["加富尔", "加富尔号"],
    "瓦良格号": ["瓦良格号", "瓦良格"], "莱蒙多·蒙特库科利号": ["莱蒙多·蒙特库科利", "蒙特库科利"],
    "东姑阿都拉曼号": ["东姑阿都拉曼"], "安东尼奥·卢纳号": ["安东尼奥·卢纳", "卢纳号"],
    "奥特亚罗瓦号": ["奥特亚罗瓦"], "特马纳号": ["特马纳"],
    "幻影2000-5": ["幻影2000-5", "幻影2000"], "大西洋2": ["大西洋”2", "大西洋2"],
    "梅林HM2": ["梅林”HM2", "梅林HM2"], "经国号": ["经国号"], "福克50": ["福克50"],
}
MODEL_RE = re.compile(r"F/A-?\d+[A-Z\d]*(?:/[A-Z\d]+)?|(?:歼|轰|运|直|空警|无侦|攻击|苏|图|伊尔|米格)-?\d+[A-Z\d]*|(?<![A-Za-z0-9])[A-Z]{1,6}-?\d+[A-Z\d]*(?:/[A-Z\d]+)?|(?<!\d)\d{2,3}[A-Z]?型", re.I)
GENERIC_TYPES = ["两栖攻击舰", "船坞登陆舰", "补给舰", "航空母舰", "航母", "驱逐舰", "巡洋舰", "护卫舰", "巡逻舰", "潜艇", "预警机", "运输机", "加油机", "轰炸机", "直升机", "无人机", "战斗机", "巡逻机"]


def clean(value: str) -> str:
    return re.sub(r"[\s\u3000\"'“”‘’（）()]", "", value).upper()


def countries(text: str) -> list[str]:
    # 台湾海峡是区域名，不提供主体国家证据。
    subject_text = text.replace("台湾海峡", "")
    return [c for c, aliases in COUNTRY_ALIASES.items() if any(a in subject_text for a in aliases)]


def regions(text: str) -> list[str]:
    tokens = {**{r: r for r in REGION_DIRECTION}, **REGION_ALIASES}
    found = []
    spans = []
    for token in sorted(tokens, key=len, reverse=True):
        for match in re.finditer(re.escape(token), text):
            if any(a <= match.start() and match.end() <= b for a, b in spans):
                continue
            spans.append(match.span())
            found.append((match.start(), tokens[token]))
    return list(dict.fromkeys(r for _, r in sorted(found)))


def tasks(text: str) -> list[str]:
    result = []
    spans = []
    for name in sorted(set(TASKS) | set(TASK_ALIASES), key=len, reverse=True):
        for match in re.finditer(re.escape(name), text):
            if any(a <= match.start() and match.end() <= b for a, b in spans):
                continue
            # 平台类别描述不等于正在执行该任务。
            if re.match(r"(?:机|飞机|船|舰)", text[match.end():]):
                continue
            spans.append(match.span())
            canonical = TASK_ALIASES.get(name, name)
            if canonical not in result:
                result.append(canonical)
    return result


def platforms(text: str, aliases: dict[str, list[str]] | None = None) -> list[str]:
    result = []
    for name, variants in {**PLATFORM_ALIASES, **(aliases or {})}.items():
        if any(clean(a) in clean(text) for a in variants):
            result.append(name)
    for match in MODEL_RE.finditer(text):
        token = match.group().upper().replace("型", "")
        # 型号常见连字符归一化；保留字母后缀区分不同平台。
        token = re.sub(r"^(歼|轰|运|直|空警|无侦|攻击|苏|图|伊尔|米格)(\d)", r"\1-\2", token)
        if token not in result and not (token == "HM2" and "梅林HM2" in result):
            result.append(token)
    if not result:
        for kind in GENERIC_TYPES:
            if kind in text:
                result.append("generic:" + ("航母" if kind == "航空母舰" else kind))
                break
    return result


def platform_catalog(labels: list[str]) -> dict[str, list[str]]:
    """从当前输入建立名称词典，词典只帮助识别名称，不核定平台事实。"""
    result = {}
    for label in labels:
        match = re.match(r"(.+?(?:号|级))", label)
        known = [p for p in platforms(label) if not p.startswith("generic:")]
        if match and not MODEL_RE.search(match.group(1)) and not known:
            name = clean(match.group(1))
            if name not in PLATFORM_ALIASES:
                result[name] = [match.group(1)]
    return result


def platform_kinds(text: str) -> list[str]:
    text = text.replace("直升机母舰", "母舰")
    result = []
    if any(word in text for word in ["战斗机", "战机", "轰炸机", "巡逻机", "反潜机", "预警机", "运输机", "加油机", "直升机", "无人机", "侦察机", "飞机", "舰载机", "电子战机"]):
        result.append("air")
    if any(word in text for word in ["航母", "母舰", "驱逐舰", "护卫舰", "巡洋舰", "巡逻舰", "补给舰", "攻击舰", "登陆舰", "潜艇", "侦察船", "运输舰", "濒海战斗舰"]):
        result.append("ship")
    return result


def region_compatible(a: str, b: str) -> bool:
    if a == b:
        return True
    # 泛称海域与同方向子海域兼容；两个不同子海域不可互相支持。
    return ((a in REGIONS and REGION_DIRECTION.get(b) == a) or
            (b in REGIONS and REGION_DIRECTION.get(a) == b))


def coordinate_direction(lon: float, lat: float, margin: float) -> str | None:
    # 启发式包围框，并非法定边界或精确子海域。边界附近拒绝推断。
    boxes = [("黄海", 119, 126.5, 32, 40), ("东海", 120, 130, 24, 32), ("南海", 105, 121, 3, 24)]
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        return None
    for direction, x1, x2, y1, y2 in boxes:
        if x1 + margin < lon < x2 - margin and y1 + margin < lat < y2 - margin:
            # 通行区域无法仅由粗包围框确定方向。
            if (118 < lon < 121 and 23 < lat < 26) or (120 < lon < 123 and 20 < lat < 22):
                return None
            return direction
    if any(x1-margin <= lon <= x2+margin and y1-margin <= lat <= y2+margin for _,x1,x2,y1,y2 in boxes):
        return None
    return "其他"

