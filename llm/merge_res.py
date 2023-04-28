from collections import Counter

import pandas as pd

from data_base_utils import DataBaseUtils

db = DataBaseUtils()
yamp = {
    "空间": ['布局', '实用', '大车', '人口', '方便', '座', '球队', '五座', '空间', '后背箱', '躺着', '第三排', '第二排',
             '二排', '乘坐', '三排', '后备箱',
             '够用',
             '宽敞', '太小', '七座', '后排', '前排', '储物', '全景', '尺寸', '视线', '宽大', '内部空间', '拥挤', '局促',
             '空间', '六座', '超大',
             '满载', '空間'],
    "动力": ['慢', '动力', '20t', '启动', '发动机', '高速', '提速', '变速箱', '加速', '提速', '速度', '百公里', '性能',
             '起步'],
    "操控": ['悬挂', '跑偏', '轻快', '操控', '轴承', '底盆', '保险杠', '车轻', '驾驶', '底盘', '灵活', '倒车', '副驾驶',
             '刹车', '启停', '电动门', '停车',
             '轮胎', '方向盘'
             '后视镜', '行驶', '手动', '减速带', '换挡', '模式', '右跑偏', '右偏'],
    "续航": ['续航', '油耗', '市区', '省油', '电动', '长途', '油箱', '个油', '出游', '排量'],
    "舒适性": ['便利', '颠簸', '路噪', '嘈声', '响声', '噪声', '滤震性', '顿挫', '静音', '座椅', '舒服', '舒适', '胎噪',
               '异响', '避震', '减震', '噪音', '隔音',
               '舒适度', '声音',
               '调节', '按摩', '享受', '安静', '感受', '偏硬', '风噪'],
    "外观": ['一键升窗', '轮毂', '车体', '皮薄', '车型', '车身', '大气', '大灯', '外形', '时尚', '好看', '整体', '天窗',
             '颜色', '车门', '显得', '档次',
             '外观', '线条', '颜值', '安全', '尾灯', '灯光','外观设计'],
    "内饰": ['内饰', '用料', '航空', '质感', '设计', '音响', '豪华', '味道', '做工', '塑料', '车内', '顶配',
             '空调', '通风', '上档次',
             '座位', '车里', '加热', '质量', '粗糙', '真皮', '气味', '耐脏', '异味', '漆面'],
    "性价比": ['性价比', '价', '价格', '价位', '保养', '实用性', '低配', '售后', '销售', '配置', ],
    '科技': ['影像', '导航', '侧滑', '巡航', '自动', '科技', '雷达', '辅助', '系统']
}
res = {}
all_res_zx = []
all_res_fx = []


def merge_count(word, count):
    global res
    for key, value in yamp.items():
        son_res = {}
        if word in value:
            son_res[key] = count
            X, Y = Counter(res), Counter(son_res)
            res = dict(X + Y)
            break


def top_count():
    res.clear()
    top_sql = '''
    select adj,count(*) as count from data_platform_dev.detail_merge where adj!='None' group by adj
    '''
    datas = db.query_sql(top_sql)
    for item in datas:
        adj = item['adj']
        count = item['count']
        merge_count(adj, count)


def write_file(name):
    df = pd.DataFrame(pd.Series(res), columns=['count'])
    df = df.reset_index().rename(columns={'index': 'type'})
    df.to_excel(name + '.xlsx', index=False, header=True)


def sentiment_count(type, all_res):
    res.clear()
    top_sql = '''
           select adj,count(*) as count from data_platform_dev.detail_merge where adj!='None' and score='type_param' group by adj
           '''
    datas = db.query_sql(top_sql.replace('type_param', type))
    for item in datas:
        adj = item['adj']
        count = item['count']
        merge_count(adj, count)
    for key, value in res.items():
        all_res.append((type, key, value))


def detail_merge():
    sql = '''
              select * from data_platform_dev.view_point_temp where adj!='None'
              '''
    datas = db.query_sql(sql)
    detail_res = []
    for item in datas:
        id = item['id']
        adj = item['adj']
        word = item['word']
        input_text = item['contents']
        score = item['score']
        for key, value in yamp.items():
            if adj in value:
                detail_res.append((id, key, adj, word, input_text, score))
                break
    pd.DataFrame(detail_res, columns=['id', 'key', 'adj', 'word', 'contents', 'score']).to_excel('detail_merge.xlsx',
                                                                                                 index=False,
                                                                                                 header=True)


if __name__ == "__main__":
    detail_merge()
    top_count()
    write_file('top')
    sorted_dict = sorted(res.items(), key=lambda x: x[1], reverse=True)
    print(sorted_dict)
    final_res = []
    sentiment_count('正向', all_res_zx)
    sentiment_count('负向', all_res_fx)
    for item in all_res_zx:
        key = item[1]
        value = item[2]
        final_res.append((key, value, res[key] if key in res else 0))
    pd.DataFrame(final_res, columns=['type', '正向', '负向']).to_excel('all_res.xlsx', index=False, header=True)
    print(final_res)
