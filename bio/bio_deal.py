import json
import re


# label.json 初始化
def initLabel():
    with open('label.json', 'r', encoding='utf-8') as fp:
        return json.load(fp)


LABEL_DICT = initLabel()


def fillSpace(bio):
    text_list = re.findall(".{1}", bio)
    return " ".join(text_list)


def doccano_to_bio(file_path):
    bios = {}
    with open(file_path, 'r', encoding='utf-8') as fp:
        for str in fp.readlines():
            json_data = json.loads(str)
            labels = json_data['label']
            labels_order = {}
            for label in labels:
                begin = label[0]
                end = label[1]
                # 将多个字符的key替换为单个字符 为了填充空格 补充字符内容
                label[2] = LABEL_DICT[label[2]]
                labels_order[label[0]] = fillSpace(label[2].ljust(end - begin, label[2])) \
                    .replace('W', 'B-Aspect') \
                    .replace('X', 'I-Aspect') \
                    .replace('Y', 'B-Opinion') \
                    .replace('Z', 'I-Opinion')
            bio = ''
            for key in sorted(labels_order.keys()):
                bio = bio + labels_order[key] + ' '
            bios[json_data['data']] = bio.rstrip()
    return bios


def write_bios(sink_path):
    global key, value
    with open(sink_path, 'w', encoding='utf-8') as fp:
        for key, value in bios.items():
            content = key + '\t' + value
            fp.write(content)
            fp.write('\r')
        fp.close()


'''
1.读取打标签数据
2.循环读取每一行json数据
3.读取json的label数据
4.将label数据的标签通过字典转为特殊字符，为了填充空格，自身字符串的复制
5.之后将转换的标签又转回来
6.将内容与补齐的标签数据一一对应写入文件
'''
if __name__ == '__main__':
    file_path = 'all.json'
    sink_path = 'bios1.txt'
    bios = doccano_to_bio(file_path)
    write_bios(sink_path)
    for key, value in bios.items():
        print(key, value)
