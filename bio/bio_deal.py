import json
import re


# 相邻字符加空格 只0加 其余不用
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
                labels_order[label[0]] = fillSpace(label[2].ljust(end - begin, label[2])) if end - begin > 1 else label[
                    2]
            bio = ''
            for key in sorted(labels_order.keys()):
                bio = bio + labels_order[key] + ' '
            bios[json_data['data']] = bio.rstrip()
    return bios


def write_bios(sink_path):
    global key, value
    with open(sink_path, 'w', encoding='utf-8') as fp:
        for key, value in bios.items():
            content = key + ' ' + value
            fp.write(content)
            fp.write('\r')
        fp.close()


if __name__ == '__main__':
    # 需要处理的json文件
    file_path = 'all.json'
    sink_path = 'bios1.txt'
    bios = doccano_to_bio(file_path)
    write_bios(sink_path)
    for key, value in bios.items():
        print(key, value)
