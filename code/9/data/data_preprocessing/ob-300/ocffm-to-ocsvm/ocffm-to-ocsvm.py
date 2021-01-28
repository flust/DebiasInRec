import sys

def build_dict(item, context_list):
    counter = 1
    item_dict = {}
    context_dict = {}

    rf = open(item, 'r')
    for line in rf:
        feats = line.strip().split()
        for tk in feats:
            f, idx, val = tk.split(':')
            key = (f, idx)
            item_dict.setdefault(key, -1)
            if item_dict[key] == -1:
                item_dict[key] = counter
                counter += 1
    rf.close()

    for context in context_list:
        rf = open(context, 'r')
        for line in rf:
            toks = line.strip().split()
            feats = toks[1:]
            for tk in feats:
                f, idx, val = tk.split(':')
                key = (f, idx)
                context_dict.setdefault(key, -1)
                if context_dict[key] == -1:
                    context_dict[key] = counter
                    counter += 1

    return (item_dict, context_dict)

def convert_feature( feature_list, convert_dict):
    converted_feature_list = []
    for feat in feature_list:
        f, idx, val = feat.split(':')
        key = (f, idx)
        converted_feature_list.append("{}:{}".format(convert_dict[key], val))
    return converted_feature_list

def convert_item( item, item_dict):
    rf = open(item, 'r')
    of = open(item.replace("ffm", "svm"), 'w')

    for line in rf:
        tokens = line.strip().split()
        features = tokens
        converted_features = convert_feature(features, item_dict)
        of.write("{}\n".format(" ".join(converted_features)))

def convert_context( context, context_dict):
    rf = open(context, 'r')
    of = open(context.replace("ffm", "svm"), 'w')

    for line in rf:
        tokens = line.strip().split()
        label = tokens[0]
        features = tokens[1:]
        converted_features = convert_feature(features, context_dict)
        of.write("{} {}\n".format(label, " ".join(converted_features)))

if __name__ == '__main__':
    item = sys.argv[1]
    feature_80 = sys.argv[2]
    feature_10 = sys.argv[3]
    context_list = sys.argv[3:]
    print("item file: {}\ncontext files: [{}]".format(item, ','.join(context_list)))

    print("Start build dictionary item: {} context: {}.".format(item, [feature_80, feature_10]))
    item_dict, context_dict = build_dict(item, [feature_80, feature_10])

    convert_item( item, item_dict)
    for context in context_list:
        convert_context(context, context_dict)
