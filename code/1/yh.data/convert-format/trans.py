import sys

def parse_line(line):
    user, item, score, dummy = line.strip().split( )
    user, item, score = map(int, [user, item, score])
    if score == 5:
        score = 1
    else:
        score = 0
    return [user-1, item-1, score]

def gen_instance( user, clicks):
    labels = ",".join(clicks)
    feat = "0:{}:1".format(user)
    return "{} {}\n".format(labels, feat)

def remap( file_name ):
    res_dict = {}
    rf = open( file_name, 'r')
    of = open( file_name + ".remap", 'w')
    for line in rf:
        user, item, score = parse_line(line)
        res_dict.setdefault(user, [])
        res_dict[user].append("{}:{}".format(item, score))
    for user, clicks in res_dict.iteritems():
        of.write(gen_instance(user, clicks))


if __name__ == "__main__":
    file_name = sys.argv[1]
    remap( file_name )
