import sys

if __name__ == '__main__':
    score_dict = {}
    l = sys.argv[1]
    num_split = 5
    for i in range(num_split):
        rf = open( "yh.all.tr.ps.{}.te.{}.k80.std".format(i, l), 'r')
        for i, line in enumerate(rf):
            if i == 0 or i == 201:
                continue
            tr_loss, va_loss = map(float, line.strip().split())
            score_dict.setdefault( i, 0.0)
            score_dict[i] += va_loss / num_split


    of = open( "yh.all.tr.ps.te.{}.k80.std.merge".format(l), 'w')
    for key, va_avg in score_dict.iteritems():
        of.write(  "{}\t{}\n".format(key,va_avg) )
