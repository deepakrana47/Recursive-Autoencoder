import re

def division(match):
    return str(float(match.group(1))/float(match.group(2)))

def line_processing(line):#, fstopwds= None, fabbwds=None):
    '''
    Line pre-processing is performed
    :param line: line input
    :return:
        ptext: processed text
    '''
    # fstopwds =  Global.fstopwds if fstopwds is None else fstopwds
    # fabbwds = "/media/zero/41FF48D81730BD9B/Final_Thesies/Coding/final_STS_machine_2/extra/abbrev1.txt"
    # fstopwds = "/media/zero/41FF48D81730BD9B/Final_Thesies/Coding/final_STS_machine_2/extra/stopwords2.txt"
    # fabbwds = "/media/zero/41FF48D81730BD9B/Final_Thesies/Coding/final_STS_machine_2/extra/abbrev1.txt"
    ptext = ''

    # stop word removal and abbreviation formatting
    ptext = line.strip('\n|\r| ')
    ptext = ptext.strip('.')
    ptext = re.sub(r'(\.{2,}|\.[ ]*"|\.[ ]*\')$', '', ptext)
    ptext = re.sub(r'(\.{2,}|\."|\.\')', ' ', ptext)
    ptext = ptext.lower()

    # pre symbol removel
    ptext = re.sub(r'\-|,"|\;|\'s |\(|\)|\[|\]|\{|\}', r' ', ptext)
    ptext = re.sub(r'\xe2\x80\x99s|\xc2\xb4s|\'|"|,|\~|\+|\&', r'', ptext)
    ptext = re.sub(r'( |^)([0-9]+)(st|th|ed|s)( |$)', r'\1\2\4', ptext)

    # wor processing
    ptext = re.sub(r'( |^)\<[^\>]+\>([ ]*)', r' ', ptext)
    # ptext = re.sub(r'\([^\)]+\)',r' ',ptext)

    ptext = re.sub(r'(\$|a\$|\$a|\$us|us\$)([0-9]+[\.]{0,1}[0-9]*)bn', r' \2 billion dollar ', ptext)
    ptext = re.sub(r'(\$|a\$|\$a|\$us|us\$)([0-9]+[\.]{0,1}[0-9]*)m', r' \2 million dollar ', ptext)
    ptext = re.sub(r'(\$|a\$|\$a|\$us|us\$)([0-9]+[\.]{0,1}[0-9]*)', r' \2 dollar ', ptext)

    ptext = re.sub(r'(\xc2\xa3|\xe2\x82\xac)([0-9]+[\.]{0,1}[0-9]*)bn', r' \2 billion pound ', ptext)
    ptext = re.sub(r'(\xc2\xa3|\xe2\x82\xac)([0-9]+[\.]{0,1}[0-9]*)m', r' \2 million pound ', ptext)
    ptext = re.sub(r'(\xc2\xa3|\xe2\x82\xac)([0-9]+[\.]{0,1}[0-9]*)', r' \2 pound ', ptext)

    # ptext = re.sub(r'( |^)([0-9]+)-([0-9]+)/([0-9]+)p( |$)',r'\1\2 \3 \4 ',ptext)
    ptext = re.sub(r'( |^)([0-9]+[\.]{0,1}[0-9]*)%( |$)', r'\1\2 percent\3', ptext)

    ptext = re.sub(r'( |^)([0-9]+[\.]{0,1}[0-9]*)(km/h|kmph|km/hr|kmphr)', r'\1\2 kilometer per hour', ptext)

    ptext = re.sub(r'https\://|http\://', r'', ptext)
    ptext = re.sub(r'( |^)([a-z]{2,})\.([^\. ]{2,})[ ]*\.([a-z]{2,})', r'\1\3 \4', ptext)
    ptext = re.sub(r'( |^)([^\. ]{2,})\.([a-z]+)', r'\1\2 \3', ptext)

    # ptext = re.sub(r'( |^)v([0-9]+[\.]{0,1}[0-9]*) ',r'\1v \2 ',ptext)

    ptext = re.sub(r'( |^|\()([0-9]+)([a-z]+)( |$|\))', r'\1\2 \3\4', ptext)
    ptext = re.sub(r'( |^)([0-9]+\.[0-9]*)([a-z]+)( |$)', r'\1\2 \3\4', ptext)

    # ptext = re.sub(r' (st|th|ed) ', r' ', ptext)


    # post symbol removel
    ptext = re.sub(r'[^\x00-\x7F]+', r' ', ptext)
    regexp1 = re.compile(
        r'\!\!|\?\!|\?\?|\!\?|`|``|\=|\'\'|\-lrb\-|\-rrb\-|\-lsb\-|\-rsb\-|\'|\:|\?|\<|\>|\%|\$|\@|\!|\^|\#|\*|/|_')
    ptext = re.sub(regexp1, r' ', ptext)
    ptext = re.sub(r'( |^)\.([^\.]+)', r'\1\2', ptext)
    ptext = re.sub(r'( |^)([^\. ])\.([^\. ])\.([^\. ])\.( |$)', r'\1\2\3\4\5', ptext)
    ptext = re.sub(r'( |^)([^\. ])\.([^\. ])\.([^\. ])\.([^\. ])\.( |$)', r'\1\2\3\4\5\6', ptext)
    # ptext=re.sub('([a-z]+\.) ',r'\1 ',ptext)
    # ptext = re.sub(r'( |^)([0-9]+[\.]{0,1}[0-9]*)([a-z]+)( |$)', r'\1\2 \3\4', ptext)


    ## stop word and abbriavation processing
    # abbwds = dict([i.strip('\n').split(',') for i in open(fabbwds, 'r')])
    # ptext = ' '.join([abbwds[i.lower()] if i.lower() in abbwds else i for i in ptext.split(' ')])
    # stopwds = [i.strip('\n') for i in open(fstopwds, 'r')]
    # ptext = ' '.join(['' if i in stopwds else i for i in ptext.split(' ')])

    ptext = re.sub('[ \t]{2,}', r' ', ptext)
    ptext = ptext.strip(' ')
    return ptext

def text_preprocessing(text):
    '''
        Text pre-processing is performed
        :param text: line data
        :return:
            ptext: processed line
        '''
    text = re.sub(r'\{\{[^\}\{]*\}\}', '', text)

    text = re.sub(r"'[']*([^']+)[']*'", r'\1', text)
    # text = re.sub(r"\([^\(\)]+\)", '', text)
    text = re.sub(r"(https\://|http\://)[^ ]+", '', text)
    text = re.sub(r'<ref[^>]*>[.]*</ref>|<ref[^>/]*/>', r'', text)

    text = re.sub(r"\<[^\>\<]+\>", '', text)
    text = re.sub(r"\=\=See also\=\=[\n\{\}a-zA-Z \|*'\[\]:\(\)0-9-;.\<\>\=/\?,\t_&\#\"\!\+]*", '', text)
    text = re.sub(r"\=[\=]+[^\=]+[\=]+\=", r'', text)
    # text = re.sub(r"\*[\:\-;\<\>\{\}\=/ '\[\]a-zA-Z\|\)\(0-9,\.\!_&\?]+",r'',text)
    text = re.sub(r"[^. ]+\.com", r'', text)
    text = re.sub(r"\[\[([^|\]]+)[^\]]*\]\]", r"\1", text)
    text = re.sub(r"\*[^*\n]+", r'', text)
    text = re.sub(r'[^\n\:]+\:', r"", text)
    text = re.sub(r'[ ]*[\n]+', '\n', text)

    text = re.sub(r'\n[ ]*[\|\:;\*\#\!\{\'][^\n]*', '', text)
    text = re.sub(r'\n[ ]*&nbsp[^\n]*', '', text)
    text = re.sub(r'\n[^\n]*\]\]', '', text)
    text = re.sub(r"\[[^\[\]]+\]", '', text)
    return text

if __name__ == '__main__':
    lines = ['s.& p. 500 slipped 12.27 points or 1.2 percent to 981.73',
'suppose confirmed one thing british public consistently dull &lt no offence robbie pleeease thousand better songs formulated cheesy pop song kids',
'pratt &whitney said 75 per cent of engine equipment outsourced to europe final assembly in germany',
'pg&e corporation shares up 39 cents or 2.6 percent 15.59 dollar on new york stock exchange on tuesday',
'feature approval of book publishers puts 33 million pages of searchable text disposal of amazon.com shoppers',
'mcarthur told internetnews.com price declines moderated and remained below 30 percent from previous year last two quarters',
'personally id like to see cartoons transformers thundercats and m.a.s.k. get full hollywood remakes']
    for line in lines:
        print line
        print line_processing(line,fstopwds = "/media/zero/41FF48D81730BD9B/Final_Thesies/Coding/final_STS_machine_2/extra/stopwords2.txt")
        raw_input()