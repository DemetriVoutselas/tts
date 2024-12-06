import re
import nltk

nltk.download('cmudict', quiet=True)

# to disambiguate between phoneme characters and regular characters
# http://www.speech.cs.cmu.edu/cgi-bin/cmudict

phoneme_dict = nltk.corpus.cmudict.dict()   

# https://github.com/r9y9/deepvoice3_pytorch/blob/master/deepvoice3_pytorch/frontend/text/cmudict.py
phoneme_symbols = {
    'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2',
    'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
    'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY',
    'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1',
    'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0',
    'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
    'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
}

phoneme_single_letter_map = {
    'B': '@B',
    'D': '@D',
    'F': '@F',
    'G': '@G',
    'K': '@K',
    'L': '@L',
    'M': '@M',
    'N': '@N',
    'P': '@P',
    'R': '@R',
    'S': '@S',
    'T': '@T',
    'V': '@V',
    'W': '@W',
    'Y': '@Y',
    'Z': '@Z',
}

alphabet_symbols = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!?.')
special_symbols = set('_')

def get_phoneme(text: str):        
    # get rid of symbols to look up phonemes
    text = re.sub(r'[^\w\d\s]', '', text)

    r = []

    for word in text.split():
        word_phonemes = phoneme_dict.get(word.lower())

        if not word_phonemes: 
            return None
        else:
            r.append(
                [phoneme_single_letter_map.get(it) or it for it in word_phonemes[0]]
            )
        
    return r
