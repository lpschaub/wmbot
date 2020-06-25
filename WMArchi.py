import numpy as np
import glob


# Parse data
def parse_babi_task(data_files, dictionary, include_question):
    """ Parse bAbI data.
    Args:
       data_files (list): a list of data file's paths.
       dictionary (dict): word's dictionary
       include_question (bool): whether count question toward input sentence.
    Returns:
        A tuple of (story, questions, qstory):
            story (3-D array)
                [position of word in sentence, sentence index, story index] = index of word in dictionary
            questions (2-D array)
                [0-9, question index], in which the first component is encoded as follows:
                    0 - story index
                    1 - index of the last sentence before the question
                    2 - index of the answer word in dictionary
                    3 to 13 - indices of supporting sentence
                    14 - line index
            qstory (2-D array) question's indices within a story
                [index of word in question, question index] = index of word in dictionary
    """
    # Try to reserve spaces beforehand (large matrices for both 1k and 10k data sets)
    # maximum number of words in sentence = 20
    story = np.zeros((20, 500, len(data_files) * 3500), np.int16)
    questions = np.zeros((14, len(data_files) * 10000), np.int16)
    qstory = np.zeros((20, len(data_files) * 10000), np.int16)

    # NOTE: question's indices are not reset when going through a new story
    story_idx, question_idx, sentence_idx, max_words, max_sentences = -1, -1, -1, 0, 0

    # Mapping line number (within a story) to sentence's index (to support the flag include_question)
    mapping = None

    for fp in data_files:
        with open(fp) as f:
            for line_idx, line in enumerate(f):
                line = line.rstrip().lower()
                words = line.split()

                # Story begins
                if words[0] == '1':
                    story_idx += 1
                    sentence_idx = -1
                    mapping = []

                # FIXME: This condition makes the code more fragile!
                if '?' not in line:
                    is_question = False
                    sentence_idx += 1
                else:
                    is_question = True
                    question_idx += 1
                    questions[0, question_idx] = story_idx
                    questions[1, question_idx] = sentence_idx
                    if include_question:
                        sentence_idx += 1

                mapping.append(sentence_idx)

                # Skip substory index
                for k in range(1, len(words)):
                    w = words[k]

                    if w.endswith('.') or w.endswith('?'):
                        w = w[:-1]

                    if w not in dictionary:
                        dictionary[w] = len(dictionary)

                    if max_words < k:
                        max_words = k

                    if not is_question:
                        story[k - 1, sentence_idx, story_idx] = dictionary[w]
                    else:
                        qstory[k - 1, question_idx] = dictionary[w]
                        if include_question:
                            story[k - 1, sentence_idx, story_idx] = dictionary[w]

                        # NOTE: Punctuation is already removed from w
                        if words[k].endswith('?'):
                            answer = words[k + 1]
                            if answer not in dictionary:
                                dictionary[answer] = len(dictionary)

                            questions[2, question_idx] = dictionary[answer]

                            # Indices of supporting sentences
                            for h in range(k + 2, len(words)):
                                questions[1 + h - k, question_idx] = mapping[int(words[h]) - 1]

                            questions[-1, question_idx] = line_idx
                            break

                if max_sentences < sentence_idx + 1:
                    max_sentences = sentence_idx + 1

    story = story[:max_words, :max_sentences, :(story_idx + 1)]
    questions = questions[:, :(question_idx + 1)]
    qstory = qstory[:max_words, :(question_idx + 1)]

    return story, questions, qstory


data_dir = "data/tasks_1-20_v1-2/en"
task_id = 6
train_files = glob.glob('%s/qa%d_*_train.txt' % (data_dir, task_id))
test_files = glob.glob('%s/qa%d_*_test.txt' % (data_dir, task_id))

dictionary = {"nil": 0}
train_story, train_questions, train_qstory = parse_babi_task(train_files, dictionary, False)
test_story, test_questions, test_qstory = parse_babi_task(test_files, dictionary, False)
train_config = {
    "init_lrate": 0.01,
    "max_grad_norm": 40,
    "in_dim": 20,
    "out_dim": 20,
    "sz": min(50, train_story.shape[1]),  # number of sentences
    "voc_sz": len(dictionary),
    "bsz": 16,
    "max_words": len(train_story),
    "weight": None
}


class MemNN(object):
    def __init__(self, X, B):
        self.X = X  # X = {x1...xi} tokens historique dialogue
        self.B = B  # B = {b1...bi} tuples KB
        self.sz = train_config["sz"]
        self.voc_sz = train_config["voc_sz"]
        self.in_dim = train_config["in_dim"]
        self.out_dim = train_config["out_dim"]

        # TODO: Mark self.nil_word and self.data as None since they will be overriden eventually
        # In build.model.py, memory[i].nil_word = dictionary['nil']"
        self.nil_word = train_config["voc_sz"]
        self.config = train_config
        self.data = np.zeros((self.sz, train_config["bsz"]), np.float32)

        self.emb_query = None
        self.emb_out = None
        self.mod_query = None
        self.mod_out = None
        self.probs = None

    def init_query_module(self):
        self.emb_query = LookupTable(self.voc_sz, self.in_dim)
        p = Parallel()
        p.add(self.emb_query)
        p.add(Identity())

        self.mod_query = Sequential()
        self.mod_query.add(p)
        self.mod_query.add(MatVecProd(True))
        self.mod_query.add(Softmax())

    def init_output_module(self):
        self.emb_out = LookupTable(self.voc_sz, self.out_dim)
        p = Parallel()
        p.add(self.emb_out)
        p.add(Identity())

        self.mod_out = Sequential()
        self.mod_out.add(p)
        self.mod_out.add(MatVecProd(False))

    def reset(self):
        self.data[:] = self.nil_word

    def put(self, data_row):
        self.data[1:, :] = self.data[:-1, :]  # shift rows down
        self.data[0, :] = data_row  # add the new data row on top

    def fprop(self, input_data):
        self.probs = self.mod_query.fprop([self.data, input_data])
        self.output = self.mod_out.fprop([self.data, self.probs])
        return self.output

    def bprop(self, input_data, grad_output):
        g1 = self.mod_out.bprop([self.data, self.probs], grad_output)
        g2 = self.mod_query.bprop([self.data, input_data], g1[1])
        self.grad_input = g2[1]
        return self.grad_input

    def update(self, params):
        self.mod_out.update(params)
        self.mod_query.update(params)
        self.emb_out.weight.D[:, self.nil_word] = 0

    def share(self, m):
        pass


class Encoder(object):
    def __init__(self):
        self.memNN = MemNN()


class WMN2seq(object):
    def __init__(self):
        self.encoder = Encoder()
