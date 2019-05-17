import pickle
import numpy as np
import tensorflow as tf
from random import randrange, sample
from tensorflow import keras
import tensorflow_hub as hub
from datetime import datetime
import bert.tokenization as btk

BERT_URL = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
BERT_MODULE = hub.Module(BERT_URL, trainable=True)

sess = tf.Session()


def import_data(filepath):
    with open(filepath, 'rb') as file:
        return pickle.load(file)


def create_balanced_train_test_sets(dataset, test_size=100, batches=None):
    data_per_emoji = {emoji: [] for data, emoji in dataset}
    for text, emoji in dataset:
        data_per_emoji[emoji].append(text)

    test = {emoji: [textlist.pop(randrange(len(textlist))) for _ in range(test_size)]
            for emoji, textlist in data_per_emoji.items()}

    lengths_per_emoji = {emoji: len(text) for emoji, text in data_per_emoji.items()}
    maxlength = max(lengths_per_emoji.values())

    for emoji in data_per_emoji.keys():
        data_per_emoji[emoji] = data_per_emoji[emoji] * (maxlength//lengths_per_emoji[emoji])
        remaining_len = maxlength - len(data_per_emoji[emoji])
        data_per_emoji[emoji] += sample(data_per_emoji[emoji], remaining_len)

    if batches:
        batchsize= int(maxlength//batches)
        train_flat = []
        for i in range(batches):
            train_flat.append([(text, emoji) for emoji, textlist in data_per_emoji.items()
                               for text in textlist[i*batchsize:(i+1)*batchsize]])

    else:
        train_flat = [(text, emoji) for emoji, textlist in data_per_emoji.items() for text in textlist]

    test_flat = [(text, emoji) for emoji, textlist in test.items() for text in textlist]

    return train_flat, test_flat


def split_x_y(dataset):
    x = [x for x, y in dataset]
    y = [y for x, y in dataset]
    return x, y


def get_labelmap(ylabels):
    emojiset = set(ylabels)
    emojidict = {}
    i = 0
    for emoji in emojiset:
        emojidict[emoji] = i
        i += 1

    return emojidict


def get_tokenizer():
    tokenization_info = BERT_MODULE(signature='tokenization_info', as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [
            tokenization_info["vocab_file"],
            tokenization_info["do_lower_case"],
        ]
    )
    return btk.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


def tokenize_single_input(text, tokenizer: btk.FullTokenizer, max_input_length):
    tokens = ['[CLS]']
    tokens += tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_masks = [1]*len(token_ids)
    segment_ids = [0]*max_input_length

    if len(token_ids) > max_input_length:
        raise ValueError('The input is %i while the maximum input can be only %i.' % (len(token_ids), max_input_length))

    remaining = max_input_length-len(token_ids)
    token_ids += [0] * remaining
    token_masks += [0] * remaining

    return token_ids, token_masks, segment_ids


def tokenize_inputlist(textdata, tokenizer: btk.FullTokenizer, max_input_length=80):
    token_ids_list = []
    token_masks_list = []
    segment_ids_list = []

    for text in textdata:
        token_id, token_mask, segment_id = tokenize_single_input(text, tokenizer, max_input_length)
        token_ids_list.append(token_id)
        token_masks_list.append(token_mask)
        segment_ids_list.append(segment_id)

    return np.array(token_ids_list), np.array(token_masks_list), np.array(segment_ids_list)


def tokenize_labels(labels, labelmap):
    labelnumbers = [labelmap[label] for label in labels]
    return keras.utils.to_categorical(labelnumbers)


def build_model(max_sequence_length=80):
    # inputs
    id_input = keras.layers.Input(shape=(max_sequence_length,), dtype="int32")
    mask_input = keras.layers.Input(shape=(max_sequence_length,), dtype="int32")
    segment_input = keras.layers.Input(shape=(max_sequence_length,), dtype="int32")

    # layers
    bertoutput = BERTLayer()([id_input, mask_input, segment_input])
    dense1output = keras.layers.Dense(256, 'relu')(bertoutput)
    modeloutput = keras.layers.Dense(60, 'softmax')(dense1output)

    model = keras.models.Model(inputs=[id_input, mask_input, segment_input], outputs=modeloutput)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    return model


def initialize_variables(session):
    session.run(tf.local_variables_initializer())
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    keras.backend.set_session(session)


class BERTLayer(tf.layers.Layer):
    def __init__(self, n_fine_tune_layers=3, **kwargs):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        super(BERTLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            BERT_URL,
            trainable=self.trainable,
            name="{}_module".format(self.name)
        )

        trainable_vars = self.bert.variables

        # Remove unused layers
        trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]

        # Select how many layers to fine tune
        trainable_vars = trainable_vars[-self.n_fine_tune_layers:]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BERTLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inputs = [keras.backend.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
            "pooled_output"
        ]
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)


def main_batched(batches):
    print("%s:      importing data" % datetime.now())
    data = import_data('full_datapickle')
    train, test = create_balanced_train_test_sets(data, batches=batches)
    x_train = []
    y_train = []
    for batch in train:
        x, y = split_x_y(batch)
        x_train.append(x)
        y_train.append(y)

    x_test, y_test = split_x_y(test)

    y_train_merged = [value for ylist in y_train for value in ylist]

    tknzr = get_tokenizer()
    lblmp = get_labelmap(y_train_merged)

    print("%s:      Saving other data" % datetime.now())
    with open('batched_otherdata', 'wb') as file:
        pickle.dump(dict(xtrain=x_train,
                         ytrain=y_train,
                         xtest=x_test,
                         ytest=y_test,
                         tokenizer=tknzr,
                         labelmap=lblmp), file)

    print("%s:      Building model" % datetime.now())
    model = build_model()

    print("%s:      Initializing variables" % datetime.now())
    initialize_variables(sess)

    id_test, mask_test, segment_test = tokenize_inputlist(x_test, tknzr)
    label_test = tokenize_labels(y_test, lblmp)

    for i in range(batches):

        print("%s:      Tokenizing training data" % datetime.now())
        id_train, mask_train, segment_train = tokenize_inputlist(x_train[i], tknzr)
        label_train = tokenize_labels(y_train[i], lblmp)
        print("%s:      Tokenizing testing data" % datetime.now())

        model.fit(x=[id_train, mask_train, segment_train], y=label_train, epochs=1, validation_split=0.1)

        print("%s:      Starting Evaluation" % datetime.now())
        evaluation = model.evaluate([id_test, mask_test, segment_test], label_test, verbose=1)

        print(evaluation)

        print("%s:      Doing predictions" % datetime.now())
        pred = model.predict([id_test, mask_test, segment_test], verbose=1)

        print("%s:      Saving predictions" % datetime.now())

        with open('predictions', 'wb') as file:
            pickle.dump(dict(metrics=evaluation,
                             predictions=pred), file)

        another_batch = bool(int(input('Another epoch?')))

        if not another_batch:
            break

    print("%s:      Saving model" % datetime.now())
    model.save('Trained_batched_BERT.h5')


if __name__ == "__main__":
    main_batched(100)