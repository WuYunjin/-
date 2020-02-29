
"""

Reference: https://www.kaggle.com/eray1yildiz/using-lstms-with-attention-for-emotion-recognition
"""
import pandas as pd 
import numpy as np
import tensorflow.keras as keras

from tensorflow.keras.preprocessing.sequence import pad_sequences

def encode_word(dataset):

    input_sentences = dataset["微博中文内容"].values.tolist()#目前只要微博内容一个信息，其他信息暂时不用
    labels = dataset["情感倾向"].values.tolist()

    word2id = dict()
    label2id = dict()
    

    label2id = {'-1': 0, '0': 1, '1': 2} #{l: i for i, l in enumerate(set(labels))}
    
    for sentence in input_sentences:
        for word in sentence:
            # Add words to word2id dict if not exist
            if word not in word2id:
                word2id[word] = len(word2id)
    
    id2label = {0: '-1', 1: '0', 2: '1'} #{v: k for k, v in label2id.items()} 

    max_words = dataset['微博中文内容'].str.len().max() 

    return word2id,max_words,label2id,id2label# maximum number of words in a sentence 

def trained_model(dataset,word2id,max_words,label2id):
   

    input_sentences = dataset["微博中文内容"].values.tolist()#目前只要微博内容一个信息，其他信息暂时不用
    labels = dataset["情感倾向"].values.tolist()

    
    # Encode input words and labels
    X = [[word2id[word] for word in sentence] for sentence in input_sentences]
    Y = [label2id[label] for label in labels]

    X = pad_sequences(X, max_words)
    
    # Convert Y to numpy array
    Y = keras.utils.to_categorical(Y, num_classes=len(label2id), dtype='float32')

    # Print shapes
    print("Shape of X: {}".format(X.shape))
    print("Shape of Y: {}".format(Y.shape))



    #Build LSTM model with attention

    embedding_dim = 100 # The dimension of word embeddings

    # Define input tensor
    sequence_input = keras.Input(shape=(max_words,), dtype='int32')

    # Word embedding layer
    embedded_inputs =keras.layers.Embedding(len(word2id) + 1,
                                            embedding_dim,
                                            input_length=max_words)(sequence_input)

    # Apply dropout to prevent overfitting
    embedded_inputs = keras.layers.Dropout(0.2)(embedded_inputs)

    # Apply Bidirectional LSTM over embedded inputs
    lstm_outs = keras.layers.Bidirectional(
        keras.layers.LSTM(embedding_dim, return_sequences=True)
    )(embedded_inputs)  

    # Apply dropout to LSTM outputs to prevent overfitting
    lstm_outs = keras.layers.Dropout(0.2)(lstm_outs)

    # Attention Mechanism - Generate attention vectors
    input_dim = int(lstm_outs.shape[2])
    permuted_inputs = keras.layers.Permute((2, 1))(lstm_outs)
    attention_vector = keras.layers.TimeDistributed(keras.layers.Dense(1))(lstm_outs)
    attention_vector = keras.layers.Reshape((max_words,))(attention_vector)
    attention_vector = keras.layers.Activation('softmax', name='attention_vec')(attention_vector)
    attention_output = keras.layers.Dot(axes=1)([lstm_outs, attention_vector])

    # Last layer: fully connected with softmax activation
    fc = keras.layers.Dense(embedding_dim, activation='relu')(attention_output)
    output = keras.layers.Dense(len(label2id), activation='softmax')(fc)

    # Finally building model
    model = keras.Model(inputs=[sequence_input], outputs=output)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer='adam')

    # Print model summary
    # print(model.summary())

    # Train model 
    model.fit(X, Y, epochs=20, batch_size=128, validation_split=0.1, shuffle=True)

    # 将整个模型保存为HDF5文件
    model.save('lstm_model.h5')

    return model
    # model.load_weights('./checkpoints/my_checkpoint')

def prediction(testset,model_path,word2id,max_words,id2label):
    model = keras.models.load_model(model_path)
 
    # print(model.summary())
    input_sentences = testset["微博中文内容"].values.tolist()#目前只要微博内容一个信息，其他信息暂时不用
    # Encode input words and labels
    X = [[word2id[word] for word in sentence] for sentence in input_sentences]
    X = pad_sequences(X, max_words)
 
    # Make predictions
    label_probs = model.predict(X)
    pred_id = np.argmax( label_probs, axis = 1)
    # print(pred_id)
    pred_label= [ id2label[id] for id in pred_id]

    y = pd.Series(pred_label, name='y', index=testset.index)
    id = pd.Series(testset["微博id"],name='id')
    result = pd.concat([id, y], axis=1)
    print("done")
    return result
    # print(result)


if __name__ == "__main__":
    

    #Loading the dataset, preprocessing
    dataset = pd.read_csv('nCoV_100k_train.labled.csv',encoding='utf-8')
    dataset.dropna(axis=0,inplace=True)
    noise_index = dataset[(dataset['情感倾向'] != '1') & (dataset['情感倾向'] != '0') &(dataset['情感倾向'] != '-1') ].index
    dataset.drop(noise_index,inplace=True)
    print("training...")
    
    word2id, max_words,label2id,id2label = encode_word(dataset)
    
    model = trained_model(dataset,word2id,max_words,label2id)

    print("predicting...")
    testset = pd.read_csv('nCoV_10k_test.csv',encoding='utf-8')
    result = prediction(testset,'lstm_model.h5',word2id,max_words,id2label)
    #write result
    result.to_csv('DMIRLAB-final.csv',encoding='utf-8',index=False)


