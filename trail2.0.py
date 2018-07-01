import  keras.layers  as  klayers 
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Input, Embedding, GlobalAveragePooling1D, Concatenate, Activation, Lambda, BatchNormalization, Convolution1D, Dropout
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras import initializers
from scipy import stats
from keras.preprocessing.text import Tokenizer
import numpy as np
from gensim.models import Word2Vec,doc2vec
import nltk
from quadratic_weighted_kappa import QWK
from sklearn.metrics import cohen_kappa_score


from additional_feature_getter import feature_getter



class Neural_Tensor_layer(Layer):
	def __init__(self,output_dim,input_dim=None, **kwargs):
		self.output_dim=output_dim
		self.input_dim=input_dim
		if self.input_dim:
			kwargs['input_shape']=(self.input_dim,)
		super(Neural_Tensor_layer,self).__init__(**kwargs)

	def build(self,input_shape):
		mean=0.0
		std=1.0
		k=self.output_dim
		d=self.input_dim
		##truncnorm generate continuous random numbers in given range
		W_val=stats.truncnorm.rvs(-2 * std, 2 * std, loc=mean, scale=std, size=(k,d,d))
		V_val=stats.truncnorm.rvs(-2 * std, 2 * std, loc=mean, scale=std, size=(2*d,k))
		self.W=K.variable(W_val)
		self.V=K.variable(V_val)
		self.b=K.zeros((self.input_dim,))
		self.trainable_weights=[self.W,self.V,self.b]

	def call(self,inputs,mask=None):
		e1=inputs[0]
		e2=inputs[1]
		batch_size=K.shape(e1)[0]
		k=self.output_dim
		

		feed_forward=K.dot(K.concatenate([e1,e2]),self.V)

		bilinear_tensor_products = [ K.sum((e2 * K.dot(e1, self.W[0])) + self.b, axis=1) ]

		for i in range(k)[1:]:	
			btp=K.sum((e2*K.dot(e1,self.W[i]))+self.b,axis=1)
			bilinear_tensor_products.append(btp)

		result=K.tanh(K.reshape(K.concatenate(bilinear_tensor_products,axis=0),(batch_size,k))+feed_forward)

		return result

	def compute_output_shape(self, input_shape):
		batch_size=input_shape[0][0]
		return(batch_size,self.output_dim)

class Temporal_Mean_Pooling(Layer): # conversion from (samples,timesteps,features) to (samples,features)
	def __init__(self, **kwargs):
		super(Temporal_Mean_Pooling,self).__init__(**kwargs)
		# masked values in x (number_of_samples,time)
		self.supports_masking=True
		# Specifies number of dimensions to each layer
		self.input_spec=InputSpec(ndim=3)

	def call(self,x,mask=None):
		if mask is None:
			mask=K.mean(K.ones_like(x),axis=-1)

		mask=K.cast(mask,K.floatx())
				#dimension size single vec/number of samples
		return K.sum(x,axis=-2)/K.sum(mask,axis=-1,keepdims=True)

	def compute_mask(self,input,mask):
		return None
	def compute_output_shape(self,input_shape):
		return (input_shape[0],input_shape[2])


fp1=open("project/glove.6B/glove.6B.300d.txt","r")
glove_emb={}
for line in fp1:
	temp=line.split(" ")
	glove_emb[temp[0]]=np.asarray([float(i) for i in temp[1:]])

print("Embedding done")

EMBEDDING_DIM=300
MAX_NB_WORDS=4000
MAX_SEQUENCE_LENGTH=500
VALIDATION_SPLIT=0.20
DELTA=20


texts=[]
labels=[]
sentences=[]

additional_features=[]

fp=open("training_set_rel3.tsv",'r')
fp.readline()
sentences=[]
doctovec=[]
for line in fp:
	temp=line.split("\t")
	if(temp[1]=='4'):
		texts.append(temp[2])
		labels.append(int(temp[6])/3.0)
		line=temp[2].strip().decode("ascii","ignore").encode("ascii")
		sentences.append(nltk.tokenize.word_tokenize(line))

print("text labels appended %s" %len(texts))

labels=np.asarray(labels)

for i in texts:
	additional_features.append(feature_getter(i))

additional_features=np.asarray(additional_features)
for i in sentences:
	temp1=np.zeros((1, EMBEDDING_DIM))
	for w in i:
		if(w in glove_emb):
			temp1+=glove_emb[w]
	temp1/=len(i)
	doctovec.append(temp1.reshape(300,))

doctovec=np.asarray(doctovec)



tokenizer=Tokenizer() #num_words=MAX_NB_WORDS) #limits vocabulory size
tokenizer.fit_on_texts(texts)
sequences=tokenizer.texts_to_sequences(texts) #returns list of sequences
word_index=tokenizer.word_index #dictionary mapping
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

print(additional_features.shape,data.shape)

print('Shape of data tensor:', data.shape)

indices=np.arange(data.shape[0])
np.random.shuffle(indices)
data=data[indices]
labels=labels[indices]
doctovec=doctovec[indices]
additional_features=additional_features[indices]
validation_size=int(VALIDATION_SPLIT*data.shape[0])

x_train=data[:-validation_size]
y_train=labels[:-validation_size]
doctovec_train=doctovec[:-validation_size]
x_val=data[-validation_size:]
y_val=labels[-validation_size:]
train_ad=additional_features[:-validation_size]
val_ad=additional_features[-validation_size:]
doctovec_val=doctovec[-validation_size:]

model=Word2Vec(sentences,min_count=1, sg=1,size=EMBEDDING_DIM)

model.save("3_embeddings")

embedding_matrix = np.zeros((len(word_index), EMBEDDING_DIM))

for word,i in word_index.items():
	if(i>=len(word_index)):
		continue
	if word in glove_emb:
			embedding_matrix[i]=glove_emb[word]
	# if word in model.wv.vocab.keys():
	# 	embedding_matrix[i]=model.wv[word]

vocab_size=len(word_index)

embedding_layer=Embedding(vocab_size,EMBEDDING_DIM,weights=[embedding_matrix],
							input_length=MAX_SEQUENCE_LENGTH,
							mask_zero=True,
							trainable=False)
side_embedding_layer=Embedding(vocab_size,EMBEDDING_DIM,weights=[embedding_matrix],
							input_length=MAX_SEQUENCE_LENGTH,
							mask_zero=False,
							trainable=False)

def SKIPFLOW(lstm_dim=50, lr=1e-4, lr_decay=1e-6, k=5, eta=3, delta=50, activation="relu", maxlen=MAX_SEQUENCE_LENGTH, seed=None):
	e = Input(name='essay',shape=(maxlen,))
	trad_feats=Input(shape=(7,))
	dtov=Input(shape=(300,))
	embed = embedding_layer(e)
	side_embed = side_embedding_layer(e)
	lstm_layer=LSTM(lstm_dim,return_sequences=True)
	hidden_states=lstm_layer(embed)
	side_hidden_states=lstm_layer(side_embed)
	htm=Temporal_Mean_Pooling()(hidden_states)
	tensor_layer=Neural_Tensor_layer(output_dim=k,input_dim=lstm_dim)
	pairs = [((eta + i * delta) % maxlen, (eta + i * delta + delta) % maxlen) for i in range(maxlen // delta)]
	hidden_pairs = [ (Lambda(lambda t: t[:, p[0], :])(side_hidden_states), Lambda(lambda t: t[:, p[1], :])(side_hidden_states)) for p in pairs]
	sigmoid = Dense(1, activation="sigmoid", kernel_initializer=initializers.glorot_normal(seed=seed))
	coherence = [sigmoid(tensor_layer([hp[0], hp[1]])) for hp in hidden_pairs]
	co_tm=Concatenate()(coherence[:]+[htm])
	dense = Dense(256, activation=activation,kernel_initializer=initializers.glorot_normal(seed=seed))(co_tm)
	dense = Dense(128, activation=activation,kernel_initializer=initializers.glorot_normal(seed=seed))(dense)
	dense = Dense(64, activation=activation,kernel_initializer=initializers.glorot_normal(seed=seed))(dense)
	out = Dense(1, activation="sigmoid")(dense)
	model = Model(inputs=[e,trad_feats,dtov], outputs=[out])
	adam = Adam(lr=lr, decay=lr_decay)
	model.compile(loss="mean_squared_error", optimizer=adam, metrics=["MSE"])
	return model


earlystopping = EarlyStopping(monitor="val_mean_squared_error", patience=5)
sf_1 = SKIPFLOW(lstm_dim=50, lr=2e-4, lr_decay=2e-6, k=4, eta=13, delta=50, activation="relu", seed=None)

epochs = 1000
hist = sf_1.fit([x_train,train_ad,doctovec_train], y_train, batch_size=1024, epochs=epochs,
                validation_data=([x_val,val_ad,doctovec_val], y_val), callbacks=[earlystopping])

y_pred=sf_1.predict([x_val,val_ad,doctovec_val])

y_val=[int(round(a*3)) for a in y_val]
y_pred=[int(round(a*3)) for a in y_pred]

# for i in range(len(y_val)):
# 	print(y_val[i],y_pred[i])

print(cohen_kappa_score(y_val,y_pred,weights="quadratic"))

