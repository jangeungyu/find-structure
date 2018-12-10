# packages and modules

import copy
from itertools import product
from functools import reduce
import numpy as np
import tensorflow as tf
from keras.models import Model, Input
from keras.layers import Dense, Lambda
from keras import optimizers
from keras import backend as K







# functions

class CustomDense(Dense):
    def __init__(self, units, **kwargs):
        super(CustomDense, self).__init__(units, **kwargs)

    def call(self, inputs):
        output = K.dot(inputs, K.softmax(self.kernel, axis=-1))
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output


error0 = 'Error: The morpheme that follows \'all\' should be in a 1-length sort.'

def get_structure(language={}, cardinality=[]):
    '''Returns a structure that matches the language. The structure includes objects needed to deal with quantifiers.'''
    
    structure = {}
    
    for morpheme in language:
        sort = language[morpheme]
        
        if len(sort) > 1:
            input_shape = 1
            for i in range(len(sort)-1):
                input_shape *= cardinality[sort[i]]
            structure[morpheme] = CustomDense(cardinality[sort[-1]], use_bias=False, input_shape=(input_shape,))
            
        elif not 'sort' in morpheme:
            structure[morpheme] = CustomDense(cardinality[sort[-1]], use_bias=False, input_shape=(1,))
            
        else:
            structure[morpheme] = Input(shape=(cardinality[sort[-1]],))
            
    return structure

# It seems like I don't need this function.
def update_language(language={}, cardinality=[]):
    '''Updates the language so that it includes objects needed to deal with quantifiers.'''
    
    for sort in range(len(cardinality)):
        for element_number in range(cardinality[sort]):
            language['sort{0}_{1}'.format(sort,element_number)] = sort

def analyze(sentence=('')):
    '''This function finds out the related morphemes of each morphemes in the given sentence.'''
    
    A = [[] for m in range(len(sentence))]
    
    i,n = 0,1
    while True:
        if n >= len(sentence):
            break
            
        elif sentence[i] == 'all':
            if len(A[i]) == 0:
                A[i].append(-1)
                i += 2
                n += 2
            else:
                i -= 1
                
        elif len(language[sentence[i]]) == 1:
            i -= 1
            
        elif len(A[i]) >= len(language[sentence[i]])-1:
            i -= 1
            
        else:
            A[i].append(n)
            i = n
            n += 1
            
    return A

def affect_to_where(sentence=(''), index=0):
    '''This function finds out, for each morphemes in the given sentence, the place to which it affects.'''
    
    A = analyze(sentence)
    
    i,n = index,0
    while True:
        if len(A[i]) == 0:
            n = i
            break
            
        elif A[i][-1] == -1:
            i += 2
            
        else:
            i = A[i][-1]
            
    return n+1
        
def quantifier_process_one(sentence, number):
    
    for i in range(len(sentence)):
        if sentence[i] == 'all':
            sort = language[sentence[i+1]]
            temp = list(sentence[:i] + sentence[i+2:])
            popped = sentence[i+1]
            
            for j in range(i, affect_to_where(temp, i)):
                if temp[j] == popped:
                    temp[j] = 'sort{}_{}'.format(sort[0],number)
                    
            language['sort{}_{}'.format(sort[0],number)] = sort
                    
            return tuple(temp)
'''
def quantifier_process(sentence):
    This function uses the function quantifier_process_one to handle all quantifers appearing in the given sentence.
    It also updates the language.
    
    resent = sentence
    number_of_quantifiers = sentence.count('all')
    for i in range(number_of_quantifiers):
        resent = quantifier_process_one(resent, i)
        
    return resent
'''

# The reason I changed the original quantifier_process function to this one is that the original function just
# increased the number without determining the sort. For example, if I pass the sentence
# ('all', 'p', 'all', a', ..., 'p', ..., 'a'), where p is of sort (0,) and a is of sort (1,),
# to the original function, it would return (..., 'sort0_0', ..., 'sort1_1').
# But this function returns (..., 'sort0_0', ..., 'sort1_0').
def quantifier_process(sentence):
    resent = copy.copy(sentence)
    
    number_of_quantifiers = [0 for m in cardinality]
    morpheme_index = 0
    
    while morpheme_index < len(resent):
        morpheme = resent[morpheme_index]
        
        if morpheme == 'all':
            sort = language[resent[morpheme_index + 1]]
            resent = quantifier_process_one(resent, number_of_quantifiers[sort[0]])
            number_of_quantifiers[sort[0]] += 1
            
        else:
            morpheme_index += 1
            
    return resent

def mix(ts):
    mixed = ts[0]
    for i in range(len(ts)-1):
        t0 = K.expand_dims(mixed, axis=-1)
        t1 = K.expand_dims(ts[i+1], axis=1)
        mixed = K.batch_flatten(t0*t1)
    return mixed

def get_neural_network(language, cardinality, structure, sentence, number_of_quantifiers):
    
    rectified_sentence = quantifier_process(sentence)
    '''
    number_of_quantified_morphemes = [0 for m in range(len(cardinality))]
    for i in range(len(sentence)):
        if sentence[i] == 'all':
            number_of_quantified_morphemes[language[sentence[i+1]][0]] += 1
    '''
    A = analyze(rectified_sentence)
    output = [0 for m in range(len(rectified_sentence))]
    one = Input(shape=(1,))
    
    for i in range(len(rectified_sentence)):
        morpheme = rectified_sentence[-i-1]
        sort = language[morpheme]
        
        if len(sort) > 1:
            temp = []
            input_shape = 1
            for j in range(len(A[-i-1])):
                temp.append(output[A[-i-1][j]])
                input_shape *= cardinality[language[rectified_sentence[A[-i-1][j]]][-1]]
                
            _input = Lambda(mix, output_shape=(input_shape,))(temp)
            output[-i-1] = structure[morpheme](_input)
        
        elif not 'sort' in morpheme:
            output[-i-1] = structure[morpheme](one)
        
        else:
            output[-i-1] = structure[morpheme]
            
    input_layer = [one]
    for i in range(len(number_of_quantifiers)):
        for j in range(number_of_quantifiers[i]):
            input_layer.append(structure['sort{}_{}'.format(i,j)])
            
    return Model(inputs=input_layer, outputs=output[0])
        

def test(sentence):
    
    resent = quantifier_process(sentence)
    A = analyze(resent[0])

    for sent in range(len(resent)):
        given_sentence = resent[sent]
        output = [0 for m in range(len(given_sentence))]

        for i in range(len(given_sentence)):

            if len(A[-i-1]) == 0:
                temp = structure[given_sentence[-i-1]]
                output[-i-1] = tf.nn.softmax(temp)

            else:
                _input = concatenate_layer([output[A[-i-1][m]] for m in range(len(A[-i-1]))])
                temp = tf.matmul(_input, structure[given_sentence[-i-1]])
                output[-i-1] = tf.nn.softmax(temp)

        print(sess.run(output[0]))
        
def maximum_number_of_quantifiers(axioms):
    maximum_number_of_appearances = [0 for m in cardinality]
    
    for sentence in axioms:
        temp = [0 for m in cardinality]
        
        for morpheme_index in range(len(sentence)):
            morpheme = sentence[morpheme_index]
            
            if morpheme == 'all':
                sort = language[sentence[morpheme_index + 1]]
                temp[sort[0]] += 1
            
        for i in range(len(maximum_number_of_appearances)):
            if temp[i] > maximum_number_of_appearances[i]:
                maximum_number_of_appearances[i] = temp[i]
                
    return maximum_number_of_appearances

def get_input(cardinality, number_of_quantifiers):
    a = [[[] for n in range(m)] for m in number_of_quantifiers]
    
    temp = [cardinality[m]**number_of_quantifiers[m] for m in range(len(cardinality))]
    batch_size = reduce(lambda x, y: x*y, temp)
    one = np.ones(shape=(batch_size, 1))
    
    prod_temp = []
    for cardinality_index in range(len(cardinality)):
        prod_temp.append(product(range(cardinality[cardinality_index]), repeat=number_of_quantifiers[cardinality_index]))
    prod = product(*prod_temp)
    
    for i in prod:
        for cardinality_index in range(len(cardinality)):
            for j in range(number_of_quantifiers[cardinality_index]):
                temp = list(np.eye(cardinality[cardinality_index])[i[cardinality_index][j]])
                a[cardinality_index][j].append(temp)
                
    for cardinality_index in range(len(cardinality)):
        for j in range(number_of_quantifiers[cardinality_index]):
            a[cardinality_index][j] = np.array(a[cardinality_index][j])
            
    return [one] + [item for sublist in a for item in sublist]

def get_answer(cardinality, number_of_quantifiers):

    temp = [cardinality[m]**number_of_quantifiers[m] for m in range(len(cardinality))]
    batch_size = reduce(lambda x, y: x*y, temp)
    
    a = []
    for i in range(batch_size):
        a.append([1,0])
    
    return np.array(a)

def softmax(ar):
    exponent = np.exp(ar)
    return exponent/exponent.sum(axis=1, keepdims=True)








# setting

# Cardinality[n] is a cardinality of sort n's universe.
cardinality = [2, 3]

# Keys are morphemes, and values are sorts.
language = {'a':(1,), 'b':(1,), 'c':(1,), '0':(1,), '-':(1,1), '+':(1,1,1), '=':(1,1,0), 'and':(0,0,0)}

# axioms of group theory
sentence0 = ('all','a','all','b','all','c','=','+','+','a','b','c','+','a','+','b','c') # associative property
sentence1 = ('all','a','and','=','+','a','0','a','=','+','0','a','a') # 0 is an identity.
sentence2 = ('all','a','and','=','+','a','-','a','0','=','+','-','a','a','0') # For all a, -a is an inverse of a.

axioms = (sentence0, sentence1, sentence2)

# Don't change the following lines.
rectified_axioms = list(axioms)
for axiom in range(len(axioms)):
    rectified_axioms[axiom] = quantifier_process(axioms[axiom])
rectified_axioms = tuple(rectified_axioms)
structure = get_structure(language=language, cardinality=cardinality)

# We should run get_neural_network function here and not below
# because otherwise the inputs shapes of the layers are not specified.
# Actually, no. Because I recently set the input shapes manually in get_structure function.
# So this can go back to below.
# Actually, I was wrong. This has to stay here, otherwise I get an error. I don't know the reason.
number_of_quantifiers = maximum_number_of_quantifiers(axioms)
net0 = get_neural_network(language, cardinality, structure, sentence0, number_of_quantifiers)
net1 = get_neural_network(language, cardinality, structure, sentence1, number_of_quantifiers)
net2 = get_neural_network(language, cardinality, structure, sentence2, number_of_quantifiers)

# fix meanings of some morphemes
structure['='].set_weights([np.array([[64,-64],[-64,64],[-64,64],
                                     [-64,64],[64,-64],[-64,64],
                                     [-64,64],[-64,64],[64,-64]], dtype='float32')])
structure['='].trainable = False

structure['and'].set_weights([np.array([[64,-64],[-64,64],[-64,64],[-64,64]], dtype='float32')])
structure['and'].trainable = False








#train

net0.compile(optimizer=optimizers.Adam(lr=0.1), loss='mean_squared_error', metrics=['accuracy'])
net1.compile(optimizer=optimizers.Adam(lr=0.1), loss='mean_squared_error', metrics=['accuracy'])
net2.compile(optimizer=optimizers.Adam(lr=0.1), loss='mean_squared_error', metrics=['accuracy'])

# I think the problem arises because it only finds the local minimizer.
# So if net0 is already fitted, then I should probably train only net1 and net2 so that the variables will not be
# keep going back to the local minimizer.
# Actually, don't apply the above thing to each net, but apply it to each data.

inputs = get_input(cardinality, number_of_quantifiers)
answers = get_answer(cardinality, number_of_quantifiers)

for epoch in range(100):
    print('epoch: {}'.format(epoch))
    net0.fit(inputs, answers)
    net1.fit(inputs, answers)
    net2.fit(inputs, answers)








#verify

verify_0 = softmax(structure['0'].get_weights()[0])
verify_plus = softmax(structure['+'].get_weights()[0])
verify_minus = softmax(structure['-'].get_weights()[0])

print('0:\n{}\n'.format(verify_0))
print('+:\n{}\n'.format(verify_plus))
print('-:\n{}\n\n'.format(verify_minus))

print('0:\n{}\n'.format(np.argmax(verify_0)))
print('+:\n{}\n'.format(np.argmax(verify_plus, axis=1).reshape(3,3)))
print('-:\n{}'.format(np.argmax(verify_minus, axis=1)))
