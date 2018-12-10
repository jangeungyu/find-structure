
# coding: utf-8

# In[64]:


# used genetic algorithm

# packages and modules

import numpy as np
import copy


# In[65]:


#class and functions

class chromosome:
    
    language = {}
    cardinality = []
    crossover_probability = 0.5
    mutate_probability = 0.025
    
    def __init__(self, structure):
        self.structure = structure
    
    def fitness(self, rectified_sentences):
        
        guesses = self.predict(rectified_sentences)
        fitness = 0
        
        for i in range(len(guesses)):
            temp = 0
            for j in range(len(guesses[i])):
                if guesses[i][j][0][0] == 1:
                    temp += 1

            temp /= len(guesses) * len(guesses[i])
            fitness += temp
            
        return fitness
                    
    @staticmethod
    def mix(ar):
        mixed = ar[0]
        for i in range(len(ar)-1):
            a0 = np.expand_dims(mixed, axis=-1)
            a1 = np.expand_dims(ar[i+1], axis=1)
            mixed = np.matmul(a0, a1)
            mixed = np.expand_dims(mixed.flatten(), axis=0)
        return mixed
    
    def predict(self, rectified_sentences):
        
        guesses = [[] for m in range(len(rectified_sentences))]
        
        for i in range(len(rectified_sentences)):
            
            A = analyze(rectified_sentences[i][0], chromosome.language)
            
            for j in range(len(rectified_sentences[i])):
                
                sentence = rectified_sentences[i][j]
                outputs = [0 for m in range(len(sentence))]
                
                for k in range(len(sentence)):
                    
                    morpheme = sentence[-k-1]
                    sort = chromosome.language[morpheme]
                    
                    if len(sort) == 1:
                        outputs[-k-1] = self.structure[morpheme]
                        
                    else:
                        mixed = chromosome.mix([outputs[m] for m in A[-k-1]])
                        outputs[-k-1] = np.matmul(mixed, self.structure[morpheme])

                guesses[i].append(outputs[0])
                
        return guesses
        
    @staticmethod
    def crossover(parent0, parent1, trainables):
        
        child0 = chromosome(copy.deepcopy(parent0.structure))
        child1 = chromosome(copy.deepcopy(parent1.structure))
        
        for morpheme in trainables:
            for i in range(len(parent1.structure[morpheme])):
                if np.random.choice((True, False), p=(chromosome.crossover_probability,1-chromosome.crossover_probability)):
                    child0.structure[morpheme][i] = parent1.structure[morpheme][i]
                    child1.structure[morpheme][i] = parent0.structure[morpheme][i]
                
        return [child0, child1]
    
    def mutate(self, trainables):
        
        child = chromosome(copy.deepcopy(self.structure))
        
        for morpheme in trainables:
            sort = chromosome.language[morpheme]
            output_size = chromosome.cardinality[sort[-1]]
            
            for i in range(len(self.structure[morpheme])):
                if np.random.choice((True, False), p=(chromosome.mutate_probability,1-chromosome.mutate_probability)):
                    child.structure[morpheme][i] = np.eye(output_size, dtype='bool')[np.random.choice(output_size)]
                    
        return child
                

def update_language(language, cardinality):
    '''Updates the language so that it includes objects needed to deal with quantifiers.'''
    
    updated_language = language.copy()
    for sort in range(len(cardinality)): 
        for element_number in range(cardinality[sort]):
            updated_language['sort{0}_{1}'.format(sort,element_number)] = (sort,)
            
    return updated_language

# Don't put this function as a getter(using @property decorator) of the chromosome class.
# Because if you do, the structure will be keep changing randomly.
def get_structure(language, cardinality, fixed_morphemes):
    '''Returns a structure that matches the language.'''
    
    structure = fixed_morphemes.copy()
    trainables = [morpheme for morpheme in language if not morpheme in fixed_morphemes]
    
    for morpheme in trainables:
        sort = language[morpheme]
        
        if len(sort) > 1:
            input_size = 1
            for i in range(len(sort)-1):
                input_size *= cardinality[sort[i]]
            output_size = cardinality[sort[-1]]
            structure[morpheme] = np.eye(output_size, dtype='bool')[np.random.choice(output_size, input_size)]
            
        elif not 'sort' in morpheme:
            output_size = cardinality[sort[-1]]
            structure[morpheme] = np.eye(output_size, dtype='bool')[np.random.choice(output_size, 1)]
            
        else:
            output_size = cardinality[sort[-1]]
            structure[morpheme] = np.eye(output_size, dtype='bool')[[int(morpheme.split('_')[1])]]
            
    return structure

def analyze(sentence, language):
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

def affect_to_where(sentence, index, language):
    '''This function finds out, for each morphemes in the given sentence, the place to which it affects.'''
    
    A = analyze(sentence, language)
    
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

def quantifier_process_one(sentence, language):
    '''This function gets rid of the first quantifier that appears in the given sentence and a morpheme that follows the quantifier,
    and then replaces the rest of the morphemes in the sentence to something else, to the place where the removed quantifier had an affect.'''
    
    for i in range(len(sentence)):
        if sentence[i] == 'all':
            sort = language[sentence[i+1]]
            if len(sort) == 1:
                temp1 = sentence[:i] + sentence[i+2:]
                popped = sentence[i+1]
                asdf = []
                                
                for j in range(cardinality[sort[0]]):
                    temp2 = list(temp1)
                    for k in range(i, affect_to_where(temp1,i,language)):
                        if temp1[k] == popped:
                            temp2[k] = 'sort{0}_{1}'.format(sort[0],j)
                    asdf.append(tuple(temp2))
                    
                return asdf
                    
            else:
                return 'Error: The morpheme that follows \'all\' should be in a 1-length sort.'
            
def quantifier_process(sentence, language):
    '''This function uses the function quantifier_process_one to handle all quantifers appearing in the given sentence.'''
    
    resent = [sentence] # abbreviation of 'rectified sentence'
    num_all = sentence.count('all')
    for i in range(num_all):
        temp = []
        for j in range(len(resent)):
            temp += quantifier_process_one(resent[j], language)
        resent = temp
    
    return resent

def softmax(a):
    b = [10000**k for k in a]
    c = [k/sum(b) for k in b]
    return c


# In[71]:


# setting

# Cardinality[n] is a cardinality of sort n's universe.
cardinality = [2, 3]

# Keys are morphemes, and values are sorts.
language = {'a':(1,), 'b':(1,), 'c':(1,), '0':(1,), '-':(1,1), '+':(1,1,1), '=':(1,1,0), 'and':(0,0,0), 'imply':(0,0,0)}

fixed_morphemes = {'a':np.array([[1,0,0]], dtype='bool'),
                  'b':np.array([[1,0,0]], dtype='bool'),
                  'c':np.array([[1,0,0]], dtype='bool'),
                  '=':np.array([[1,0],
                               [0,1],
                               [0,1],
                               [0,1],
                               [1,0],
                               [0,1],
                               [0,1],
                               [0,1],
                               [1,0]], dtype='bool'),
                  'and':np.array([[1,0],
                                 [0,1],
                                 [0,1],
                                 [0,1]], dtype='bool'),
                  'imply':np.array([[1,0],
                                   [0,1],
                                   [1,0],
                                   [1,0]], dtype='bool')}
trainables = [morpheme for morpheme in language if not morpheme in fixed_morphemes]

# group theory

sentence0 = ('all','a','all','b','all','c','=','+','+','a','b','c','+','a','+','b','c') # associative property
sentence1 = ('all','a','and','=','+','a','0','a','=','+','0','a','a') # 0 is an identity.
sentence2 = ('all','a','and','=','+','a','-','a','0','=','+','-','a','a','0') # For all a, -a is an inverse of a.
#sentence3 = ('all','a','all','b','imply','=','+','a','b','b','=','a','0') # This is a theorem in group theory.
# I included this in the hope for better performance.
#sentence4 = ('all','a','all','b','imply','=','+','a','b','0','=','b','-','a') # This is a theorem in group theory.
# I included this in the hope for better performance.

axioms = (sentence0, sentence1, sentence2)

chromosome.language = update_language(language, cardinality)
chromosome.cardinality = cardinality
rectified_axioms = [quantifier_process(axioms[m], chromosome.language) for m in range(len(axioms))]


# In[76]:


population = 4 # population should be at least 4.

chromos = [chromosome(get_structure(chromosome.language, chromosome.cardinality, fixed_morphemes)) for m in range(population)]

for epoch in range(1000):
    fitnesses = [chromo.fitness(rectified_axioms) for chromo in chromos]
    if fitnesses[0]>=0.9 or fitnesses[1]>=0.9 or fitnesses[2]>=0.9 or fitnesses[3]>=0.9:
        print(epoch)
        print(fitnesses)
        break
    p = softmax(fitnesses)
    parents_indexes = np.random.choice(population, 2, replace=False, p=p)
    parents = [chromos[parents_indexes[0]], chromos[parents_indexes[1]]]
    children = chromosome.crossover(parents[0], parents[1], trainables)
    children[0] = children[0].mutate(trainables)
    children[1] = children[1].mutate(trainables)
    chromos = parents + children
    print(fitnesses)
    print(parents_indexes)
    
max_index = np.argmax(fitnesses)


# In[85]:


# verify

verify_0 = chromos[max_index].predict([quantifier_process(('0',), chromosome.language)])
verify_plus = chromos[max_index].predict([quantifier_process(('all','a','all','b','+','a','b'), chromosome.language)])
verify_minus = chromos[max_index].predict([quantifier_process(('all','a','-','a'), chromosome.language)])

visualize_0 = np.argmax(verify_0[0][0][0])

visualize_plus = []
for card in range(cardinality[1]**2):
    visualize_plus.append(np.argmax(verify_plus[0][card][0]))
visualize_plus = np.array(visualize_plus).reshape(3,3)

visualize_minus = []
for card in range(cardinality[1]):
    visualize_minus.append(np.argmax(verify_minus[0][card][0]))
visualize_minus = np.array(visualize_minus)

print('0:\n{}\n'.format(visualize_0))
print('+:\n{}\n'.format(visualize_plus))
print('-:\n{}'.format(visualize_minus))


# In[87]:


chromos[max_index].predict(rectified_axioms)

