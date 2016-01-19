import random
from string import ascii_uppercase
import numpy
from random import randint


source = []
training_set = []
testing_set = []
initial_weight = []
perceptrons = {}
learning_rate = 0.2
confusion_matrix = []

alphabet = 0
# declare all the variables I need
for alphabet in range(26):
    source.append([])
    training_set.append([])
    testing_set.append([])
    confusion_matrix.append([])
    for alphabet2 in range(26):
        confusion_matrix[alphabet].append(0)


# read data into three dimentional lists
def read_data(the_list):
    with open('letter-recognition.data', 'r') as f:
        while True:
            one_line = f.readline()
            if not one_line:
                break
            one_data_set = one_line.split(',', 16)

            # remove the letter from the list
            letter = one_data_set.pop(0)

            # charIntValue is the ascii integer value for the letter
            letter_int_value = ord(letter)

            # remove the new line element from last element in list
            one_data_set[-1] = one_data_set[-1].strip()

            # add this set value to the list
            the_list[letter_int_value-65].append(one_data_set)


# split data in two parts one is training set, another is testing set
# size try to be half and half
def split_data(the_list, split_list1, split_list2):
    for i in range(26):
        half_size = len(the_list[i])/2
        size = len(the_list[i])
        for j in range(size):
            if j < half_size:
                split_list1[i].append(the_list[i][j])

            else:
                split_list2[i].append(the_list[i][j])

# convert the data from string to integer
def convert_to_int(to_convert_list):
    for i in range(26):
        size = len(to_convert_list[i])
        for j in range(size):
            length = len(to_convert_list[i][j])
            for k in range(length):
                to_convert_list[i][j][k] = int(to_convert_list[i][j][k])


# scale the data in the range of 0 to 1
def preprocess_data(to_scale):
    for i in range(26):
        size = len(to_scale[i])
        for j in range(size):
            length = len(to_scale[i][j])
            for k in range(length):
                to_scale[i][j][k] /= 15.0

            # add bias at the end of the input data for each set
            to_scale[i][j].append(1)


# init the weight
def init_weight(to_init):
    for i in range(17):
        temp = random.uniform(-1, 1)
        to_init.append(temp)


# perceptron['A']['B'] = [17 weights]
def get_perceptrons(the_dict):
    for i in ascii_uppercase:
        the_dict.update({i: {}})
        for j in ascii_uppercase:
            if ord(j) > ord(i):
                the_dict[i].update({j: []})

    for key1 in the_dict:
        for key2 in the_dict[key1]:
            for k in range(17):
                temp = random.uniform(-1, 1)
                the_dict[key1][key2].append(temp)
            # print key1, key2, the_dict[key1][key2]


# dot product
# weight[0]*input_data[0] + weight[1]*input_data[1]....
def get_output(weight, input_data):
    output = sum([i * j for i, j in zip(weight, input_data)])
    if output >= 0:
        return 1
    else:
        return -1


# calculate the accuracy on current weight
# the data has to be sorted as data= [[As' list],[Bs' list]]
def get_accuracy(weight, data):
    correctness = 0
    total_len = len(data[0]) + len(data[1])
    for i in range(2):
        for j in range(len(data[i])):
            if i == 0:
                target = 1
            else:
                target = -1
            output = get_output(weight, data[i][j])
            #print 'target is: ', target, 'output is: ', output
            if output == target:
                correctness += 1
    return correctness/float(total_len)


# calculate the weight using equation: weight += learning_rate * input * target
def update_weights(weight, shuffled_data):
        for i in range(len(weight)):
            weight[i] += learning_rate * shuffled_data[i] * shuffled_data[len(shuffled_data) - 1]

# def one_weight_acc(weight, data_sets):
#     correct = 0
#     for i in range(len(data_sets)):
#         output = get_output(weight, data_sets[i])
#         if output == data_sets[i][len(data_sets[i]) - 1]:
#             correct += 1
#     return correct / float(len(data_sets))


# train one perceptron with a set of training data
def train(weight, data):
    temp_list = []
    temp_data = data[:]
    for j in range(2):
        for k in range(len(temp_data[j])):
            if j == 0:
                temp_data[j][k].append(1)
            else:
                temp_data[j][k].append(-1)
            temp_list.append(data[j][k])
    # shuffle the list
    random.shuffle(temp_list)
    old_acc = 0.0
    new_acc = -0.1
    for i in range(len(temp_list)):
        output = get_output(weight, temp_list[i])
        if output != temp_list[i][len(temp_list[i]) - 1]:
            update_weights(weight, temp_list[i])



# loop through the 325 perceptrons and train each of them
def train_wrapper(the_perceptrons, the_training_data):
    for i in the_perceptrons:
        for j in the_perceptrons[i]:
            temp_list = [the_training_data[ord(i)-ord('A')], the_training_data[ord(j)-ord('A')]]
            train(the_perceptrons[i][j], temp_list)
            # get_accuracy(the_perceptrons[i][j], temp_list)


#    A   B
# A a1  p1
# B p2  a2
# index1 and index2 should be char
# def fill_matrix(weight, the_testing_data, index1, index2):
#     predicted1 = 0
#     predicted2 = 0
#     actual1 = 0
#     actual2 = 0
#
#     temp_list = []
#     temp_data = the_testing_data[:]
#     for j in range(2):
#         for k in range(len(temp_data[j])):
#             if j == 0:
#                 temp_data[j][k].append(1)
#             else:
#                 temp_data[j][k].append(-1)
#             temp_list.append(the_testing_data[j][k])
#
#     random.shuffle(temp_list)
#     for i in range(len(temp_list)):
#         output = get_output(weight, temp_list[i])
#         target = temp_list[i][len(temp_list[i]) - 1]
#         if output == target and target == 1:
#             actual1 += 1
#         elif output == target and target == -1:
#             actual2 += 1
#         elif output != target and target == 1:
#             predicted1 += 1
#         # if output != target and target == -1:
#         else:
#             predicted2 += 1
#
#     confusion_matrix[ord(index1)-ord('A')][ord(index1)-ord('A')] += actual1
#     confusion_matrix[ord(index1)-ord('A')][ord(index2)-ord('A')] += predicted1
#     confusion_matrix[ord(index2)-ord('A')][ord(index2)-ord('A')] += actual2
#     confusion_matrix[ord(index2)-ord('A')][ord(index1)-ord('A')] += predicted2
#
#
# # iterate for all perceptrons to fill out entire matrix
# def fill_matrix_wrapper(the_perceptrons, the_testing_data):
#     for i in the_perceptrons:
#         for j in the_perceptrons[i]:
#             temp_list = [the_testing_data[ord(i)-ord('A')], the_testing_data[ord(j)-ord('A')]]
#             fill_matrix(the_perceptrons[i][j], temp_list, i, j)


# pass in all perceptrons and one set of data to get predicted answer
def find_prediction(the_perceptrons, the_testing_data):
    vote = []
    for vote_index in range(26):
        vote.append(0)
    for i in the_perceptrons:
        for j in the_perceptrons[i]:
            output = get_output(the_perceptrons[i][j], the_testing_data)
            if output == 1:
                vote[ord(i) - ord('A')] += 1
            else:
                vote[ord(j) - ord('A')] += 1
    max_num = max(vote)
    max_list = []
    for k in range(26):
        if vote[k] == max_num:
            max_list.append(k)
    if len(max_list) == 1:
        return max_list[0]
    else:
        return max_list[randint(0, len(max_list)-1)]


# fill the matrix
def fillup_matrix(the_perceptrons, the_testing_data):
    temp_list = []
    for i in range(len(the_testing_data)):
        for j in range(len(the_testing_data[i])):
            the_testing_data[i][j].append(i)
            temp_list.append(the_testing_data[i][j])
    random.shuffle(temp_list)
    for k in range(len(temp_list)):
        index2 = find_prediction(the_perceptrons, temp_list[k])
        index1 = temp_list[k][len(temp_list[k])-1]
        confusion_matrix[index1][index2] += 1

    # for i in range(len(the_testing_data)):
    #     for j in range(len(the_testing_data[i])):
    #         # index2 is the predict result
    #         index2 = find_prediction(the_perceptrons, the_testing_data[i][j])
    #         # index1 is the actual result
    #         index1 = i
    #         confusion_matrix[index1][index2] += 1


# number of correct / total number
def calculate_acc(the_matrix):
    sum_actual = 0
    total = 0
    for i in range(26):
        sum_actual += the_matrix[i][i]
        for j in range(26):
            total += the_matrix[i][j]
    return sum_actual/float(total)


# print the matrix in matrix format
def print_matrix(the_matrix):
    for ab in range(26):
        for ac in range(26):
            print ' ', the_matrix[ab][ac],
        print

read_data(source)
split_data(source, training_set, testing_set)
convert_to_int(training_set)
convert_to_int(testing_set)
preprocess_data(training_set)
preprocess_data(testing_set)

# init_weight(initial_weight)
get_perceptrons(perceptrons)

# training process
train_wrapper(perceptrons, training_set)

fillup_matrix(perceptrons, testing_set)
# fill_matrix_wrapper(perceptrons, testing_set)
print calculate_acc(confusion_matrix)
print_matrix(confusion_matrix)


