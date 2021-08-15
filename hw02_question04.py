import numpy as np
import math
def sigmoid(x):
    result = 1/(1+math.exp(-x))
    return result

def transposeMatrix(A): # Chuyển vị ma trận A
    row = len(A) # tính số dòng của ma trận A
    col = len(A[0]) # tính số cột của ma trận A
    return [[A[i][j] for i in range(row)] for j in range(col)] # chuyển vị ma trận

def calculate_hidden(x, weight_1):
    result = []
    new_weight = transposeMatrix(weight_1)
    for i in range(3):
        sum = 0
        for j in range(len(x)):
            sum += x[j]*new_weight[i][j]
        result.append(sigmoid(sum))
    return result

def calculate_output(output_hidden, weight_2):
    result = []
    new_weight = transposeMatrix(weight_2)
    for i in range(2):
        sum = 0
        for j in range(len(output_hidden)):
            sum += output_hidden[j]*new_weight[i][j]
        result.append(sigmoid(sum))
    return result

def calculate_error(target_output, actual_output):
    error = 0
    for i in range(2):
        error += 0.5*((target_output[i] - actual_output[i])**2)
    return error

def update_weight_2(target, actual, alpha, weight_2):
    updated_weight = []
    error_gradients = []
    for j in range(len(weight_2)):
        temp = []
        for k in range(len(weight_2[0])):
            error_gradient = actual[k]*(1-actual[k])*(target[k] - actual[k])
            error_gradients.append(error_gradient)
            delta_weight = alpha*actual[k]*error_gradient
            temp.append(weight_2[j][k] + delta_weight)
        updated_weight.append(temp)
    return updated_weight, error_gradients

def calculate_sth(error_gradients, weight_2, pos):
    sum = 0
    for i in range(2):
        sum += error_gradients[i]*weight_2[pos][i]
    return sum

def update_weight_1(output_hidden, alpha, weight_1, weight_2, x, error_gradients):
    updated_weight = []
    for j in range(len(weight_1)):
        temp = []
        for k in range(len(weight_1[0])):
            error_gradient = output_hidden[k]*(1-output_hidden[k])*calculate_sth(error_gradients, weight_2, k)
            delta_weight = alpha*x[j]*error_gradient
            temp.append(weight_1[j][k] + delta_weight)
        updated_weight.append(temp)
    return updated_weight

def back_propagation():
    # Khai báo các biến cần thiết
    alpha = 0.1 # learning rate
    x = [0.1, 0.5]
    target_output = [1, 0]
    output_hidden = []
    last_output = []
    weight_1 = [[0.2, 0.4, -0.4], [0.1, 0, 0.3]]
    weight_2 = [[0.6, -0.2], [0.1, -0.1], [-0.4, 0.2]]
    time = 0
    heso = 0
    while time <= 10000:
        output_hidden = calculate_hidden(x, weight_1)
        last_output = calculate_output(output_hidden, weight_2)
        error = calculate_error(target_output, last_output)
        # update weight_2 (hidden layer to output layer)
        new_weight_2, error_gradients = update_weight_2(target_output, last_output, alpha, weight_2)
        # update weight_1 (input layer to hidden layer)
        new_weight_1 = update_weight_1(output_hidden, alpha, weight_1, weight_2, x, error_gradients)
        if time == 1000 * heso:
            print("Lặp", time, ":")
            print("Output at hidden layer = ", output_hidden)
            print("Output at output layer = ", last_output)
            print("Wij(1) after updated: ",new_weight_1)
            print("Wij(2) after updated: ", new_weight_2)
            print("TOTAL ERROR = ", error)
            heso += 1
        weight_2 = new_weight_2
        weight_1 = new_weight_1
        time += 1
        

def main():
    back_propagation()

if __name__ == "__main__":
    main()
