
# coding: utf-8

# In[8]:


import numpy
import scipy.io
import math
# import geneNewData

def main():
    myID='5543'
    geneNewData.geneData(myID)
    Numpyfile0 = scipy.io.loadmat('digit0_stu_train'+myID+'.mat')
    Numpyfile1 = scipy.io.loadmat('digit1_stu_train'+myID+'.mat')
    Numpyfile2 = scipy.io.loadmat('digit0_testset'+'.mat')
    Numpyfile3 = scipy.io.loadmat('digit1_testset'+'.mat')
    train0 = Numpyfile0.get('target_img')
    train1 = Numpyfile1.get('target_img')
    test0 = Numpyfile2.get('target_img')
    test1 = Numpyfile3.get('target_img')
    print([len(train0),len(train1),len(test0),len(test1)])
    print('Your trainset and testset are generated successfully!')
    
    pass
# end of main function

def calculateMeanSD():
    myID='5543'
    geneNewData.geneData(myID)
    Numpyfile0 = scipy.io.loadmat('digit0_stu_train'+myID+'.mat')
    Numpyfile1 = scipy.io.loadmat('digit1_stu_train'+myID+'.mat')
    Numpyfile2 = scipy.io.loadmat('digit0_testset'+'.mat')
    Numpyfile3 = scipy.io.loadmat('digit1_testset'+'.mat')
    train0 = Numpyfile0.get('target_img')
    train1 = Numpyfile1.get('target_img')
    test0 = Numpyfile2.get('target_img')
    test1 = Numpyfile3.get('target_img')
    
    ### TRAIN 0 ###
    
    #rows and cols in the final matrix containing mean and sd for each image - 5000x2 matrix
    rows = 5000;
    cols = 2;
    matrix = []
    for i in range(rows):
        row = []
        matrix.append(row)
#     print("*****");
#     print(matrix)
    
    #end of nested for loop for initializing the array
    
    for i in range(rows):
        sum = 0;
        innerArr = [];
        for j in range(28):
            for k in range(28):
                innerArr.append(train0[i][j][k])
        numpy_innerArr = numpy.array(innerArr)
        numpy_mean = numpy.mean(numpy_innerArr)
        matrix[i].append(numpy_mean)
    #end of triple nested for loop for MEAN

    for i in range(rows):
        sum = 0;
        innerArr = [];
        for j in range(28):
            for k in range(28):
                sum = sum + train0[i][j][k]
                innerArr.append(train0[i][j][k])
        numpy_sd = numpy.std(innerArr);
        matrix[i].append(numpy_sd);
    #end of triple nested for loop for SD

#     print(matrix)
    ######################################################################################
    # *******  TASK 2  ******* #

    # Mean of feature 1 for digit0
    feature1_array = [];
    for i in range(len(matrix)):
        feature1_array.append(matrix[i][0]);
    #end of for loop

    mean_feature1_digit0 = numpy.mean(feature1_array);
    print("Mean of feature1 for digit0: ");
    print(mean_feature1_digit0);

    # Mean of feature 2 for digit 0
    feature2_array = [];
    for i in range(len(matrix)):
        feature2_array.append(matrix[i][1]);
    #end of for loop

    mean_feature2_digit0 = numpy.mean(feature2_array);
    print("Mean of feature2 for digit0: ");
    print(mean_feature2_digit0);

    # Variance of feature 1 for digit 0
    variance_feature1_digit0 = numpy.var(feature1_array);
    print("Variance of feature 1 for digit 0");
    print(variance_feature1_digit0);

    # Variance of feature 2 for digit 0
    variance_feature2_digit0 = numpy.var(feature2_array);
    print("Variance of feature 2 for digit 0");
    print(variance_feature2_digit0);
    
    ### end of train 0 ###
    ######################
    ### TRAIN 1 (for digit1)###
    #rows and cols in the final matrix containing mean and sd for each image
    
    matrix_digit1 = []
    for i in range(rows):
        row = []
        matrix_digit1.append(row)
    
    #end of nested for loop for initializing the array
    
    for i in range(rows):
        sum = 0;
        innerArr = [];
        for j in range(28):
            for k in range(28):
                innerArr.append(train1[i][j][k])
        numpy_mean = numpy.mean(innerArr)
        matrix_digit1[i].append(numpy_mean)
    #end of triple nested for loop for MEAN

    for i in range(rows):
        sum = 0;
        innerArr = [];
        for j in range(28):
            for k in range(28):
                sum = sum + train1[i][j][k]
                innerArr.append(train1[i][j][k])
        numpy_sd = numpy.std(innerArr);
        matrix_digit1[i].append(numpy_sd);
    #end of triple nested for loop for SD
#     print("Matrix_digit1:")  
#     print(matrix_digit1)
    
    # *******  TASK 2 for train 1  ******* #

    # Mean of feature 1 for digit1
    feature1_array_digit1 = [];
    for i in range(len(matrix_digit1)):
        feature1_array_digit1.append(matrix_digit1[i][0]);
    #end of for loop

    mean_feature1_digit1 = numpy.mean(feature1_array_digit1);
    print("Mean of feature1 for digit1: ");
    print(mean_feature1_digit1);

    # Mean of feature 2 for digit 1
    feature2_array_digit1 = [];
    for i in range(len(matrix_digit1)):
        feature2_array_digit1.append(matrix_digit1[i][1]);
    #end of for loop

    mean_feature2_digit1 = numpy.mean(feature2_array_digit1);
    print("Mean of feature2 for digit1: ");
    print(mean_feature2_digit1);

    # Variance of feature 1 for digit 1
    variance_feature1_digit1 = numpy.var(feature1_array_digit1);
    print("Variance of feature 1 for digit 1");
    print(variance_feature1_digit1);

    # Variance of feature 2 for digit 1
    variance_feature2_digit1 = numpy.var(feature2_array_digit1);
    print("Variance of feature 2 for digit 1");
    print(variance_feature2_digit1);
#     print(test0);
    
    ########################################################################
    #####  Task3  #####
    rows_test0 = 980;
    matrix_test0 = []
    for i in range(rows_test0):
        row = []
        matrix_test0.append(row)
#     print("*****");
#     print(matrix)
    
    #end of nested for loop for initializing the array
    
    for i in range(rows_test0):
        sum = 0;
        innerArr = [];
        for j in range(28):
            for k in range(28):
                innerArr.append(test0[i][j][k])
        numpy_mean = numpy.mean(innerArr)
        matrix_test0[i].append(numpy_mean)
    #end of triple nested for loop for MEAN

    for i in range(rows_test0):
        sum = 0;
        innerArr = [];
        for j in range(28):
            for k in range(28):
                sum = sum + test0[i][j][k]
                innerArr.append(test0[i][j][k])
        numpy_sd = numpy.std(innerArr);
        matrix_test0[i].append(numpy_sd);
    #end of triple nested for loop for SD
    print("Matrix_test0:")
    print(matrix_test0)
    
    ####### Task 3 for Test1 #############
    ######################################
    rows_test1 = 1135;
    matrix_test1 = []
    for i in range(rows_test1):
        row = []
        matrix_test1.append(row)
#     print("*****");
#     print(matrix)
    
    #end of nested for loop for initializing the array
    
    for i in range(rows_test1):
        sum = 0;
        innerArr = [];
        for j in range(28):
            for k in range(28):
                innerArr.append(test1[i][j][k])
        numpy_mean = numpy.mean(innerArr)
        matrix_test1[i].append(numpy_mean)
    #end of triple nested for loop for MEAN

    for i in range(rows_test1):
        sum = 0;
        innerArr = [];
        for j in range(28):
            for k in range(28):
                innerArr.append(test1[i][j][k])
        numpy_sd = numpy.std(innerArr);
        matrix_test1[i].append(numpy_sd);
    #end of triple nested for loop for SD
    print("Matrix_test1:")
    print(matrix_test1)
    
    #####################################################################################
    
    ######### TASK 3 ###########
    
    #### LABEL PREDICTION FOR TEST 0(using matrix_test0) #####
    
    #### P(X|Y=0) = P(X1|Y=0)*P(X2|Y=0)*0.5 --> probability of digit0 ####
    
    ### P(X1|Y=0) ###
    
    label_prediction_test0 = [];
    
    for i in range(len(matrix_test0)):
        
        prob_feature1_digit0 = conditionalProb(matrix_test0[i][0], mean_feature1_digit0, math.sqrt(variance_feature1_digit0));
    
        prob_feature2_digit0 = conditionalProb(matrix_test0[i][1], mean_feature2_digit0, math.sqrt(variance_feature2_digit0));

        prob_digit0 = prob_feature1_digit0*prob_feature2_digit0*0.5

        ### P(X|Y=1) = P(X1|Y=1)*P(X2|Y=1)*0.5

        prob_feature1_digit1 = conditionalProb(matrix_test0[i][0], mean_feature1_digit1, math.sqrt(variance_feature1_digit1));

        prob_feature2_digit1 = conditionalProb(matrix_test0[i][1], mean_feature2_digit1, math.sqrt(variance_feature2_digit1));

        prob_digit1 = prob_feature1_digit1*prob_feature2_digit1*0.5;

        label = compareProb(prob_digit0, prob_digit1);

        label_prediction_test0.append(label);

#         print(label_prediction);
        
    # end of for loop 
    print("label_prediction_test0: ");
    print(label_prediction_test0);
    
    
    #### LABEL PREDICTION FOR TEST 0(using matrix_test1) #####
    
    #### P(X|Y=0) = P(X1|Y=0)*P(X2|Y=0)*0.5 --> probability of digit0 ####
    
    ### P(X1|Y=0) ###
    
    label_prediction_test1 = [];
    
    for j in range(len(matrix_test1)):
        
        prob_feature1_digit0 = conditionalProb(matrix_test1[j][0], mean_feature1_digit0, math.sqrt(variance_feature1_digit0));
    
        prob_feature2_digit0 = conditionalProb(matrix_test1[j][1], mean_feature2_digit0, math.sqrt(variance_feature2_digit0));

        prob_digit0 = prob_feature1_digit0*prob_feature2_digit0*0.5

        ### P(X|Y=1) = P(X1|Y=1)*P(X2|Y=1)*0.5

        prob_feature1_digit1 = conditionalProb(matrix_test1[j][0], mean_feature1_digit1, math.sqrt(variance_feature1_digit1));

        prob_feature2_digit1 = conditionalProb(matrix_test1[j][1], mean_feature2_digit1, math.sqrt(variance_feature2_digit1));

        prob_digit1 = prob_feature1_digit1*prob_feature2_digit1*0.5;

        label = compareProb(prob_digit0, prob_digit1);

        label_prediction_test1.append(label);

#         print(label_prediction);
        
    # end of for loop 
    print("label_prediction_test1: ");
    print(label_prediction_test1);
    
    ##################################################################################
    
    ##### TASK 4 #####
    ### Predicting the accuracy - lies between 0 and 1 ###
    ## for digit 0 ##
    
    count_digit0 = 0;
    
    for x in range(len(label_prediction_test0)):
        if(label_prediction_test0[x] == 0):
            count_digit0 += 1;
    #end of for loop for accuracy_digit0
    accuracy_digit0 = count_digit0*(1.0)/len(label_prediction_test0);
    
    print("accuracy_digit0:");
    print(accuracy_digit0);
    
    ## for digit 1 ##
    
    count_digit1 = 0;
    
    for x in range(len(label_prediction_test1)):
        if(label_prediction_test1[x] == 1):
            count_digit1 += 1;
    #end of for loop for accuracy_digit0
    accuracy_digit1 = count_digit1/len(label_prediction_test1);
    print("accuracy_digit1:");
    print(accuracy_digit1);
    
    


######################################## End of calculateMeanSD Function #############################################


####### conditionalProb function #############
def conditionalProb(x,nu,sigma):
    term1 = 1/(sigma*math.sqrt(2*math.pi));
    term2 = math.exp(-(math.pow((x-nu), 2))/(2*math.pow(sigma,2)));
    return term1*term2;
# end of function conditionalProb

def compareProb(p0, p1):
    if p0 >= p1:
        return 0;
    else:
        return 1;
# end of function compareProb
    
    

if __name__ == '__main__':
    main()
    calculateMeanSD();
#     arr = [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  10, 128,  79,  73,   0,   0,  16,
#     10,   0,   0,   0,   0,   0,   0,   0,   0,   0];
#     print(mean(arr))

