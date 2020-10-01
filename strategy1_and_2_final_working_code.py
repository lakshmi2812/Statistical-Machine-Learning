import numpy
# data = np.load('AllSamples.npy')
# data = numpy.array([[0,0],[0,2],[2,0],[2,1],[2,2],[4,0],[4,2],[6,0],[6,2],[10,2]]);
data = numpy.array([[0,0],[0,3],[0,9],[0,14],[0,20]]);
import math;


# In[3]:


# k1,i_point1,k2,i_point2 = initial_S2('5543') # please replace 0111 with your last four digit of your ID


# In[13]:

k1= 4;
k2 = 6;
i_point1 = numpy.array([0,0]);
i_point2 = numpy.array([5,0]);

print(k1)
print(i_point1)
print(k2)
print(i_point2)

################## Start of code for Strategy 2 #########################

def isPointInSample(array, point):
    for x,y in array:
        if(point[0] == x and point[1] == y):
            return True;
    return False;
#end of isPointInSample function

def euclidianDistance(point1,point2):
    return math.sqrt(
        math.pow(point1[0]-point2[0], 2) + 
        math.pow(point1[1]-point2[1], 2) )
#end of euclidianDistance funcion

def averageOfDistances(array_2d, point):
    sum = 0;
    for x in array_2d:
        dist = euclidianDistance(x,point);
        sum += dist;
    #end of for loop
    average = sum*(1.0)/len(array_2d);
    # print('average is {0}. sum is {1}'.format(average, sum))
    return average; 
#end of averageOfDistances function

def findNewCentroid(centroid_array, data_array):
    average_distances = [];
    for i,x in enumerate(data_array):
        if(isPointInSample(centroid_array, x)):
            # print("skipping point {0} -> centroid array = {1}".format(x, centroid_array))
            continue
        average_dist = averageOfDistances(centroid_array,x);
        average_distances.append((average_dist,x));
    average_distances.sort(key=lambda x:x[0]);
    # print('average distances: {0}'.format(average_distances));
    return average_distances[-1][1];
#end of findNewCentroid function

def getAllCentroids(given_centroids, data_array, k=4):
    for i in range(k-1):
        centroid = findNewCentroid(given_centroids, data_array);
        given_centroids.append(tuple(centroid));
    return given_centroids;
#end of getAllCentroids function

curr_sample = numpy.array([[0,0],[2,1]])

#Testing for isPointInSample function
assert isPointInSample(curr_sample, [0,0])
assert isPointInSample(curr_sample, [2,1])
assert not isPointInSample(curr_sample, [1,1])

#Testing for euclidianDistance function
assert euclidianDistance([0,0],[3,4]) == 5.0
assert euclidianDistance([0,0],[-3,-4]) == 5.0
assert euclidianDistance([0,0],[3,-4]) == 5.0
assert euclidianDistance([0,0],[-3,4]) == 5.0
assert euclidianDistance([0,0],[6,8]) == 10.0 

curr_sample_two = numpy.array([[3,4],[6,8],[9,12]]);
curr_sample_three = numpy.array([[5,0],[0,15]]);
#Testing for averageOfDistances function
assert round(averageOfDistances(curr_sample_two, [0,0]),2) == 10.0;
assert averageOfDistances(curr_sample_three, [0,0]) == 10.0;

#Centroid array for dataset1
centroid_array_test = [[0,0]];
centroid_array_test1 = [[0,0],[0,20],[0,3]];
assert all(findNewCentroid(centroid_array_test, data) == numpy.array([0,20]));
assert all(findNewCentroid(centroid_array_test1, data) == [0,14]);


print(getAllCentroids(centroid_array_test, data))

#Testing assignSamplesToClusters
sample_data = numpy.array([
    [0,0],
    [0,1],
    [0,2],
    [0,3],

    [0,10],
    [0,11],
    [0,12],

    [0,20],
    [0,21],
    [0,22],
    [0,23],

    [0,56],
    [0,57],
    [0,59]
]);

initial_centroids = [(0,2),(0,12),(0,22),(0,56)];

def assignSamplesToClusters(centroid_array, data_array):
    result = {}
    for x in centroid_array:
        result[tuple(x)] = [];
    ####
    for x in data_array:
        best_centroid = findBestCentroid(x,centroid_array);
        result[tuple(best_centroid)].append(x);
    ####
    # print("Resulting clusters: ->");
    # print(result);
    return result;
#end of assignSamplesToArray function

def findBestCentroid(point, centroid_array):
    distances = [];
    for x in centroid_array:
        dist = euclidianDistance(x,point);
        distances.append((dist, x));
    ####
    distances.sort(key=lambda x:x[0]);
    # print("Distances array: ->");
    # print(distances)
    return list(distances[0][1]);
#end of findBestCentroid function

#Testing findBestCentroid function
assert findBestCentroid([0,0], initial_centroids) == [0,2];
assert findBestCentroid([0,11], initial_centroids) == [0,12];
assert findBestCentroid([1,22], initial_centroids) == [0,22];
assert findBestCentroid([0,100], initial_centroids) == [0,56];
assert findBestCentroid([0,-1], initial_centroids) == [0,2];


expected_output = {
    (0,2) : numpy.array([
            [0,0],
            [0,1],
            [0,2],
            [0,3],
            ]),

    (0,12): numpy.array([
            [0,10],
            [0,11],
            [0,12],
            ]),

    (0,22): numpy.array([[0,20],
            [0,21],
            [0,22],
            [0,23]
            ]),
    
    (0,56): numpy.array([
            [0,56],
            [0,57],
            [0,59]
    ])

}

all_assignments = assignSamplesToClusters(initial_centroids, sample_data)

def findMean(array_2d):
    sum_x = 0;
    sum_y = 0;
    for x,y in array_2d:
        sum_x += x;
        sum_y += y;
    mean_x = sum_x*(1.0)/len(array_2d);
    mean_y = sum_y*(1.0)/len(array_2d);
    return [mean_x,mean_y];
#end of findMean function

assert findMean(numpy.array([
            [0,0],
            [0,1],
            [0,2],
            [0,3],
            ])) == [0,1.5]

# def findNewCentroids(initial_centroids, data_array):
#     clusters = assignSamplesToClusters(initial_centroids, data_array);
#     new_centroid_array = [];
#     for k,v in clusters.items():
#         if len(v) == 0:
#             new_centroid = k
#         else:
#             new_centroid = findMean(v);
#         new_centroid_array.append(tuple(new_centroid));
#     new_centroid_array.sort()
#     print("New centroid array = {0}\n\n".format(new_centroid_array))
#     return new_centroid_array;
# ########

def findNewCentroids_new(initial_centroids, data_array):
    clusters = assignSamplesToClusters(initial_centroids, data_array);
    new_centroid_array = [];
    for k,v in clusters.items():
        if len(v) == 0:
            new_centroid = k
        else:
            new_centroid = findMean(v);
        new_centroid_array.append(tuple(new_centroid));
    new_centroid_array.sort()
    # print("New centroid array = {0}\n\n".format(new_centroid_array))
    return [clusters, new_centroid_array];
########

# def run_k_means(n, initial_centroids, sample_data):
#     for i in range(n):
#         print("{0} --> {1}".format(i, initial_centroids))
#         initial_centroids = findNewCentroids(initial_centroids, sample_data)
#     print("Final centroids:->");
#     print(initial_centroids);

def centroidDistanceDiffCloseEnough(newCentroidValues, origCentroidValues):
    comparisonNew = newCentroidValues == origCentroidValues;
    return comparisonNew;

#Testing for centroidDistanceDiffCloseEnough function
arr1 = numpy.array([[1,2],[2,3],[3,4]]);
arr2 = numpy.array([[1,2],[2,3],[3,4]]);
# print("COMPARING 2-D ARRAYS:")
# print(centroidDistanceDiffCloseEnough(arr1,arr2));

def run_k_means_new(n, initial_centroids, sample_data):
    iter_count = 0
    while True:
        iter_count += 1
        # print("{0}".format(initial_centroids))
        clusters, new_centroids = findNewCentroids_new(initial_centroids, sample_data);
        # find the difference between initial and new centriods. If same break
        if centroidDistanceDiffCloseEnough(new_centroids, initial_centroids):
            break
        initial_centroids = new_centroids
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    print("Final centroids: ->");
    print(initial_centroids);
    print("Final loss: ->");
    print(findTotalLoss(clusters));
    return iter_count


def findLossPerCluster(cluster, centroid):
    loss = 0;
    for x in cluster:
        dist = euclidianDistance(x,centroid);
        dist_square = dist*dist;
        loss += dist_square;
    return loss;
####

def findTotalLoss(clusters):
    #clusters in the form - result = {(a,b):[[0,1],[1,1],[1,2]]}
    total_loss = 0;
    for k,v in clusters.items():
        loss = findLossPerCluster(v,list(k));
        total_loss += loss
    ###
    return total_loss;
#####

#Testing findLossPerCluster function

cluster = [[7,1],[2,1],[3,1],[4,1]];
centroid = [3,1];
# print("*********************************");
# print("Given cluster:")
# print("Loss per Cluster: ->");
# print(findLossPerCluster(cluster,centroid));

#Testing findTotalLoss function

#print("*********************************");
# print("Total Loss  for all Clusters: ->");
# print(findTotalLoss(expected_output));

lakshmi_data = numpy.array(
[[ 2.05924902, 7.20598798],
 [ 8.87578072, 8.96092361],
 [ 8.00706441, 2.77531997],
 [ 5.01728788, 3.76311975],
 [ 6.39056222, 5.17956451],
 [ 1.95480368, 7.78421782],
 [ 4.80754093, 3.03464954],
 [ 1.3483716 , 3.96379638],
 [ 3.04101702,-0.36138487],
 [ 8.61947945, 2.98598319],
 [ 6.11106851, 6.23497555],
 [ 3.84278989, 5.53546695],
 [ 1.69565649, 7.68082458],
 [ 4.05095774, 4.05212767],
 [ 2.48989693, 8.40047863],
 [ 6.6384501 , 8.33574252],
 [ 6.6781262 , 1.1080157 ],
 [ 6.2396717 , 6.55049457],
 [ 4.72935154, 3.80839045],
 [ 3.81135136, 5.98125361],
 [ 4.90270653, 3.48642863],
 [ 7.59763505, 7.93924   ],
 [ 7.59731342, 1.16504743],
 [ 2.07898569, 7.16739313],
 [ 7.30246332, 3.16580577],
 [ 6.63352332, 0.98020705],
 [ 3.06996954, 5.97020551],
 [ 4.34489155, 3.99726667],
 [ 3.02105687, 9.26213796],
 [ 2.20011496, 1.53863221],
 [ 6.47098788, 5.4510163 ],
 [ 5.37413088, 5.44219234],
 [ 3.85212146,-1.08715226],
 [ 4.59083727, 7.53490523],
 [ 4.9511002 , 8.08344216],
 [ 1.9311184 , 6.93692984],
 [ 8.21925014, 9.11712554],
 [ 5.27137631, 5.53516715],
 [ 9.26998864, 9.62492869],
 [ 2.37650624, 8.15241778],
 [ 3.2881521 , 0.71796855],
 [ 3.2115245 , 1.1089788 ],
 [ 4.66005931, 7.06059555],
 [ 4.99874427, 2.87525327],
 [ 2.77605992, 2.74592055],
 [ 7.51393398, 1.84048228],
 [ 7.83816267, 2.49139275],
 [ 3.98724311, 4.0425478 ],
 [ 4.91688902, 7.51334885],
 [ 1.66972218, 8.1292424 ],
 [ 6.97690573, 7.96509199],
 [ 2.61234619, 8.39116666],
 [ 5.38398051, 3.53840433],
 [ 2.7845243 , 6.61847158],
 [ 1.20162248, 7.68639714],
 [ 8.07641652, 9.27162002],
 [ 7.68097556, 0.83542043],
 [ 2.04945194, 2.75937105],
 [ 5.91832765, 3.04231385],
 [ 4.91330965, 3.25772425],
 [ 5.14255397, 8.37451307],
 [ 5.36626615, 6.51434231],
 [ 3.04743588, 1.55807635],
 [ 3.13088669, 6.30135711],
 [ 2.36057145, 6.00796623],
 [ 5.40840468, 2.96754178],
 [ 8.37895231, 8.62509614],
 [ 8.20129386, 9.21291541],
 [ 3.9649361 , 5.20027567],
 [ 2.11728317, 6.61574036],
 [ 1.89256383, 3.05142539],
 [ 1.72614408, 6.81819407],
 [ 1.81229618, 3.40781697],
 [ 2.69511302, 5.93967352],
 [ 6.12393256, 5.49223251],
 [ 7.15364076, 2.61344894],
 [ 2.95147442, 7.76615605],
 [ 3.57542555, 5.47748903],
 [ 2.16482565, 7.993515  ],
 [ 6.94511561, 8.30517945],
 [ 2.91008221, 7.51943984],
 [ 2.78903847, 6.44350728],
 [ 2.3537231 , 6.29810755],
 [ 7.88828694, 8.41093125],
 [ 7.85355511, 2.53104656],
 [ 5.68766272, 5.38279515],
 [ 2.47238755, 3.7285616 ],
 [ 2.33338702, 7.23913284],
 [ 7.35456962, 0.93930822],
 [ 2.70699582, 1.64002569],
 [ 4.94956074, 3.22624797],
 [ 5.2979492 , 3.65258141],
 [ 6.46350009, 0.77471754],
 [ 1.79534908, 3.7348206 ],
 [ 6.40483149, 5.60578084],
 [ 7.57805025, 3.82487017],
 [ 2.87448907, 2.657599  ],
 [ 7.94375954, 8.21165063],
 [ 6.2153903 , 6.26139225],
 [ 8.22627485, 2.26048701],
 [ 7.89366657, 3.58341277],
 [ 5.57009665, 8.3870942 ],
 [ 6.8113456 , 0.99804859],
 [ 2.40998489, 7.99174945],
 [ 2.68080913, 1.61298226],
 [ 3.14009486, 0.34589487],
 [ 8.1118272 , 3.27768018],
 [ 5.07250754, 7.89834048],
 [ 5.08001625, 3.25348762],
 [ 7.9628009 , 2.81761275],
 [ 7.39015357, 1.13206806],
 [ 7.1712312 , 5.16316266],
 [ 2.10606162, 8.23183769],
 [ 4.84306328, 7.50757895],
 [ 7.33424973, 2.97894225],
 [ 3.40504475, 1.04980673],
 [ 8.09209017, 3.39065059],
 [ 8.21897526, 8.9510505 ],
 [ 2.80096609, 1.03176348],
 [ 8.527899  , 8.55183237],
 [ 8.75754845, 8.81745441],
 [ 4.95728696, 6.90897984],
 [ 6.79251832, 2.56208095],
 [ 6.8150111 , 2.13543395],
 [ 7.80003043, 1.90963115],
 [ 4.96433498, 7.88753239],
 [ 8.9702889 , 3.32150578],
 [ 4.6733967 , 7.14753742],
 [ 2.14633887, 8.83030888],
 [ 3.49606966, 5.79440796],
 [ 6.48423011, 5.04416608],
 [ 2.81629029, 3.1999725 ],
 [ 5.77144223, 9.04075394],
 [ 6.85653225, 7.72468825],
 [ 7.72715541, 7.62018213],
 [ 8.60402994, 8.76147163],
 [ 7.25412082, 2.77862318],
 [ 2.65875751, 1.7541119 ],
 [ 6.6161895 , 0.66750633],
 [ 2.16641743, 2.99414637],
 [ 6.8950152 , 0.95350302],
 [ 4.62125558, 7.81235824],
 [ 5.52279832, 5.52162016],
 [ 3.03696341, 5.82211317],
 [ 4.30954572, 6.96097943],
 [ 5.60001917, 3.02332715],
 [ 5.68845261, 8.27229082],
 [ 3.24569117, 8.79368337],
 [ 1.92561853, 2.73857632],
 [ 4.40304734, 4.81434354],
 [ 8.26213369, 3.53415034],
 [ 3.18340392, 5.42184013],
 [ 2.25790845, 7.44778003],
 [ 3.39448396, 0.63811821],
 [ 5.33498937, 3.07430754],
 [ 7.06572   , 2.08940967],
 [ 4.47456424, 4.0283604 ],
 [ 4.43990951, 3.70495907],
 [ 4.10720306, 0.25056515],
 [ 6.4095594 , 5.35040201],
 [ 4.74683942, 8.03399056],
 [ 6.05509889, 7.23007608],
 [ 4.68128498, 3.98291658],
 [ 5.60944242, 2.91327032],
 [ 7.77126987, 8.91428052],
 [ 5.48121965, 6.55171777],
 [ 6.5807212 ,-0.0766824 ],
 [ 8.33664582, 9.23795257],
 [ 1.713841  , 4.31350258],
 [ 3.12073696, 0.48979079],
 [ 5.30543981, 3.39751664],
 [ 3.75004647, 4.90070114],
 [ 1.73949419, 7.46085844],
 [ 7.52095236, 8.80020339],
 [ 7.90345455, 2.28430161],
 [ 3.54461267, 0.94261882],
 [ 2.73285832, 2.83024707],
 [ 2.46087695, 6.86898874],
 [ 3.01047612, 6.54286455],
 [ 3.89523379, 0.70718356],
 [ 2.06136024, 3.54047797],
 [ 8.12343078, 2.60762469],
 [ 7.22537424, 8.46609363],
 [ 7.52963009, 8.79617112],
 [ 2.64145141, 2.62206822],
 [ 7.45225989, 2.26860809],
 [ 6.03237178, 8.86195452],
 [ 1.91975789, 7.2336754 ],
 [ 2.6446214 , 5.5279038 ],
 [ 1.52668895, 4.24557918],
 [ 5.14167285, 5.71626939],
 [ 2.95297924, 9.65073899],
 [ 6.2091503 , 6.16038763],
 [ 7.12751003, 1.23747391],
 [ 4.97304553, 7.4290438 ],
 [ 8.21026885, 8.18439548],
 [ 4.40450545, 6.75422193],
 [ 3.24516611, 0.8218365 ],
 [ 5.07631894, 3.30296197],
 [ 3.0093283 , 1.45065717],
 [ 2.23518365, 3.77218252],
 [ 7.93019866, 8.14006634],
 [ 8.09469345, 7.79199846],
 [ 3.79752017, 0.69134312],
 [ 2.4817742 , 1.67402547],
 [ 1.05217427, 3.88943741],
 [ 4.21807424, 4.26660054],
 [ 7.39793659, 2.19143804],
 [ 7.41668593, 3.16558163],
 [ 6.39627447, 1.24125663],
 [ 1.76666071, 4.41759655],
 [ 5.32508246, 7.68399917],
 [ 1.51180219, 7.48293717],
 [ 2.58046907, 6.53023549],
 [ 5.17889443, 8.78645074],
 [ 2.44868927, 2.55261552],
 [ 3.66118224,-0.63372377],
 [ 1.96079533, 3.68536495],
 [ 1.89785053, 3.50014156],
 [ 3.0226944 , 0.86402039],
 [ 8.06160243, 4.04423262],
 [ 3.25224641, 2.4788534 ],
 [ 4.7585105 , 8.24317459],
 [ 6.76851611, 1.38337541],
 [ 2.77208277, 5.87425986],
 [ 7.10604472, 1.19751007],
 [ 7.95300821, 3.1028738 ],
 [ 1.91568768, 6.83080871],
 [ 2.0614632 , 8.22584366],
 [ 8.53986559, 3.38241162],
 [ 6.09952696, 9.0178614 ],
 [ 1.76496239, 6.98004057],
 [ 4.30228618, 7.08489147],
 [ 7.67406359, 7.37819153],
 [ 8.44178587, 2.18453296],
 [ 6.99180377, 5.7932428 ],
 [ 2.18568667, 3.11739024],
 [ 4.89972495, 7.37650893],
 [ 3.02640736, 5.74083968],
 [ 6.90753101, 8.3019514 ],
 [ 8.46078528, 2.85204573],
 [ 6.47011829, 5.54035543],
 [ 7.93432052, 8.17735191],
 [ 5.04470093, 8.49060119],
 [ 4.4280969 , 7.41377907],
 [ 2.97661653, 6.01021497],
 [ 1.89404312, 3.36258443],
 [ 4.50236445, 2.9288804 ],
 [ 3.2492998 , 5.59125171],
 [ 6.46270852, 5.83507122],
 [ 7.60284588, 0.778726  ],
 [ 3.72610844, 5.20432439],
 [ 2.61508272, 3.80685209],
 [ 2.5366924 , 2.24222672],
 [ 5.74511019, 5.32034026],
 [ 3.09999409, 0.8385499 ],
 [ 4.74625798, 3.54661053],
 [ 8.67805277, 9.08757916],
 [ 3.12914724, 3.40388727],
 [ 3.32202131, 6.15602339],
 [ 6.6113666 , 4.57049451],
 [ 4.95185958, 4.11756694],
 [ 5.09046134, 8.01800423],
 [ 3.35409838, 5.79603723],
 [ 3.52782703, 5.73858063],
 [ 8.03150205, 8.88381354],
 [ 3.81485895, 6.91844078],
 [ 2.10054891, 1.44144019],
 [ 3.08143147, 2.18786562],
 [ 5.02471033, 8.23879873],
 [ 2.38952606, 7.22195564],
 [ 2.97097541, 2.39669382],
 [ 4.5872861 , 7.29024049],
 [ 1.87131855, 3.43365823],
 [ 7.56399709, 7.83135288],
 [ 5.25103144, 8.74212485],
 [ 8.36230458, 3.08961725],
 [ 2.51555209, 6.41071774],
 [ 5.14468217, 3.26589278],
 [ 1.96633923, 7.30845038],
 [ 7.44472802, 2.41849318],
 [ 6.92525072, 2.26330209],
 [ 1.77775261, 7.21854537],
 [ 7.75261716, 8.67289362],
 [ 6.90743481, 6.00718092],
 [ 4.50496872, 4.7214697 ],
 [ 6.65537695, 1.30451652],
 [ 8.22144628, 8.60551337],
 [ 7.74867074, 1.71812324],
 [ 2.3085098 , 7.39324133],
 [ 4.75184863, 4.20214023],
 [ 3.53350737, 0.33198894],
 [ 6.60277235, 6.31081582],
 [ 9.21069612, 4.5106493 ],
 [ 5.6651354 , 2.7313015 ],
 [ 4.78363211, 7.10644288],
 [ 2.64683045, 6.32344268],
 [ 4.32239695, 0.33088885],
 [ 2.36430335, 1.05209713],
 [ 7.78551305, 3.12724529],]



);

import random
#lakshmi_initial_centroids =  random.sample([(x,y) for x,y in lakshmi_data[:]], 4)  #[(100,100),(1,1),(12,2),(13,3)]
#### Final centroids for dataset1 #####
lakshmi_initial_centroids_one =  getAllCentroids([(2.95147442 ,7.76615605)], lakshmi_data, 4)
#lakshmi_initial_centroids_one =  getAllCentroids([(4.21807424  , 4.26660054)], lakshmi_data, 4)
print("@@@@@@@@@@@@@@@@");
print("STRATEGY 2");
print("Initial Centroids = {0}".format(lakshmi_initial_centroids_one))
print("Final centroids and loss for dataset1 ->")
number_of_iter_run_strategy2_ds1 = run_k_means_new(5, lakshmi_initial_centroids_one, lakshmi_data);

##### Final centroids for dataset2 #####
lakshmi_initial_centroids_two =  getAllCentroids([(8.21925014, 9.11712554)], lakshmi_data, 6)
#lakshmi_initial_centroids_two =  getAllCentroids([ (4.32239695 , 0.33088885)], lakshmi_data, 6)
print("@@@@@@@@@@@@@@@@");
print("STRATEGY 2");
print("Initial Centroids = {0}".format(lakshmi_initial_centroids_two))
print("Final centroids and loss for dataset2 ->")
number_of_iter_run_strategy2_ds2 = run_k_means_new(5, lakshmi_initial_centroids_two, lakshmi_data);

##############################################################################################################################
########## FOR STRATEGY 1 ############################
##### Final centroids for dataset1 for Strategy1 #####
k1_strategy1 = 3;
initial_centroids_one = [[ 2.07898569,  7.16739313],
 [ 8.20129386,  9.21291541],
 [ 8.9702889,   3.32150578]];
print("@@@@@@@@@@@@@@@@");
print("STRATEGY 1");
print("Final centroids and loss for dataset1 ->")
number_of_iter_run_strategy1_ds1 = run_k_means_new(5, initial_centroids_one, lakshmi_data);

k2_strategy2 = 5;
initial_centroids_two = [[ 6.11106851,  6.23497555],
 [ 7.60284588,  0.778726  ],
 [ 2.51555209,  6.41071774],
 [ 1.3483716,   3.96379638],
 [ 2.18568667,  3.11739024]];
print("@@@@@@@@@@@@@@@@");
print("STRATEGY 1");
print("Final centroids and loss for dataset2 ->")
number_of_iter_run_strategy1_ds2 = run_k_means_new(5, initial_centroids_two, lakshmi_data);

###############################################################################################################################


# #### Testing using sklearn for dataset1 #####
# from sklearn.cluster import KMeans;
# nd_array_centroids_one = numpy.array(lakshmi_initial_centroids);

# kmeans_one = KMeans(4,nd_array_centroids_one,10).fit(lakshmi_data);
# print("kmeans_one:");
# print(kmeans_one)
# final_centroids_one = kmeans_one.cluster_centers_;
# print("Final centroids for *dataset1* using sklearn: ->");
# print(final_centroids_one);

#### Testing using sklearn for dataset2 #####
from sklearn.cluster import KMeans;

def runSklearnKMeans(inital_centroid_data, k, iter_count, dataSetName):
    print("#### {0} #####".format(dataSetName))
    nd_array_centroids_two = numpy.array(inital_centroid_data);

    kmeans_two = KMeans(k,nd_array_centroids_two, iter_count).fit(lakshmi_data);
    print("kmeans_two:");
    print(kmeans_two)
    final_centroids_two = kmeans_two.cluster_centers_;
    print("Final centroids for *dataset2* using sklearn: ->");
    print(final_centroids_two);
    print("Final Loss:");
    print(kmeans_two.inertia_);
    print("\n\n")


runSklearnKMeans(lakshmi_initial_centroids_one, k = 4, iter_count = number_of_iter_run_strategy2_ds1,dataSetName = "S2-DS1")
runSklearnKMeans(lakshmi_initial_centroids_two, k = 6, iter_count = number_of_iter_run_strategy2_ds2, dataSetName ="S2-DS2")
runSklearnKMeans(initial_centroids_one, k = 3, iter_count = number_of_iter_run_strategy1_ds1, dataSetName ="S1-DS1")
runSklearnKMeans(initial_centroids_two, k = 5, iter_count = number_of_iter_run_strategy1_ds2, dataSetName ="S1-DS2")