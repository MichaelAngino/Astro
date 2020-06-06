import numpy as np
# import sklearn as sk
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class Point:
    """
    Object representing a point in Space
    """


    def __init__(self, label, actual_pollution_value, x):
        """
        Creates a Point Object
        :param label: Label of point ex:005, 015, 995
        :param actual_pollution_value: Actual value of the pollution measurment
        """
        self.label = label
        # self.actual_pollution_value = actual_pollution_value
        self.pollution_value = actual_pollution_value

        self.x = x

        self.y = 0


    def __str__(self):
        return "[Point: " + str(self.label) + "| APV: " + str(self.pollution_value)+"| Position: {" + str(self.x)+ "," + str(self.y)+ "} ]"

    def get_pollution_value(self):
        return self.pollution_value

    def set_pollution_value(self, new_pollution_value):
        self.pollution_value = new_pollution_value

    def get_label(self):
        return self.label

    def get_x_cord(self):
        return self.x

    def get_position(self):
        """
        Returns Position of the point
        :return: list [x,y]
        """
        return [self.x,self.y]


def create_map_of_random_pollution_points(length, mean, std):
    """
    Creates a map of points with random Pollution Values using a gaussian distribution in 1D evenly starting at x = 05 every 10
    :param length: Wanted length of list
    :param mean: Mean of gaussian distribution of pollution values
    :param std: Standard Deviation of gaussian distribution of pollution values
    :return: map of points with key being their label and the value being a point object
    """
    new_map = {}
    x = 5
    for i in range(0,100):
        new_map[i] = (Point(i,np.random.normal(mean,std),x))
        x = x+10
    return new_map

def predict_point(known_points, wanted_point_positions):
    """
    Predicts points based on known data using Kriging (Gaussian Processes)
    :param known_points: list of points
    :param wanted_point_positions: list of wanted point posistions [[x1,y1], [x2,y2]]
    :return: a list of all predicted pollution values
    """


    gp = GaussianProcessRegressor(C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))) # Instantiate a Gaussian Process model


    known_points_position_list = []  #converts point list into a list of [x,y] values
    for point in known_points:
        known_points_position_list.append(point.get_position())

    pollution_value_list = [] #converts point list into a list of pollution values
    for point in known_points:
        pollution_value_list.append(point.get_pollution_value())

    gp.fit(known_points_position_list,pollution_value_list) #Fits model to data


    prediction_list = known_points_position_list
    prediction_list.extend(wanted_point_positions)
    return gp.predict(prediction_list)[len(known_points):] # predicts on new data









random_points = create_map_of_random_pollution_points(100,100,10).values()
test = []
for i in random_points:
    test.append(i.get_pollution_value())

testNP = np.array(test)
prediction = predict_point(random_points,[[100,0],[3,0]])
prediction_list = prediction.tolist()
print(prediction_list)


