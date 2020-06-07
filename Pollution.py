import numpy as np
import math
# import sklearn as sk
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class Point:
    """
    Object representing a point in Space
    """


    def __init__(self, label, actual_pollution_value, x, pollution_value = float('NaN')):
        """
        Creates a Point Object
        :param label: Label of point ex:005, 015, 995
        :param actual_pollution_value: Actual value of the pollution measurment
        """
        self.label = label
        # self.actual_pollution_value = actual_pollution_value
        self.pollution_value = pollution_value

        self.actual_pollution_value = actual_pollution_value

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

    def read_pollution_value(self):
        """
        Generates pollution value using actual pollution value. Currently it just assigns pollution value = actual_polution_value
        :return: Nothing
        """
        if not math.isnan(self.pollution_value):
            raise Exception("Just tried to read a pollution value from a point with a assigned pollution_value")
        self.pollution_value = self.actual_pollution_value

    def copy(self):
        return Point(self.label,self.get_pollution_value(),self.x, self.get_pollution_value())

    def get_actual_pollution_value(self):
        return self.actual_pollution_value

def copy_dictionary_with_points(points):
    new_dict = {}
    for key, value in points.items():
        new_dict[key] = value.copy()

    return new_dict

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
    for i in range(0,length):
        new_map[i] = (Point(i,np.random.normal(mean,std),x))
        x = x+10
    return new_map

def interpolate_points_using_posistions(known_points, wanted_point_positions):
    """
    Predicts points based on known data using Kriging (Gaussian Processes)
    :param known_points: list of points
    :param wanted_point_positions: list of wanted point posistions [[x1,y1], [x2,y2]]
    :return: a list of all predicted pollution values
    """


    gp = GaussianProcessRegressor(C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))) # Instantiate a Gaussian Process model


    known_points_position_list = to_list_of_posistions(known_points)

    pollution_value_list = [] #converts point list into a list of pollution values
    for point in known_points.values():
        pollution_value_list.append(point.get_pollution_value())

    gp.fit(known_points_position_list,pollution_value_list) #Fits model to data


    prediction_list = known_points_position_list
    prediction_list.extend(wanted_point_positions)
    return gp.predict(prediction_list)[len(known_points):] # predicts on new data

def pick_uniform_random_points(points, pick_number ):
    """
    Picks a number of points from a list of points using a uniform random distribution
    :param points: Map of all the points {label: point}
    :param pick_number: Number of points to pick
    :return: A new map with the picked points {label: point}
    """
    random = np.random.default_rng()
    random_picks = random.integers(0,len(points),pick_number)

    new_map  = {}
    for i in random_picks:
        new_map[i] = Point(i,points.get(i).get_actual_pollution_value(),points.get(i).get_x_cord())
        new_map.get(i).read_pollution_value()

    return new_map

def interpolate_unknown_points(known_points, all_points):
    unkown_posistions = []
    unkown_labels = []
    for i in range(0,len(all_points)):
        if not (i in known_points):
            unkown_posistions.append(all_points.get(i).get_position())
            unkown_labels.append(i)

    interpolated_pollution_values = interpolate_points_using_posistions(known_points, unkown_posistions)

    interpolated_map = copy_dictionary_with_points(known_points)

    for i in range(0,len(unkown_labels)):
        interpolated_map[unkown_labels[i]] = all_points.get(unkown_labels[i]).copy()
        interpolated_map[unkown_labels[i]].set_pollution_value(interpolated_pollution_values[i])

    return interpolated_map


def to_list_of_posistions(points):
    points_position_list = []  # converts point list into a list of [x,y] values
    for point in points.values():
        points_position_list.append(point.get_position())

    return points_position_list





random_points1 = create_map_of_random_pollution_points(10,100,10)
p = pick_uniform_random_points(random_points1,5)

a = interpolate_unknown_points(p,random_points1)



random_points = create_map_of_random_pollution_points(10,100,10).values()


test = []
for i in random_points:
    test.append(i.get_pollution_value())

testNP = np.array(test)
prediction = interpolate_points_using_posistions(random_points, [[100, 0], [3, 0]])
prediction_list = prediction.tolist()
print(prediction_list)



