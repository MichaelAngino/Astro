import numpy as np
import math
# import sklearn as sk
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class Point:
    """
    Object representing a point in Space
    """


    def __init__(self, label, actual_pollution_value, x):
        """
        Initalizes new point Object
        :param label: Label for point
        :param actual_pollution_value: True value of pollution at the point
        :param x: X coordinate of the point
        """
        self.label = label
        # self.actual_pollution_value = actual_pollution_value
        self.pollution_value = float('NaN')

        self.actual_pollution_value = actual_pollution_value

        self.x = x

        self.y = 0



    def __str__(self):
        return "[Point: " + str(self.label) + "| APV: " + str(self.actual_pollution_value)+ "| PV: "+ str(self.pollution_value) +"| Position: {" + str(self.x)+ "," + str(self.y)+ "} ]"

    def get_pollution_value(self):
        """
        Gets measured pollution value, also stores the value of the interpolated pollution value
        :return: Pollution_value (float)
        """
        return self.pollution_value

    def set_pollution_value(self, new_pollution_value):
        """
        Sets measured pollution value or to store the interpolated pollution value
        :param new_pollution_value: New pollution value to be set
        :return: Nothing
        """
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
        """
        Does a copy of the point
        :return: A new copied point
        """
        temp = Point(self.label,self.get_actual_pollution_value(),self.x)
        temp.set_pollution_value(self.get_pollution_value())
        return temp

    def get_actual_pollution_value(self):
        return self.actual_pollution_value

def copy_dictionary_with_points(points):
    """
    Does a deep copy of a dictionary with values being points
    :param points: A dictionary with keys being labels and values being points {label:point}
    :return: A new copied dictionary
    """
    new_dict = {}
    for key, value in points.items():
        new_dict[key] = value.copy()

    return new_dict

def create_points_with_random_pollution(length, mean, std):
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

    random_picks = []

    while len(random_picks) < pick_number:
        random_num = random.integers(0,len(points)) # picks uniform random number
        if not random_num in random_picks:
            random_picks.append(random_num)
    new_map  = {}
    for i in random_picks: #assigns and copies picked points into a new dictionary
        new_map[i] = Point(i,points.get(i).get_actual_pollution_value(),points.get(i).get_x_cord())
        new_map.get(i).read_pollution_value()

    return new_map

def interpolate_unknown_points(known_points, all_points):
    """
    Interpolate pollution values for points that are have not been measured
    :param known_points: A Dictionary of all points that have been measured {label:point}
    :param all_points: A Dictionary of all points that exist {Label: point}
    :return: A new map with interpolated pollution values {Label : point}
    """
    unkown_posistions = []
    unkown_labels = []
    for i in range(0,len(all_points)): #Creates a list of posistions of points that have not been measured and a list of their respective labels
        if not (i in known_points):
            unkown_posistions.append(all_points.get(i).get_position())
            unkown_labels.append(i)

    interpolated_pollution_values = interpolate_points_using_posistions(known_points, unkown_posistions) #interpolates the pollution values for the posistions where we have not measured pollution values

    interpolated_map = copy_dictionary_with_points(known_points) #creates a copy of the known_points dictionary

    for i in range(0,len(unkown_labels)): # adds missing points and their interpolated pollution values into the new dictionary
        interpolated_map[unkown_labels[i]] = all_points.get(unkown_labels[i]).copy()
        interpolated_map[unkown_labels[i]].set_pollution_value(interpolated_pollution_values[i])

    return interpolated_map


def to_list_of_posistions(points):
    """
    Converts a dictionary of points into a list of posistions [[x1,y1],[x2,y2]]
    :param points: A dictionary of poitns {label : point}
    :return: A list of posistions of the points [[x1,y1],[x2,y2]]
    """
    points_position_list = []  # converts point list into a list of [x,y] values
    for point in points.values():
        points_position_list.append(point.get_position())

    return points_position_list




print( Point(1,1,1))
#
#
# random_points1 = create_points_with_random_pollution(10, 100, 10)
# p = pick_uniform_random_points(random_points1,5)
#
# a = interpolate_unknown_points(p,random_points1)







