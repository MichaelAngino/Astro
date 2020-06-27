import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import animation
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct as DP
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from mpl_toolkits.mplot3d import Axes3D


# test comment

# np.random.multivariate_normal([0, 0], [[1,1], [1,1]] , 1)[0]
# with vars names: np.random.multivariate_normal(mean_vector_of_each_point=[0, 0], COV=[[1,1], [1,1]] , nb_maps=1)[id_map=0]

class Point:
    """
    Object representing a point in Space
    """

    def __init__(self, label, actual_pollution_value, x, y=0):
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

        self.y = y

    def __str__(self):
        return "[Point: " + str(self.label) + "| APV: " + str(self.actual_pollution_value) + "| PV: " + str(
            self.pollution_value) + "| Position: {" + str(self.x) + "," + str(self.y) + "} ]"

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

    def get_y_cord(self):
        return self.y

    def get_position(self):
        """
        Returns Position of the point
        :return: list [x,y]
        """
        return [self.x, self.y]

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
        temp = Point(self.label, self.get_actual_pollution_value(), self.x, self.y)
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


def distance(first_point, second_point):
    return math.sqrt(math.pow(first_point.x - second_point.x, 2) + math.pow(first_point.y - second_point.y, 2))


def create_points_with_random_pollution_1d(length, mean, std):
    """
    Creates a map of points with random Pollution Values using a gaussian distribution in 1D evenly starting at x = 05 every 10
    :param length: Wanted length of list
    :param mean: Mean of gaussian distribution of pollution values
    :param std: Standard Deviation of gaussian distribution of pollution values
    :return: map of points with key being their label and the value being a point object
    """
    new_map = {}
    x = 5
    for i in range(0, length):
        new_map[i] = (Point(i, np.random.normal(mean, std), x))
        x = x + 10
    return new_map


def create_points_with_random_pollution_2d(side_length, mean, std):
    """
    Creates a map of points with random Pollution Values using a gaussian distribution in 2D evenly starting at x = 05 every 10
    :param side_length: Number of points on each side of the square
    :param mean: Mean of gaussian distribution of pollution values
    :param std: Standard Deviation of gaussian distribution of pollution values
    :return: map of points with key being their label and the value being a point object
    """
    new_map = {}
    x = 5

    label_index = 0
    for i in range(0, side_length):
        y = 5
        for j in range(0, side_length):
            new_map[label_index] = (Point(label_index, np.random.normal(mean, std), x, y))
            label_index = label_index + 1
            y = y + 10
        x = x + 10
    return new_map


def create_points_with_spatially_correlated_pollution_2d(side_length, mean, std_dev, length_scale, num_maps, ):
    """
    Creates a map of pollution values determined by their spatial correlation
    :param side_length: Number of points on each side of the matrix
    :param mean: Mean of the gaussian distribution of pollution values
    :param std_dev: Standard Deviation of the pollution values
    :param length_scale: The length scale of the RBF kernel
    :param num_maps: Number of pollution maps used in data
    """
    # with vars names: np.random.multivariate_normal(mean_vector_of_each_point=[0, 0], COV=[[1,1], [1,1]] , nb_maps=1)[id_map=0]
    mean_vector = []
    for i in range(side_length * side_length):
        mean_vector.append(mean)
    pollution_maps = {}

    point_map = {}
    x = 5
    label_index = 0
    for i in range(0, side_length):
        y = 5
        for j in range(0, side_length):
            point_map[label_index] = Point(label_index, 'NaN', x, y)
            label_index += 1
            y += 10
        x += 10
    for i in range(0, num_maps):
        pollution_maps[i] = copy_dictionary_with_points(point_map)

    covariance_matrix = create_covariance_matrix(point_map, std_dev, length_scale)
    maps_of_pollution_values = np.random.multivariate_normal(mean_vector, covariance_matrix, num_maps)

    for i in range(num_maps):
        for j in range(side_length * side_length):
            pollution_maps[i][j].actual_pollution_value = maps_of_pollution_values[i][j]

    return pollution_maps


def create_covariance_matrix(points, length_scale,std_dev):
    """
    Creates Covariance Matrix
    :param points: Map of all points
    :param length_scale: Length Scale
    :return: A matrix
    """

    covariance = []
    for i in range(0, len(points)):
        covariance.append([])
        for j in range(0, len(points)):
            # covariance[i].append(
            #     np.exp(-np.power(distance(points[i], points[j]), 2) / (2 * length_scale * length_scale)))
            covariance[i].append((std_dev ** 2) * np.exp(
                -np.power(distance(points[i], points[j]), 2) / (2 * length_scale * length_scale)))

    return covariance


def interpolate_points_using_positions(known_points, wanted_point_positions, kernel= None,
                                       fixed=False):
    """
     Predicts points based on known data using Kriging (Gaussian Processes)
    :param known_points: list of points
    :param wanted_point_positions: list of wanted point posistions [[x1,y1], [x2,y2]]
    :param kernel:  Kernal to use in interpolation
    :param fixed:  True = no opitimization of hyperparamater, False = optimization of hyperparamter
    :return: a list of all predicted pollution values
    """

    # kernel = DP(1)
    # kernel = RBF(10, (1e-2, 1e2)) * C(1)

    if fixed:
        gp = GaussianProcessRegressor(kernel, alpha = .001, n_restarts_optimizer=10,
                                      optimizer=None)  # Instantiate a fixed Gaussian Process model
    else:
        gp = GaussianProcessRegressor(kernel, alpha=.001, n_restarts_optimizer=10)  # Instantiate an optimized Gaussian Process model

    known_points_position_list = to_list_of_positions(known_points)

    pollution_value_list = []  # converts point list into a list of pollution values
    for point in known_points.values():
        pollution_value_list.append(point.get_pollution_value())

    gp.fit(known_points_position_list, pollution_value_list)  # Fits model to data

    #
    # prediction_list = known_points_position_list
    # prediction_list.extend(wanted_point_positions)
    # return gp.predict(prediction_list)[len(known_points):]  # predicts on new data

    return (gp.predict(wanted_point_positions), gp.kernel_.length_scale)


def interpolate_unknown_points(known_points, all_points, kernel=None, fixed=False):
    """
    Interpolate pollution values for points that are have not been measured
    :param known_points: A Dictionary of all points that have been measured {label:point}
    :param all_points: A Dictionary of all points that exist {Label: point}
    :param kernel:  Kernal to use in interpolation
    :param fixed:  True = no opitimization of hyperparamater, False = optimization of hyperparamter
    :return: A new map with interpolated pollution values {Label : point}
    """
    unknown_positions = []
    unknown_labels = []
    for i in range(0, len(
            all_points)):  # Creates a list of posistions of points that have not been measured and a list of their respective labels
        if not (i in known_points):
            unknown_positions.append(all_points.get(i).get_position())
            unknown_labels.append(i)

    interpolated_pollution_values, length_scale = interpolate_points_using_positions(known_points,
                                                                                     unknown_positions,
                                                                                     kernel,
                                                                                     fixed)  # interpolates the pollution values for the posistions where we have not measured pollution values

    interpolated_map = copy_dictionary_with_points(known_points)  # creates a copy of the known_points dictionary

    for i in range(0, len(
            unknown_labels)):  # adds missing points and their interpolated pollution values into the new dictionary
        interpolated_map[unknown_labels[i]] = all_points.get(unknown_labels[i]).copy()
        interpolated_map[unknown_labels[i]].set_pollution_value(interpolated_pollution_values[i])

    return (interpolated_map, length_scale)


def interpolate_unknown_points_of_a_map_of_maps_of_points(known_points, all_points, kernel=None, fixed=False):
    """

    :param known_points: A map of maps of all points that have been measured
    :param all_points:  A  map of maps of all points that exist
    :param kernel:  Kernal to use in interpolation
    :param fixed:  True = no opitimization of hyperparamater, False = optimization of hyperparamter
    :return: A tuple of (A new map of maps of interpolated values, length of kernel)
    """

    interpolated_maps = {}
    for label in all_points.keys():
        interpolated_maps[label] = interpolate_unknown_points(known_points[label], all_points[label], kernel, fixed)

    return interpolated_maps


def pick_uniform_random_points_on_map_of_maps(points, pick_number):
    """
        Picks a number of points from a list of points using a uniform random distribution for each map in a map
        :param points: Map of all the points {label: point}
        :param pick_number: Number of points to pick
        :return: A new map with the picked points {label: point}
        """

    new_map = {}
    for label in points.keys():
        new_map[label] = pick_uniform_random_points(points[label], pick_number)

    return new_map


def pick_uniform_random_points(points, pick_number):
    """
    Picks a number of points from a list of points using a uniform random distribution
    :param points: Map of all the points {label: point}
    :param pick_number: Number of points to pick
    :return: A new map with the picked points {label: point}
    """
    random = np.random.default_rng()

    random_picks = []

    while len(random_picks) < pick_number:
        random_num = random.integers(0, len(points))  # picks uniform random number
        if not random_num in random_picks:  # Checks to make sure number hasnt been picked yet
            random_picks.append(random_num)
    new_map = {}
    for i in random_picks:  # assigns and copies picked points into a new dictionary
        new_map[i] = Point(i, points.get(i).get_actual_pollution_value(), points.get(i).get_x_cord(),
                           points.get(i).get_y_cord())
        new_map.get(i).read_pollution_value()

    return new_map


def pick_poisson_random_points(points, pick_number, lam):
    """
    Picks random points using poisson distribution
    :param points: All possible points
    :param pick_number: Number of points to pick
    :param lam: Lambda variable for the Poisson Function (typically a number)
    :return: A dictionary of points
    """
    random = np.random.default_rng()

    random_picks = []

    while len(random_picks) < pick_number:
        random_num = random.poisson(lam)  # picks poisson random number
        if not random_num in random_picks and 0 < random_num < len(
                points):  # Checks to make sure number hasnt been picked yet and is within range
            random_picks.append(random_num)
    new_map = {}
    for i in random_picks:  # assigns and copies picked points into a new dictionary
        new_map[i] = Point(i, points.get(i).get_actual_pollution_value(), points.get(i).get_x_cord())
        new_map.get(i).read_pollution_value()

    return new_map


def to_list_of_positions(points):
    """
    Converts a dictionary of points into a list of posistions [[x1,y1],[x2,y2]]
    :param points: A dictionary of points {label : point}
    :return: A list of positions of the points [[x1,y1],[x2,y2]]
    """
    points_position_list = []  # converts point list into a list of [x,y] values
    for point in points.values():
        points_position_list.append(point.get_position())

    return points_position_list


def root_mean_square_error(points):
    """
    Finds the RMSE of the interpolated pollution values
    param points: A dictionary of points {label : point}
    :return: The RMSE of the pollution values found through interpolation
    """
    sum = 0
    for point in points.values():
        sum += pow(point.get_pollution_value() - point.get_actual_pollution_value(), 2)
    rmse = math.sqrt(sum / len(points))
    return rmse


def average_rmse_of_maps(maps_of_points):
    num_of_maps = len(maps_of_points)
    sum = 0
    for label in maps_of_points.keys():
        sum += root_mean_square_error(maps_of_points[label][0])
    return sum / num_of_maps


def plot_numbers(x_axis, y_axis, x_axis_2, y_axis_2, x_label, y_label, x_log_scale = False):
    """
    Plots Numbers on a graph
    :param y_label: Label for Y-axis of graph
    :param x_label: Label for X-axis of graph
    :param x_axis, x_axis_2: X-value for the graph in list form
    :param y_axis, y_axis_2: Y-values for the graph in list form
    :return:
    """
    plt.plot(x_axis, y_axis, "ro", x_axis_2, y_axis_2, "go")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if x_log_scale:
        plt.xscale("log")

    plt.show()


def run_interpolations_with_random_betas_1d():
    """
    Runs Interpolation with random picking of the value of Beta
    """
    rmse_values = []
    picked_points_values = []
    number_of_times = 100  # this will be the number of points displayed on the graph
    random = np.random.default_rng()

    for i in range(0, number_of_times):
        test_points = create_points_with_random_pollution_1d(100, 100, 10)
        test_picked_points = pick_uniform_random_points(test_points, random.integers(1,
                                                                                     100))  # picks a random number of known points
        test_interpolated_points = interpolate_unknown_points(test_picked_points, test_points)
        test_rmse = root_mean_square_error(test_interpolated_points)
        rmse_values.append(test_rmse)
        picked_points_values.append(len(test_picked_points))

    plot_numbers(rmse_values, picked_points_values)


def run_interpolation_with_various_betas(points, kernel=RBF(10, (1e-2, 1e2)) * C(1)):
    """
    Runs Interpolation with number of picked points(beta) from 1 to all points-1 picked and using uniform distribution in
    the picking in 1 Dimesnsion
    """

    rmse_data = []

    for i in range(1, len(
            points)):  # runs through all number of picked points starting at 1 and ending with all points picked-1
        sum_rmse = 0
        for j in range(0, 3):  # runs every interpolation with a certain beta 5 times and averages the results
            picked_points = pick_uniform_random_points(points, i)
            interpolated_points = interpolate_unknown_points(picked_points, points, kernel)
            sum_rmse = sum_rmse + root_mean_square_error(interpolated_points)
        rmse_data.append(sum_rmse / 5)

    plot_numbers(rmse_data, range(1, len(points)))

    return rmse_data


def see_what_its_doing_1d():
    """
    Graphs all points and interpolates unknown points, useful for visualizing Gaussian Interpolation and affects of kernals
    :return:
    """
    all_points = create_points_with_random_pollution_1d(100, 100, 10)
    picked_points = pick_uniform_random_points(all_points, 20)
    interpolated_points = interpolate_unknown_points(picked_points, all_points)

    picked_x = []
    picked_pollution = []
    for label, point in picked_points.items():
        picked_x.append(label)
        picked_pollution.append(point.get_pollution_value())

    interp_x = []
    inter_pollution = []

    for label, point in interpolated_points.items():
        if not label in picked_x:
            interp_x.append(label)
            inter_pollution.append(point.get_pollution_value())

    plt.plot(picked_x, picked_pollution, "ro", interp_x, inter_pollution, "go")
    plt.xlabel("Point Label")
    plt.ylabel("Pollution Value")
    plt.show()


def run_experiment_with_various_length_scales_linear(bottom_bound, top_bound, side_length, mean,std, pick_number, number_of_maps, step):
    """
    Experiment to see rmse of cheating and not cheating regreassion on varous length scales (traverses linearly)  in 2D. Uses uniform point selection and RBF kernel
    :param step: step
    :param bottom_bound: bottom bound of length scale
    :param top_bound: top bound of length scale not inclusive
    :param side_length: number of points on one side of the square of points
    :param mean: Mean pollution value to be set
    :param std: Standard Deviation
    :param pick_number: the Beta (the number of points to select to be measured)
    :param number_of_maps: Number of trials for each length scale
    :return:
    """
    not_cheating_data = []
    cheating_data = []
    for length_scale in range(bottom_bound,top_bound,step): #runs through each length scale
        points = create_points_with_spatially_correlated_pollution_2d(side_length,mean,std,length_scale, number_of_maps) # Creates all points
        picked_points = pick_uniform_random_points_on_map_of_maps(points,pick_number) # Picks points to be measured
        interpolated_points  = interpolate_unknown_points_of_a_map_of_maps_of_points(picked_points, points, # Interpolates using noncheating method
                               RBF(np.random.randint(1e-05, 100 + 1)), fixed=False)

        not_cheating_data.append(average_rmse_of_maps(interpolated_points)) #adds average rms of all the trials for the noncheating method
        interpolated_points = interpolate_unknown_points_of_a_map_of_maps_of_points(picked_points, points, # Interpolates using cheating method
                                                                                    RBF(length_scale,
                                                                                        ), fixed=True)

        cheating_data.append(average_rmse_of_maps(interpolated_points)) #adds average rmse of all the trials for the cheating method
        print(length_scale)


    plot_numbers(range(bottom_bound, top_bound, step), not_cheating_data, range(bottom_bound, top_bound,step), cheating_data, #Plots the data Red is not cheating, Green Cheating
                 "Length Scale", "RMSE")

def run_experiment_with_various_length_scales_log(bottom_bound, top_bound, side_length, mean,std, pick_number, number_of_maps):
    """
    Experiment to see rmse of cheating and not cheating regreassion on varous length scales (traverses by powers of 10)  in 2D. Uses uniform point selection and RBF kernel
    :param bottom_bound: bottom bound of length scale
    :param top_bound: top bound of length scale not inclusive
    :param side_length: number of points on one side of the square of points
    :param mean: Mean pollution value to be set
    :param std: Standard Deviation
    :param pick_number: the Beta (the number of points to select to be measured)
    :param number_of_maps: Number of trials for each length scale
    :return:
    """
    not_cheating_data = []
    cheating_data = []
    length_scale = bottom_bound
    length_scale_list = []
    while length_scale <= top_bound: #runs through each length scale
        points = create_points_with_spatially_correlated_pollution_2d(side_length,mean,std, length_scale, number_of_maps) # Creates all points
        picked_points = pick_uniform_random_points_on_map_of_maps(points,pick_number) # Picks points to be measured
        interpolated_points  = interpolate_unknown_points_of_a_map_of_maps_of_points(picked_points, points, # Interpolates using noncheating method
                               RBF(np.random.randint(1e-05, 100 + 1)), fixed=False)

        not_cheating_data.append(average_rmse_of_maps(interpolated_points)) #adds average rms of all the trials for the noncheating method
        interpolated_points = interpolate_unknown_points_of_a_map_of_maps_of_points(picked_points, points, # Interpolates using cheating method
                                                                                    RBF(length_scale,
                                                                                        ), fixed=True)

        cheating_data.append(average_rmse_of_maps(interpolated_points)) #adds average rmse of all the trials for the cheating method
        length_scale_list.append(length_scale)
        length_scale = length_scale * 10



    plot_numbers(length_scale_list, not_cheating_data, length_scale_list, cheating_data,  #Plots the data Red is not cheating, Green Cheating
                 "Length Scale", "RMSE", x_log_scale= True)




def see_what_its_doing_2d(length_scale, fixed):
    """
    3D graphs the pollution value of the measured and interpolated pollution values
    :param length_scale:
    :param fixed:
    :return:
    """



    a = create_points_with_spatially_correlated_pollution_2d(10, 100,10, length_scale, 1)
    b = pick_uniform_random_points_on_map_of_maps(a, 20)
    if fixed:
        c = interpolate_unknown_points_of_a_map_of_maps_of_points(b, a, RBF(length_scale), fixed= True)
    else:
        c = interpolate_unknown_points_of_a_map_of_maps_of_points(b, a, RBF(np.random.randint(1,10000)), fixed=False)

    x1 = []
    y1= []
    z1 =[]
    for point in b[0].values():
        x1.append(point.get_x_cord())
        y1.append(point.get_y_cord())
        z1.append(point.get_pollution_value())

    x2=[]
    y2=[]
    z2=[]

    for label, point in c[0][0].items():
        if not label in b[0].keys():
            x2.append(point.get_x_cord())
            y2.append(point.get_y_cord())
            z2.append(point.get_pollution_value())

    plot_numbers_3d_and_save(x1,y1,z1,x2,y2,z2,"Rotating Graph.gif")


    # mywriter = animation.FFMpegWriter(fps=60)
    # rot_animation.save("rotation.mp4",dpi = 80, writer= mywriter)


def see_what_its_doing_2d_comparison(length_scale):
    """
    ALlows visual comparison between interpolation with a cheating and not cheating interpolation
    :param length_scale:
    :return:
    """

    a = create_points_with_spatially_correlated_pollution_2d(10, 100,10, length_scale, 1)
    b = pick_uniform_random_points_on_map_of_maps(a, 20)
    c1 = interpolate_unknown_points_of_a_map_of_maps_of_points(b, a, RBF(length_scale), fixed= True)
    c2 = interpolate_unknown_points_of_a_map_of_maps_of_points(b, a, RBF(np.random.randint(1,10000)), fixed=False)

    x1= []
    y1= []
    z1 =[]
    for point in b[0].values():
        x1.append(point.get_x_cord())
        y1.append(point.get_y_cord())
        z1.append(point.get_pollution_value())

    x2_fixed=[]
    y2_fixed=[]
    z2_fixed=[]

    for label, point in c1[0][0].items():
        if not label in b[0].keys():
            x2_fixed.append(point.get_x_cord())
            y2_fixed.append(point.get_y_cord())
            z2_fixed.append(point.get_pollution_value())


    x2_not_fixed = []
    y2_not_fixed = []
    z2_not_fixed = []

    for label, point in c2[0][0].items():
        if not label in b[0].keys():
            x2_not_fixed.append(point.get_x_cord())
            y2_not_fixed.append(point.get_y_cord())
            z2_not_fixed.append(point.get_pollution_value())

    plot_numbers_3d_and_save(x1,y1,z1,x2_fixed,y2_fixed,z2_fixed,"Fixed Rotating Graph.gif")
    plot_numbers_3d_and_save(x1,y1,z1,x2_not_fixed,y2_not_fixed,z2_not_fixed, "Not Fixed Rotating Graph.gif")

def plot_numbers_3d_and_save(x1,y1,z1,x2,y2,z2,filename = "Rotating Graph.gif"):
    """
    Scatterplot in 3d and saves rotating gif to file
    :param x1:
    :param y1:
    :param z1:
    :param x2:
    :param y2:
    :param z2:
    :param filename: File name ending in .gif
    :return:
    """
    fig = plt.figure()
    sub = fig.add_subplot(1, 1, 1, projection="3d")
    sub.scatter(x1, y1, z1, marker="o", edgecolor="r", facecolor="r")
    sub.scatter(x2, y2, z2, marker="^", edgecolor="g", facecolor="g")
    def rotate(angle):
        sub.view_init(azim=angle)
    rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 362, 2), interval=100)
    # mywriter = animation.FFMpegWriter(fps=60)
    # rot_animation.save("rotation.mp4",dpi = 80, writer= mywriter)
    print("Starting Save")
    rot_animation.save(filename, dpi=80, writer='imagemagick')
    print("Finished Save")
    
    
    
    
    
# run_experiment_with_various_length_scales_log(.000001, 1000000, 10, 100, 20, 2)
# run_experiment_with_various_length_scales_linear(10,100,10,100,10,20,100,5)


# see_what_its_doing_2d(100, False)
see_what_its_doing_2d_comparison(100)

# length_scale = 100
# a = create_points_with_spatially_correlated_pollution_2d(10,100,length_scale,1)
# b = pick_uniform_random_points_on_map_of_maps(a,20)
# c = interpolate_unknown_points_of_a_map_of_maps_of_points(b,a, RBF(length_scale), fixed=True)
# d = interpolate_unknown_points_of_a_map_of_maps_of_points(b,a, RBF(1) )
# print()


#
#
# random_points1 = create_points_with_random_pollution(10, 100, 10)
# p = pick_uniform_random_points(random_points1,5)
#
# a = interpolate_unknown_points(p,random_points1)
# test_pollution_maps = create_points_with_spatially_correlated_pollution_2d(10, 100, 1, 2)
# for i in range(len(test_pollution_maps)):
#    for j in range(len(test_pollution_maps[0])):
#       print(test_pollution_maps[i][j], end=' ')
#    print()
