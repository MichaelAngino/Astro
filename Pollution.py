import numpy


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
        return [self.x,self.y]




def create_list_of_random_pollution_points(length, mean, std):
    """
    Creates a list of points with random Pollution Values using a gaussian distribution in 1D evenly starting at x = 05 every 10
    :param length: Wanted length of list
    :param mean: Mean of gaussian distribution of pollution values
    :param std: Standard Deviation of gaussian distribution of pollution values
    :return: list of points
    """
    new_list = []
    x = 5
    for i in range(0,100):
        new_list.append(Point(i,numpy.random.normal(mean,std),x))
        x = x+10
    return new_list


