import unittest
from collections import defaultdict
import Pollution


class MyTestCase(unittest.TestCase):
    def test_point_list(self):
        """
        Testing Create_list_of_points method
        :return:
        """
        test_list = Pollution.create_points_with_random_pollution(100,100,0)
        self.assertEqual(100, len(test_list))
        for i in range(0,len(test_list)):
            self.assertEqual(100,test_list[i].get_actual_pollution_value())

        self.assertEqual(5, test_list[0].get_x_cord())
        self.assertEqual(995, test_list[99].get_x_cord())

    def test_point(self):
        """
        Testing Point Class
        :return:
        """
        test_point = Pollution.Point(1,100,0)
        self.assertEqual(1,test_point.get_label())
        self.assertEqual(100,test_point.get_actual_pollution_value())
        test_point.set_pollution_value(20)

        self.assertEqual(20, test_point.get_pollution_value())

    def test_uniform_point_selection(self):
        """
        Tests uniform point selction
        :return:
        """

        test_points = Pollution.create_points_with_random_pollution(10,10,1)
        counter = defaultdict(int)
        num_of_runs = 10000

        for i in range(0,num_of_runs):
            a = Pollution.pick_uniform_random_points(test_points, 5)
            for key in a.keys():
                counter[key] = counter[key]+1

        for key in counter.keys():
            counter[key] = counter[key]/ num_of_runs

        for i in range(0,9):
            self.assertAlmostEqual(.5, counter[i],None,None,.02)

    def test_interpolation_mutation(self):
        """
        Tests to make sure that the interpolation doesnt mutate existing measured points' pollution values
        :return:
        """

        points = Pollution.create_points_with_random_pollution(100,100,10)

        picked_points = Pollution.pick_uniform_random_points(points,50)

        interpolated_points = Pollution.interpolate_unknown_points(picked_points,points)

        for label, point in picked_points.items():
            self.assertEqual(picked_points[label].get_pollution_value(),interpolated_points[label].get_pollution_value())

        for label,point in points.items():
            self.assertEqual(points[label].get_position(), interpolated_points[label].get_position())
            self.assertEqual(points[label].get_actual_pollution_value(), interpolated_points[label].get_actual_pollution_value(),
                             "test failed 1 APV:" + str(points[label].get_actual_pollution_value()) + "| 2 APV: " + str(interpolated_points[label].get_actual_pollution_value()))




