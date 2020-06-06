import unittest

import Pollution


class MyTestCase(unittest.TestCase):
    def test_point_list(self):
        """
        Testing Create_list_of_points method
        :return:
        """
        test_list = Pollution.create_list_of_random_pollution_points(100,100,0)
        self.assertEqual(100, len(test_list))
        for i in range(0,len(test_list)):
            self.assertEqual(100,test_list[i].get_pollution_value())

        self.assertEqual(5, test_list[0].get_x_cord())
        self.assertEqual(995, test_list[99].get_x_cord())

    def test_point(self):
        """
        Testing Point Class
        :return:
        """
        test_point = Pollution.Point(1,100,0)
        self.assertEqual(1,test_point.get_label())
        self.assertEqual(100,test_point.get_pollution_value())
        test_point.set_pollution_value(20)

        self.assertEqual(20, test_point.get_pollution_value())

    d





