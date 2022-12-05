#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest

import numpy as np

from RNN import RNN, argmax, nll_loss, softmax


class RNN_tests(unittest.TestCase):

    def setUp(self) -> None:
        """ Initialized the random number generator to be used across tests.
        """
        self.rng = np.random.default_rng()

    def tearDown(self) -> None:
        """ Deletes the random number generator.
        """
        del self.rng

    def test_softmax_1d(self):
        """ Tests output dimensionality of the softmax function for a 1-D input.
        """ 
        x = self.rng.standard_normal(size=5)
        self.assertEqual(softmax(x).ndim, 1)

    def test_softmax_2d(self):
        """ Tests output dimensionality of the softmax function for a 2-D input.
        """ 
        x = self.rng.standard_normal(size=(5, 5))
        self.assertEqual(softmax(x).ndim, 2)

    def test_argmax_1d(self):
        """ Tests the argmax function for a 1-D input and resulting output.
        """ 
        x = np.array([0, 0, 1, 0, 0])
        ground_truth = 2
        self.assertEqual(argmax(x), ground_truth)

    def test_argmax_2d(self):
        """ Tests the argmax function for a 2-D input and resulting output.
        """ 
        x = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0]
        ])
        ground_truth = [1, 0, 3, 0]
        self.assertEqual(argmax(x).tolist(), ground_truth)

    def test_nll_loss(self):
        """ Tests the negative log likelihood function's first output type 
        and second output dimensionality.
        """
        x = np.abs(self.rng.standard_normal(size=(5, 5)))
        t = self.rng.integers(low=0, high=4, size=5)
        total, partial = nll_loss(x, t)
        self.assertTrue(isinstance(total, float))
        self.assertEqual(partial.ndim, 1)

    def test_rnn_train(self):
        """ Tests the RNN's __call__ method output in training mode, asserts 
        the types of the first output, the dimensionality and length of the 
        second and the dimensionality and shape of the third.
        """
        x = self.rng.standard_normal(size=(5, 10))
        rnn = RNN(10, 5)
        pred, h, prob = rnn(x)
        self.assertTrue(isinstance(pred, list))
        self.assertTrue(isinstance(pred[0], np.int64))
        self.assertEqual(len(pred), 5)

        self.assertEqual(h.ndim, 1)
        self.assertEqual(len(h), 5)

        self.assertEqual(prob.ndim, 2)
        self.assertTrue(np.all(prob.shape == np.array([5, 10])))

    def test_rnn_eval(self):
        """ Similar to the test_rnn_train test, but the RNN is in evaluation mode.
        """
        x = self.rng.standard_normal(size=10)
        rnn = RNN(10, 5)
        rnn.eval()
        pred, h, prob = rnn(x)
        self.assertTrue(isinstance(pred, list))
        self.assertEqual(len(pred), 1)
        self.assertTrue(isinstance(pred[0], np.int64))

        self.assertEqual(h.ndim, 1)

        self.assertEqual(prob.ndim, 2)
        self.assertEqual(prob.shape[0], 1)


if __name__ == '__main__':
    unittest.main()
