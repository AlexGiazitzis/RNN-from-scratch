#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
from typing import Optional

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """ Applies the softmax function to the specified matrix in direction 
    of the highest dimension.

    :param x: A numerical matrix.
    :type x: np.ndarray
    :return: The normalized values of the inputted matrix based on the softmax formula.
    :rtype: np.ndarray
    """
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=(x.ndim - 1), keepdims=True)


def argmax(x: np.ndarray) -> np.ndarray:
    """ Applies the argmax function to the specified matrix in the direction 
    of the highest dimension.

    :param x: A numerical matrix.
    :type x: np.ndarray
    :return: The indices of the maximum argument in the highest dimension.
    :rtype: np.ndarray
    """
    return np.argmax(x, axis=(x.ndim - 1))


def nll_loss(output_probs: np.ndarray, target: np.ndarray) -> (float, np.ndarray):
    """ Calculates the Negative Loss Likelihood given the probabilities of 
    the classes and the actual classes of each observation.

    :param output_probs: A matrix containing probabilities for each observation.
    :type output_probs: np.ndarray
    :param target: A matrix containing the actual class of each observation.
    :type target: np.ndarray
    :return: The total loss from each prediction and the loss for each observation.
    :rtype: tuple[float, np.ndarray]
    """
    neg_log_probs = -np.log(output_probs)
    nll_loss_target = neg_log_probs[range(target.shape[0]), target]
    return np.sum(nll_loss_target), nll_loss_target


def accuracy(output_idx: np.ndarray, target: np.ndarray) -> float:
    """ Calculates the accuracy of the predictions given the actual values.

    :param output_idx: A matrix containing prediction classes.
    :type output_idx: np.ndarray
    :param target: A matrix containing the actual classes for each observation.
    :type target: np.ndarray
    :return: The percentage for the accurate predictions.
    :rtype: float
    """
    count = sum([1 if np.all(output_idx[i] == target[i]) else 0 for i in range(target.shape[0])])
    return count * 100 / target.shape[0]


class RNN:
    """ A class representing the Recurrent Neural Networks, a type of Neural Network 
    that carries over information from previous predictions/state allowing the usage 
    of relations between values in an observation be taken into account.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_state_size: int,
                 output_dim: Optional[int] = None) -> None:
        """ Constructs an instance of RNN.

        :param input_dim: The dimension of the, to be given, input data.
        :type input_dim: int
        :param hidden_state_size: The dimension of the hidden state.
        :type hidden_state_size: int
        :param output_dim: The dimension of the predictions, defaults to input_dim if not specified.
        :type output_dim: Optional[int], optional
        """
        if not output_dim:
            output_dim = input_dim
        rng = np.random.default_rng()

        # Initialize hidden state to zeros
        self.h = np.zeros(hidden_state_size)

        # Initialize weight matrices and bias vectors
        self.W_hh = rng.standard_normal(size=(hidden_state_size, hidden_state_size))
        self.W_hx = rng.standard_normal(size=(hidden_state_size, input_dim))
        self.W_oh = rng.standard_normal(size=(output_dim, hidden_state_size))

        self.b_h = rng.standard_normal(size=hidden_state_size)
        self.b_o = rng.standard_normal(size=output_dim)

        self.h_t = []

        self.training = True

    def __call__(self, x: np.ndarray, length: int = 1) -> (list[np.int64], np.ndarray, np.ndarray):
        """ Calculates the prediction of the RNN model using the standardized formulas.

        :param x: Input predictors
        :type x: np.ndarray
        :param length: The amount of steps to predict if in evaluation mode, defaults to 1.
        :type length: int, optional
        :return: The predicted classes, the hidden state and the predicted class probabilities.
        :rtype: tuple[list[np.int64], np.ndarray, np.ndarray]
        """
        output_probs = []
        output_preds = []
        steps = x.shape[0] if self.training else length if length > 0 else 1
        x_t = x[0]
        for step in range(steps):
            self.h = np.tanh(self.W_hh @ self.h + self.W_hx @ x_t + self.b_h)
            output_probs.append(softmax(self.W_oh @ self.h + self.b_o))
            output_preds.append(argmax(output_probs[-1]))
            if self.training:
                x_t = x[step + 1 if step < steps - 1 else 0]
                self.h_t.append(self.h)
            else:
                x_t = np.zeros_like(x_t)
                x_t[output_preds[-1]] = 1
        output_probs = np.array(output_probs)

        return output_preds, self.h, output_probs

    def bppt(self, output_probs: np.ndarray, targets: list[int], x: np.ndarray, lr: float, grad_thresh: Optional[float] = None) -> None:
        """ Performs the Back Propagation Through Time algorithm on the RNN with a Gradient Descent 
        optimizer and a Negative Log Likelihood loss function.

        :param output_probs: The class probabilities predicted by the model for each observation.
        :type output_probs: np.ndarray
        :param targets: The actual classes for each observation.
        :type targets: list[int]
        :param x: The input data given to the RNN.
        :type x: np.ndarray
        :param lr: A learning rate to apply to the optimization process.
        :type lr: float
        :param grad_thresh: A threshold for the gradients, to mitigate the exploding/vanishing gradients.
        :type grad_thresh: float
        """
        steps = len(targets)

        do = output_probs
        do[range(output_probs.shape[0]), targets] -= 1

        dh = [self.W_oh.T @ do[-1]]
        diag_h_t_sqrd = [np.diag(1 - np.square(h)) for h in self.h_t]

        for t in range(steps - 2, -1, -1):
            dh.insert(0, self.W_hh.T @ diag_h_t_sqrd[t + 1] @ dh[0] + self.W_oh.T @ do[t])

        dW_hh = np.array([diag_h_t_sqrd[t] @ dh[t][:, np.newaxis] @ self.h_t[t - 1][:, np.newaxis].T
                          for t in range(1, steps)]).sum(axis=0)
        dW_oh = np.zeros_like(self.W_oh)
        dW_hx = np.zeros_like(self.W_hx)
        db_h = np.zeros_like(self.b_h)
        db_o = do.sum(axis=0)

        for t in range(steps):
            dW_oh += do[t][:, np.newaxis] @ self.h_t[t][:, np.newaxis].T  # convert 1D to 2D so a OxH matrix is produced
            dW_hx += diag_h_t_sqrd[t] @ dh[t][:, np.newaxis] @ x[[t]]
            db_h += diag_h_t_sqrd[t] @ dh[t]

        l2_dW_hh = np.linalg.norm(dW_hh, ord=2)
        l2_dW_oh = np.linalg.norm(dW_oh, ord=2)
        l2_dW_hx = np.linalg.norm(dW_hx, ord=2)
        l2_db_h = np.linalg.norm(db_h, ord=2)
        l2_db_o = np.linalg.norm(db_o, ord=2)

        if grad_thresh:
            if l2_dW_hh >= grad_thresh:
                dW_hh = grad_thresh * dW_hh / l2_dW_hh

            if l2_dW_oh >= grad_thresh:
                dW_oh = grad_thresh * dW_oh / l2_dW_oh

            if l2_dW_hx >= grad_thresh:
                dW_hx = grad_thresh * dW_hx / l2_dW_hx

            if l2_db_h >= grad_thresh:
                db_h = grad_thresh * db_h / l2_db_h

            if l2_db_o >= grad_thresh:
                db_o = grad_thresh * db_o / l2_dW_hh

        self.W_hh -= lr * dW_hh
        self.W_oh -= lr * dW_oh
        self.W_hx -= lr * dW_hx
        self.b_h -= lr * db_h
        self.b_o -= lr * db_o

        self.h_t.clear()
        self.reset()

    def train(self) -> None:
        """ Sets the model in training mode.
        """
        self.training = True

    def eval(self) -> None:
        """ Sets the model in evaluation mode.
        """
        self.training = False

    def reset(self) -> None:
        """ Resets the hidden state of the RNN.
        """
        self.h = np.zeros_like(self.h)
    
    def save(self, file_path: str) -> None:
        """ Exports the weight and bias matrices of the RNN so that it can be moved/loaded elsewhere.

        :param file_path: An absolute or relative filepath where the model's state will be saved. Must be a path of a file with the .npz extension.
        :type file_path: str
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.savez_compressed(
            file_path,
            W_hh=self.W_hh,
            W_hx=self.W_hx,
            W_oh=self.W_oh,
            b_h=self.b_h,
            b_o=self.b_o
        )
    
    @staticmethod
    def load(file_path: str) -> "RNN":
        """ Loads the model state from a given .npz file.

        :param file_path: An absolute or relative filepath where the model's state was saved. Must be a path of a file with the .npz extension.
        :type file_path: str
        :return: An RNN model with the loaded weight and bias matrices.
        :rtype: RNN
        """
        mats = np.load(file_path)
        rnn = RNN(mats["W_hx"].shape[1], mats["W_hx"].shape[0], output_dim=mats["W_oh"].shape[0])
        rnn.W_hh = mats["W_hh"]
        rnn.W_hx = mats["W_hx"]
        rnn.W_oh = mats["W_oh"]
        rnn.b_h = mats["b_h"]
        rnn.b_o = mats["b_o"]
        return rnn
