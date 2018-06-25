import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import numpy as np
from scipy.stats import multivariate_normal

ecg_c = 10.0
ecg_epsilon = 0.1


def ecg_loss_np(y_true, y_pred):
    assert (y_true.shape == y_pred.shape)
    losses = []
    for row in range(y_true.shape[0]):
        y_true_row = y_true[row, :]
        y_pred_row = y_pred[row, :]
        n = multivariate_normal.pdf(
            y_true_row, mean=y_pred_row, cov=np.identity(len(y_pred_row)))
        nc = multivariate_normal.pdf(
            y_true_row, mean=y_pred_row, cov=np.identity(len(y_pred_row)) * ecg_c)
        losses.append(np.log((1.0 - ecg_epsilon) *
                             n + ecg_epsilon * nc) * -1.0)
    return np.mean(losses)


def ecg_loss(y_true, y_pred):
    num_dims = y_pred.get_shape().as_list()[1]
    n = tfd.MultivariateNormalDiag(
        loc=y_pred, scale_diag=tf.ones(num_dims)).prob(y_true)
    nc = tfd.MultivariateNormalDiag(
        loc=y_pred, scale_diag=tf.ones(num_dims) * ecg_c).prob(y_true)
    return tf.reduce_mean(tf.log((1.0 - ecg_epsilon) * n + ecg_epsilon * nc) * -1.0)


if __name__ == "__main__":
    with tf.Session() as sess:
        y_true = np.random.rand(5, 10)
        y_pred = np.random.rand(5, 10)
        loss_np = ecg_loss_np(y_true, y_pred)
        print(loss_np)

        y_true_tf = tf.placeholder(tf.float32, y_true.shape)
        y_pred_tf = tf.placeholder(tf.float32, y_pred.shape)
        loss_tf = ecg_loss(y_true_tf, y_pred_tf)

        loss_tf_result = sess.run(loss_tf,
                                  feed_dict={y_true_tf: y_true,
                                             y_pred_tf: y_pred})
        print(loss_tf_result)

        assert np.isclose(loss_np, loss_tf_result)
