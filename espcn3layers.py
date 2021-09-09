"""ESPCN model for GDFLAB dataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from common.layers import conv2d_weight_norm
import models


def update_argparser(parser):
  models.update_argparser(parser)
  args, _ = parser.parse_known_args()
  if args.dataset == 'div2k_std':
    parser.add_argument(
        '--num-blocks',
        help='Number of residual blocks in networks',
        default=16,
        type=int)
    parser.add_argument(
        '--num-residual-units',
        help='Number of residual units in networks',
        default=32,
        type=int)
    parser.set_defaults(
#         train_steps=2000000,
        train_steps=500000,
        learning_rate=((250000, 350000, 400000, 450000), (1e-3, 5e-4, 2e-4,
                                                              1e-4, 5e-5)),
#         save_checkpoints_steps=5,
#         save_summary_steps=2,
    )
  else:
    raise NotImplementedError('Needs to tune hyper parameters for new dataset.')
    
    ########################################################
def model_fn(features, labels, mode, params, config):
  predictions = None
  loss = None
  train_op = None
  eval_metric_ops = None
  export_outputs = None

  lr = features['source']
  def _espcn(x):
    #weight_parameters = []
    #bias_parameters = []
    #     inputs = tf.layers.Input(train_data)
    c1 = tf.layers.conv2d(x, filters=64, kernel_size=[5, 5], strides=(1,1),
                          kernel_initializer=tf.glorot_uniform_initializer(),
                          data_format='channels_last',dilation_rate=(1, 1), 
                          bias_initializer=tf.zeros_initializer(),
                          activation=tf.nn.relu, padding='same', use_bias=True)
    c2 = tf.layers.conv2d(c1, filters=32, kernel_size=[3, 3], strides=1,
                          data_format='channels_last',
                          kernel_initializer=tf.glorot_uniform_initializer(),
                          activation=tf.nn.relu, padding='same', use_bias=True)
    #c2 = tf.layers.conv2d(c2, filters=32, kernel_size=[3, 3], strides=1,
    #                      data_format='channels_last',
    #                      kernel_initializer=tf.glorot_uniform_initializer(),
    #                      activation=tf.nn.relu, padding='same', use_bias=True)
    #c2 = tf.layers.conv2d(c2, filters=32, kernel_size=[3, 3], strides=1,
    #                      data_format='channels_last',
    #                      kernel_initializer=tf.glorot_uniform_initializer(),
    #                      activation=tf.nn.relu, padding='same', use_bias=True)
    
    c3 = tf.layers.conv2d(c2, filters=params.scale*2*3, kernel_size=[3, 3], strides=1,
                          data_format='channels_last',
                          kernel_initializer=tf.glorot_uniform_initializer(),
                          activation=tf.nn.relu, padding='SAME', use_bias=True)
    c4 = tf.depth_to_space(c3, params.scale)
    #weight_parameters += [c1, c2, c3, c4]
    return c4 #, weight_parameters, bias_parameters
###############################################################
  sr = _espcn(lr)

  predictions = tf.clip_by_value(sr, 0.0, 1.0)

  if mode == tf.estimator.ModeKeys.PREDICT:
    export_outputs = {
        tf.saved_model.signature_constants.PREDICT_METHOD_NAME:
            tf.estimator.export.PredictOutput(predictions)
    }
  else:
    hr = labels['target']
    loss = tf.losses.absolute_difference(labels=hr, predictions=sr)
    if mode == tf.estimator.ModeKeys.EVAL:

      def _ignore_boundary(images):
        boundary_size = params.scale + 6
        images = images[:, boundary_size:-boundary_size, boundary_size:
                        -boundary_size, :]
        return images

      def _float32_to_uint8(images):
        images = images * 255.0
        images = tf.round(images)
        images = tf.saturate_cast(images, tf.uint8)
        return images

      psnr = tf.image.psnr(
          _float32_to_uint8(_ignore_boundary(hr)),
          _float32_to_uint8(_ignore_boundary(predictions)),
          max_val=255,
      )
      ssim = tf.image.ssim(
          _float32_to_uint8(_ignore_boundary(hr)),
          _float32_to_uint8(_ignore_boundary(predictions)),
          max_val=255,
      )

      eval_metric_ops = {
          'PSNR': tf.metrics.mean(psnr),
          'SSIM': tf.metrics.mean(ssim),
      }
    if mode == tf.estimator.ModeKeys.TRAIN:
      global_step = tf.train.get_or_create_global_step()
      learning_rate = tf.train.piecewise_constant(
          global_step, params.learning_rate[0], params.learning_rate[1])
      opt = tf.train.AdamOptimizer(learning_rate)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        train_op = opt.minimize(loss, global_step=global_step)
      stats = tf.profiler.profile()
      print("Total parameters:", stats.total_parameters)

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops,
      export_outputs=export_outputs,
  )
