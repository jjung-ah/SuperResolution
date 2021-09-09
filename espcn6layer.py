"""WDSR model for DIV2K dataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import models


#dataset test
def update_argparser(parser):
  models.update_argparser(parser)
  args, _ = parser.parse_known_args()
  if args.dataset == 'div2k':
#    parser.add_argument(
#        '--num-blocks',
#        help='Nu3mber of residual blocks in networks',
#        default=8,
#        type=int)
#    parser.add_argument(
#        '--num-residual-units',
#        help='Number of residual units in networks',
#        default=16,
#        type=int)
    parser.set_defaults(
        train_steps=1000000,
        learning_rate=((500000, 750000, 875000), (1e-3, 1e-4, 1e-5,
                                                              1e-7)),
        save_checkpoints_steps=100,
        save_summary_steps=100,
    )
  else:
    raise NotImplementedError('Needs to tune hyper parameters for new dataset.')




def model_fn(features, labels, mode, params, config):
  predictions = None
  loss = None
  train_op = None
  eval_metric_ops = None
  export_outputs = None

  lr = features['source']


  def get_model6(inputs, scale, batch_size=128):
    weight_parameters = []
    bias_parameters = []
    #     inputs = tf.layers.Input(train_data)
    c1 = tf.layers.conv2d(inputs, filters=64, kernel_size=[5, 5], strides=1,
                          kernel_initializer=tf.glorot_uniform_initializer(),
                          activation=tf.nn.relu, padding='VALID', use_bias=True)
    c2 = tf.layers.conv2d(c1, filters=32, kernel_size=[3, 3], strides=1,
                          kernel_initializer=tf.glorot_uniform_initializer(),
                          activation=tf.nn.relu, padding='SAME', use_bias=True)

    c2 = tf.layers.conv2d(c2, filters=32, kernel_size=[3, 3], strides=1,
                          kernel_initializer=tf.glorot_uniform_initializer(),
                          activation=tf.nn.relu, padding='SAME', use_bias=True)
    c2 = tf.layers.conv2d(c2, filters=32, kernel_size=[3, 3], strides=1,
                          kernel_initializer=tf.glorot_uniform_initializer(),
                          activation=tf.nn.relu, padding='SAME', use_bias=True)

    c3 = tf.layers.conv2d(c2, filters=scale ** 2, kernel_size=[3, 3], strides=1,
                          kernel_initializer=tf.glorot_uniform_initializer(),
                          activation=tf.nn.relu, padding='SAME', use_bias=True)
    c4 = tf.depth_to_space(c3, scale)
    weight_parameters += [c1, c2, c3, c4]
    return c4, weight_parameters, bias_parameters

  def get_shape(scale):
    inputs = [14, 11, 10]
    patch_size = inputs[scale - 2]
    conv_size = scale * 2
    label_size = patch_size * scale - (2 * conv_size)
    block_step = int(patch_size / 2)

    return patch_size, label_size, conv_size, block_step


  sr = get_model6(inputs=lr, scale=2, batch_size=128)
  print("It is SR ", sr)

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
        #boundary_size = params.scale + 12
        #boundary_size = params.scale + 2
        images = images[:, boundary_size:-boundary_size, boundary_size:
                       -boundary_size, :]
        return images

      def _float32_to_uint8(images):
        images = images * 255.0
        #images = images * 65535.0
        images = tf.round(images)
        images = tf.saturate_cast(images, tf.uint8)
        #images = tf.saturate_cast(images, tf.uint16)
        return images

      psnr = tf.image.psnr(
          _float32_to_uint8(_ignore_boundary(hr)),
          _float32_to_uint8(_ignore_boundary(predictions)),
          max_val=255,
          #max_val=65535,
      )

      ssim = tf.image.ssim(
          _float32_to_uint8(_ignore_boundary(hr)),
          _float32_to_uint8(_ignore_boundary(predictions)),
          max_val=255,
          #max_val=65535,
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
