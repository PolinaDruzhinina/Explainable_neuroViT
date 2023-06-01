import io
import numpy as np
import os

from PIL import Image
from scipy import interpolate
from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple


def create_blurred_image(full_img: np.ndarray, pixel_mask: np.ndarray,
    method: str = 'linear') -> np.ndarray:
  """ Creates a blurred (interpolated) image.

  Args:
    full_img: an original input image that should be used as the source for
      interpolation. The image should be represented by a numpy array with
      dimensions [H, W, C] or [H, W].
    pixel_mask: a binary mask, where 'True' values represent pixels that should
      be retrieved from the original image as the source for the interpolation
      and 'False' values represent pixels, which values should be found. The
      method always sets the corner pixels of the mask to True. The mask
      dimensions should be [H, W].
    method: the method to use for the interpolation. The 'linear' method is
      recommended. The alternative value is 'nearest'.

    Returns:
      A numpy array that encodes the blurred image with exactly the same
      dimensions and type as `full_img`.
  """
  data_type = full_img.dtype
  full_img = full_img.copy()
  full_img = torch.squeeze(full_img)
  #has_color_channel = full_img.ndim > 2
  #if not has_color_channel:
  #  full_img = np.expand_dims(full_img, axis=2)
  #channels = full_img.shape[2]

  # Always include corners.
  pixel_mask = pixel_mask.copy()
  height = pixel_mask.shape[0]
  width = pixel_mask.shape[1]
  depth = pixel_mask.shape[2]
  pixel_mask[[0, height - 1, 0, 0, height - 1, height - 1, 0, height - 1,],
             [0, 0, width - 1, 0, width - 1, 0, width - 1, width - 1,],
             [0, 0, 0, depth - 1, 0, depth - 1, depth - 1, depth - 1,]] = True

  mean_color = np.mean(full_img)

  # If the mask consists of all pixels set to True then return the original
  # image.
  if np.all(pixel_mask):
    return full_img

  blurred_img = full_img * pixel_mask.astype(np.float32)

  # Interpolate the unmasked values of the image pixels.
  data_points = np.argwhere(pixel_mask > 0)
  data_values = full_img[tuple(data_points.T)]
  unknown_points = np.argwhere(pixel_mask == 0)
  interpolated_values = interpolate.griddata(np.array(data_points),
                                             np.array(data_values),
                                             np.array(unknown_points),
                                             method=method,
                                             fill_value=mean_color)
  blurred_img[tuple(unknown_points.T)] = interpolated_values

  if issubclass(data_type.type, np.integer):
    blurred_img = np.round(blurred_img)

  return blurred_img.astype(data_type).unsqueeze(0)


def generate_random_mask(image_height: int, image_width: int, image_depth: int, fraction=0.01) -> np.ndarray:
  mask = np.zeros(shape=[image_height, image_width, image_depth], dtype=bool)
  size = mask.size
  indices = np.random.choice(size, replace=False, size=int(size * fraction))
  mask[np.unravel_index(indices, mask.shape)] = True
  return mask

'''
def estimate_image_entropy(image: np.ndarray) -> float:
  """Estimates the amount of information in a given image.

    Args:
      image: an image, which entropy should be estimated. The dimensions of the
        array should be [H, W, C] or [H, W] of type uint8.
    Returns:
      The estimated amount of information in the image.
  """
  buffer = io.BytesIO()
  pil_image = Image.fromarray(image)
  pil_image.save(buffer, format='webp', lossless=True, quality=100)
  buffer.seek(0, os.SEEK_END)
  length = buffer.tell()
  buffer.close()
  return length
'''

def estimate_image_entropy(image: np.ndarray) -> int:
  nifti_image = nib.Nifti1Image(np.squeeze(image), np.eye(4))
  nib.save(nifti_image, 'tmp_img.nii.gz')
  return os.path.getsize('tmp_img.nii.gz')


def estimate_3d_image_entropy(image: np.ndarray) -> float:
  # Convert the 3D image to a 2D image by taking the average of each voxel in each slice.
  2d_image = np.mean(image, axis=-1)

  # Calculate the entropy of the 2D image.
  entropy = estimate_image_entropy(2d_image)

  # Multiply the entropy of the 2D image by the number of slices in the 3D image.
  return entropy * image.shape[-1]


class PicMetricResult(NamedTuple):
  """Holds results of compute_pic_metric(...) method."""
  # x-axis coordinates of PIC curve data points.
  curve_x: Sequence[float]
  # y-axis coordinates of PIC curve data points.
  curve_y: Sequence[float]
  # A sequence of intermediate blurred images used for PIC computation with
  # the fully blurred image in front and the original image at the end.
  blurred_images: Sequence[np.ndarray]
  # Model predictions for images in the `blurred_images` sequence.
  predictions: Sequence[float]
  # Saliency thresholds that were used to generate corresponding
  # `blurred_images`.
  thresholds: Sequence[float]
  # Area under the curve.
  auc: float


def compute_pic_metric(
    img: np.ndarray,
    saliency_map: np.ndarray,
    random_mask: np.ndarray,
    model,
    saliency_thresholds: Sequence[float],
    min_pred_value: float = 0.6,
    keep_monotonous: bool = True,
    num_data_points: int = 1000
) -> PicMetricResult:
  if img.dtype.type != np.uint8:
    raise ValueError('The `img` array that holds the input image should be of'
                     ' type uint8. The actual type is {}.'.format(img.dtype))
  blurred_images = []
  predictions = []

  # This list will contain mapping of image entropy for a given saliency
  # threshold to model prediction.
  entropy_pred_tuples = []

  # Estimate entropy of the original image.
  original_img_entropy = estimate_image_entropy(img)

  # Estimate entropy of the completely blurred image.
  fully_blurred_img = create_blurred_image(full_img=img, pixel_mask=random_mask)
  fully_blurred_img_entropy = estimate_image_entropy(fully_blurred_img)

  # Compute model prediction for the original image.
  original_img_pred = model(torch.Tensor(img).unqueeze(0).unsqueeze(0)).item()

  if original_img_pred < min_pred_value:
    message = ('The model prediction score on the original image is lower than'
               ' `min_pred_value`. Skip this image or decrease the'
               ' value of `min_pred_value` argument. min_pred_value'
               ' = {}, the image prediction'
               ' = {}.'.format(min_pred_value, original_img_pred))
    raise ComputePicMetricError(message)

  # Compute model prediction for the completely blurred image.
  fully_blurred_img_pred = model(torch.Tensor(fully_blurred_img).unsqueeze(0).unsqueeze(0)).item()

  blurred_images.append(fully_blurred_img)
  predictions.append(fully_blurred_img_pred)

  # If the entropy of the completely blurred image is higher or equal to the
  # entropy of the original image then the metric cannot be used for this
  # image. Don't include this image in the aggregated result.
  if fully_blurred_img_entropy >= original_img_entropy:
    message = (
        'The entropy in the completely blurred image is not lower than'
        ' the entropy in the original image. Catch the error and exclude this'
        ' image from evaluation. Blurred entropy: {}, original'
        ' entropy {}'.format(fully_blurred_img_entropy, original_img_entropy))
    raise ComputePicMetricError(message)

  # If the score of the model on completely blurred image is higher or equal to
  # the score of the model on the original image then the metric cannot be used
  # for this image. Don't include this image in the aggregated result.
  if fully_blurred_img_pred >= original_img_pred:
    message = (
        'The model prediction score on the completely blurred image is not'
        ' lower than the score on the original image. Catch the error and'
        ' exclude this image from the evaluation. Blurred score: {}, original'
        ' score {}'.format(fully_blurred_img_pred, original_img_pred))
    raise ComputePicMetricError(message)

  # Iterate through saliency thresholds and compute prediction of the model
  # for the corresponding blurred images with the saliency pixels revealed.
  max_normalized_pred = 0.0
  for threshold in saliency_thresholds:
    quantile = np.quantile(saliency_map, 1 - threshold)
    pixel_mask = saliency_map >= quantile
    pixel_mask = np.logical_or(pixel_mask, random_mask)
    blurred_image = create_blurred_image(full_img=img, pixel_mask=pixel_mask)
    entropy = estimate_image_entropy(blurred_image)
    pred = model(torch.Tensor(blurred_image).unsqueeze(0).unsqueeze(0))[0]
    # Normalize the values, so they lie in [0, 1] interval.
    normalized_entropy = (entropy - fully_blurred_img_entropy) / (
        original_img_entropy - fully_blurred_img_entropy)
    normalized_entropy = np.clip(normalized_entropy, 0.0, 1.0)
    normalized_pred = (pred - fully_blurred_img_pred) / (
        original_img_pred - fully_blurred_img_pred)
    normalized_pred = np.clip(normalized_pred, 0.0, 1.0)
    max_normalized_pred = max(max_normalized_pred, normalized_pred)

    # Make normalized_pred only grow if keep_monotonous is true.
    if keep_monotonous:
      entropy_pred_tuples.append((normalized_entropy, max_normalized_pred))
    else:
      entropy_pred_tuples.append((normalized_entropy, normalized_pred))

    blurred_images.append(blurred_image)
    predictions.append(pred)

  # Interpolate the PIC curve.
  entropy_pred_tuples.append((0.0, 0.0))
  entropy_pred_tuples.append((1.0, 1.0))

  entropy_data, pred_data = zip(*entropy_pred_tuples)
  interp_func = interpolate.interp1d(x=entropy_data, y=pred_data)

  curve_x = np.linspace(start=0.0, stop=1.0, num=num_data_points,
                        endpoint=False)
  curve_y = np.asarray([interp_func(x) for x in curve_x])

  curve_x = np.append(curve_x, 1.0)
  curve_y = np.append(curve_y, 1.0)

  auc = np.trapz(curve_y, curve_x)

  blurred_images.append(img)
  predictions.append(original_img_pred)

  thresholds = [0.0] + list(saliency_thresholds) + [1.0]

  return PicMetricResult(curve_x=curve_x, curve_y=curve_y,
                         blurred_images=blurred_images,
                         predictions=predictions, thresholds=thresholds,
                         auc=auc)


class AggregateMetricResult(NamedTuple):
  """Holds results of aggregate_individual_pic_results(...) method."""
  # x-axis coordinates of aggregated PIC curve data points.
  curve_x: Sequence[float]
  # y-axis coordinates of aggregated PIC curve data points.
  curve_y: Sequence[float]
  # Area under the curve.
  auc: float


def aggregate_individual_pic_results(
    compute_pic_metrics_results: List[PicMetricResult],
    method: str = 'median') -> AggregateMetricResult:

  if not compute_pic_metrics_results:
    raise ValueError('The list of results should have at least one element.')

  curve_ys = [r.curve_y for r in compute_pic_metrics_results]
  curve_ys = np.asarray(curve_ys)

  # Validate that x-axis points for all individual results are the same.
  curve_xs = [r.curve_x for r in compute_pic_metrics_results]
  curve_xs = np.asarray(curve_xs)
  _, counts = np.unique(curve_xs, axis=1, return_counts=True)
  if not np.all(counts == 1):
    raise ValueError('Individual results have different x-axis data points.')

  if method == 'mean':
    aggr_curve_y = np.mean(curve_ys, axis=0)
  elif method == 'median':
    aggr_curve_y = np.median(curve_ys, axis=0)
  else:
    raise ValueError('Unknown method {}.'.format(method))

  auc = np.trapz(aggr_curve_y, curve_xs[0])

  return AggregateMetricResult(curve_x=curve_xs[0], curve_y=aggr_curve_y,
                               auc=auc)