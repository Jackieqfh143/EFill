import random
from enum import Enum
import cv2
import numpy as np
import enum
from copy import deepcopy
from skimage import img_as_ubyte
from skimage.transform import rescale, resize


class ObjectMask():
    def __init__(self, mask):
        self.height, self.width = mask.shape
        (self.up, self.down), (self.left, self.right) = self._get_limits(mask)
        self.mask = mask[self.up:self.down, self.left:self.right].copy()

    @staticmethod
    def _get_limits(mask):
        def indicator_limits(indicator):
            lower = indicator.argmax()
            upper = len(indicator) - indicator[::-1].argmax()
            return lower, upper

        vertical_indicator = mask.any(axis=1)
        vertical_limits = indicator_limits(vertical_indicator)

        horizontal_indicator = mask.any(axis=0)
        horizontal_limits = indicator_limits(horizontal_indicator)

        return vertical_limits, horizontal_limits

    def _clean(self):
        self.up, self.down, self.left, self.right = 0, 0, 0, 0
        self.mask = np.empty((0, 0))

    def horizontal_flip(self, inplace=False):
        if not inplace:
            flipped = deepcopy(self)
            return flipped.horizontal_flip(inplace=True)

        self.mask = self.mask[:, ::-1]
        return self

    def vertical_flip(self, inplace=False):
        if not inplace:
            flipped = deepcopy(self)
            return flipped.vertical_flip(inplace=True)

        self.mask = self.mask[::-1, :]
        return self

    def image_center(self):
        y_center = self.up + (self.down - self.up) / 2
        x_center = self.left + (self.right - self.left) / 2
        return y_center, x_center

    def rescale(self, scaling_factor, inplace=False):
        if not inplace:
            scaled = deepcopy(self)
            return scaled.rescale(scaling_factor, inplace=True)

        scaled_mask = rescale(self.mask.astype(float), scaling_factor, order=0) > 0.5
        (up, down), (left, right) = self._get_limits(scaled_mask)
        self.mask = scaled_mask[up:down, left:right]

        y_center, x_center = self.image_center()
        mask_height, mask_width = self.mask.shape
        self.up = int(round(y_center - mask_height / 2))
        self.down = self.up + mask_height
        self.left = int(round(x_center - mask_width / 2))
        self.right = self.left + mask_width
        return self

    def crop_to_canvas(self, vertical=True, horizontal=True, inplace=False):
        if not inplace:
            cropped = deepcopy(self)
            cropped.crop_to_canvas(vertical=vertical, horizontal=horizontal, inplace=True)
            return cropped

        if vertical:
            if self.up >= self.height or self.down <= 0:
                self._clean()
            else:
                cut_up, cut_down = max(-self.up, 0), max(self.down - self.height, 0)
                if cut_up != 0:
                    self.mask = self.mask[cut_up:]
                    self.up = 0
                if cut_down != 0:
                    self.mask = self.mask[:-cut_down]
                    self.down = self.height

        if horizontal:
            if self.left >= self.width or self.right <= 0:
                self._clean()
            else:
                cut_left, cut_right = max(-self.left, 0), max(self.right - self.width, 0)
                if cut_left != 0:
                    self.mask = self.mask[:, cut_left:]
                    self.left = 0
                if cut_right != 0:
                    self.mask = self.mask[:, :-cut_right]
                    self.right = self.width

        return self

    def restore_full_mask(self, allow_crop=False):
        cropped = self.crop_to_canvas(inplace=allow_crop)
        mask = np.zeros((cropped.height, cropped.width), dtype=bool)
        mask[cropped.up:cropped.down, cropped.left:cropped.right] = cropped.mask
        return mask

    def shift(self, vertical=0, horizontal=0, inplace=False):
        if not inplace:
            shifted = deepcopy(self)
            return shifted.shift(vertical=vertical, horizontal=horizontal, inplace=True)

        self.up += vertical
        self.down += vertical
        self.left += horizontal
        self.right += horizontal
        return self

    def area(self):
        return self.mask.sum()

def upgrade_type(arr):
  dtype = arr.dtype

  if dtype == np.uint8:
    return arr.astype(np.uint16), True
  elif dtype == np.uint16:
    return arr.astype(np.uint32), True
  elif dtype == np.uint32:
    return arr.astype(np.uint64), True

  return arr, False

def zero_corrected_countless(data):
    """
    Vectorized implementation of downsampling a 2D
    image by 2 on each side using the COUNTLESS algorithm.

    data is a 2D numpy array with even dimensions.
    """
    # allows us to prevent losing 1/2 a bit of information
    # at the top end by using a bigger type. Without this 255 is handled incorrectly.
    data, upgraded = upgrade_type(data)

    # offset from zero, raw countless doesn't handle 0 correctly
    # we'll remove the extra 1 at the end.
    data += 1

    sections = []

    # This loop splits the 2D array apart into four arrays that are
    # all the result of striding by 2 and offset by (0,0), (0,1), (1,0),
    # and (1,1) representing the A, B, C, and D positions from Figure 1.
    factor = (2, 2)
    for offset in np.ndindex(factor):
        part = data[tuple(np.s_[o::f] for o, f in zip(offset, factor))]
        sections.append(part)

    a, b, c, d = sections

    ab = a * (a == b)  # PICK(A,B)
    ac = a * (a == c)  # PICK(A,C)
    bc = b * (b == c)  # PICK(B,C)

    a = ab | ac | bc  # Bitwise OR, safe b/c non-matches are zeroed

    result = a + (a == 0) * d - 1  # a or d - 1

    if upgraded:
        return downgrade_type(result)

    # only need to reset data if we weren't upgraded
    # b/c no copy was made in that case
    data -= 1

    return result

# from metrics.evaluation.masks.mask import SegmentationMask
try:
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    DETECTRON_INSTALLED = True
except:
    print("Detectron v2 is not installed")
    DETECTRON_INSTALLED = False

class RigidnessMode(enum.Enum):
    soft = 0
    rigid = 1

class SegmentationMask:
    def __init__(self, confidence_threshold=0.5, rigidness_mode=RigidnessMode.rigid,
                 max_object_area=0.3, min_mask_area=0.02, downsample_levels=6, num_variants_per_mask=4,
                 max_mask_intersection=0.5, max_foreground_coverage=0.5, max_foreground_intersection=0.5,
                 max_hidden_area=0.2, max_scale_change=0.25, horizontal_flip=True,
                 max_vertical_shift=0.1, position_shuffle=True):
        """
        :param confidence_threshold: float; threshold for confidence of the panoptic segmentator to allow for
        the instance.
        :param rigidness_mode: RigidnessMode object
            when soft, checks intersection only with the object from which the mask_object was produced
            when rigid, checks intersection with any foreground class object
        :param max_object_area: float; allowed upper bound for to be considered as mask_object.
        :param min_mask_area: float; lower bound for mask to be considered valid
        :param downsample_levels: int; defines width of the resized segmentation to obtain shifted masks;
        :param num_variants_per_mask: int; maximal number of the masks for the same object;
        :param max_mask_intersection: float; maximum allowed area fraction of intersection for 2 masks
        produced by horizontal shift of the same mask_object; higher value -> more diversity
        :param max_foreground_coverage: float; maximum allowed area fraction of intersection for foreground object to be
        covered by mask; lower value -> less the objects are covered
        :param max_foreground_intersection: float; maximum allowed area of intersection for the mask with foreground
        object; lower value -> mask is more on the background than on the objects
        :param max_hidden_area: upper bound on part of the object hidden by shifting object outside the screen area;
        :param max_scale_change: allowed scale change for the mask_object;
        :param horizontal_flip: if horizontal flips are allowed;
        :param max_vertical_shift: amount of vertical movement allowed;
        :param position_shuffle: shuffle
        """

        assert DETECTRON_INSTALLED, 'Cannot use SegmentationMask without detectron2'
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        self.cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
        self.predictor = DefaultPredictor(self.cfg)

        self.rigidness_mode = RigidnessMode(rigidness_mode)
        self.max_object_area = max_object_area
        self.min_mask_area = min_mask_area
        self.downsample_levels = downsample_levels
        self.num_variants_per_mask = num_variants_per_mask
        self.max_mask_intersection = max_mask_intersection
        self.max_foreground_coverage = max_foreground_coverage
        self.max_foreground_intersection = max_foreground_intersection
        self.max_hidden_area = max_hidden_area
        self.position_shuffle = position_shuffle

        self.max_scale_change = max_scale_change
        self.horizontal_flip = horizontal_flip
        self.max_vertical_shift = max_vertical_shift

    def get_segmentation(self, img):
        im = img_as_ubyte(img)
        panoptic_seg, segment_info = self.predictor(im)["panoptic_seg"]
        return panoptic_seg, segment_info

    @staticmethod
    def _is_power_of_two(n):
        return (n != 0) and (n & (n-1) == 0)

    def identify_candidates(self, panoptic_seg, segments_info):
        potential_mask_ids = []
        for segment in segments_info:
            if not segment["isthing"]:
                continue
            mask = (panoptic_seg == segment["id"]).int().detach().cpu().numpy()
            area = mask.sum().item() / np.prod(panoptic_seg.shape)
            if area >= self.max_object_area:
                continue
            potential_mask_ids.append(segment["id"])
        return potential_mask_ids

    def downsample_mask(self, mask):
        height, width = mask.shape
        if not (self._is_power_of_two(height) and self._is_power_of_two(width)):
            raise ValueError("Image sides are not power of 2.")

        num_iterations = width.bit_length() - 1 - self.downsample_levels
        if num_iterations < 0:
            raise ValueError(f"Width is lower than 2^{self.downsample_levels}.")

        if height.bit_length() - 1 < num_iterations:
            raise ValueError("Height is too low to perform downsampling")

        downsampled = mask
        for _ in range(num_iterations):
            downsampled = zero_corrected_countless(downsampled)

        return downsampled

    def _augmentation_params(self):
        scaling_factor = np.random.uniform(1 - self.max_scale_change, 1 + self.max_scale_change)
        if self.horizontal_flip:
            horizontal_flip = bool(np.random.choice(2))
        else:
            horizontal_flip = False
        vertical_shift = np.random.uniform(-self.max_vertical_shift, self.max_vertical_shift)

        return {
            "scaling_factor": scaling_factor,
            "horizontal_flip": horizontal_flip,
            "vertical_shift": vertical_shift
        }

    def _get_intersection(self, mask_array, mask_object):
        intersection = mask_array[
            mask_object.up:mask_object.down, mask_object.left:mask_object.right
        ] & mask_object.mask
        return intersection

    def _check_masks_intersection(self, aug_mask, total_mask_area, prev_masks):
        for existing_mask in prev_masks:
            intersection_area = self._get_intersection(existing_mask, aug_mask).sum()
            intersection_existing = intersection_area / existing_mask.sum()
            intersection_current = 1 - (aug_mask.area() - intersection_area) / total_mask_area
            if (intersection_existing > self.max_mask_intersection) or \
               (intersection_current > self.max_mask_intersection):
                return False
        return True

    def _check_foreground_intersection(self, aug_mask, foreground):
        for existing_mask in foreground:
            intersection_area = self._get_intersection(existing_mask, aug_mask).sum()
            intersection_existing = intersection_area / existing_mask.sum()
            if intersection_existing > self.max_foreground_coverage:
                return False
            intersection_mask = intersection_area / aug_mask.area()
            if intersection_mask > self.max_foreground_intersection:
                return False
        return True

    def _move_mask(self, mask, foreground):
        # Obtaining properties of the original mask_object:
        orig_mask = ObjectMask(mask)

        chosen_masks = []
        chosen_parameters = []
        # to fix the case when resizing gives mask_object consisting only of False
        scaling_factor_lower_bound = 0.

        for var_idx in range(self.num_variants_per_mask):
            # Obtaining augmentation parameters and applying them to the downscaled mask_object
            augmentation_params = self._augmentation_params()
            augmentation_params["scaling_factor"] = min([
                augmentation_params["scaling_factor"],
                2 * min(orig_mask.up, orig_mask.height - orig_mask.down) / orig_mask.height + 1.,
                2 * min(orig_mask.left, orig_mask.width - orig_mask.right) / orig_mask.width + 1.
            ])
            augmentation_params["scaling_factor"] = max([
                augmentation_params["scaling_factor"], scaling_factor_lower_bound
            ])

            aug_mask = deepcopy(orig_mask)
            aug_mask.rescale(augmentation_params["scaling_factor"], inplace=True)
            if augmentation_params["horizontal_flip"]:
                aug_mask.horizontal_flip(inplace=True)
            total_aug_area = aug_mask.area()
            if total_aug_area == 0:
                scaling_factor_lower_bound = 1.
                continue

            # Fix if the element vertical shift is too strong and shown area is too small:
            vertical_area = aug_mask.mask.sum(axis=1) / total_aug_area  # share of area taken by rows
            # number of rows which are allowed to be hidden from upper and lower parts of image respectively
            max_hidden_up = np.searchsorted(vertical_area.cumsum(), self.max_hidden_area)
            max_hidden_down = np.searchsorted(vertical_area[::-1].cumsum(), self.max_hidden_area)
            # correcting vertical shift, so not too much area will be hidden
            augmentation_params["vertical_shift"] = np.clip(
                augmentation_params["vertical_shift"],
                -(aug_mask.up + max_hidden_up) / aug_mask.height,
                (aug_mask.height - aug_mask.down + max_hidden_down) / aug_mask.height
            )
            # Applying vertical shift:
            vertical_shift = int(round(aug_mask.height * augmentation_params["vertical_shift"]))
            aug_mask.shift(vertical=vertical_shift, inplace=True)
            aug_mask.crop_to_canvas(vertical=True, horizontal=False, inplace=True)

            # Choosing horizontal shift:
            max_hidden_area = self.max_hidden_area - (1 - aug_mask.area() / total_aug_area)
            horizontal_area = aug_mask.mask.sum(axis=0) / total_aug_area
            max_hidden_left = np.searchsorted(horizontal_area.cumsum(), max_hidden_area)
            max_hidden_right = np.searchsorted(horizontal_area[::-1].cumsum(), max_hidden_area)
            allowed_shifts = np.arange(-max_hidden_left, aug_mask.width -
                                      (aug_mask.right - aug_mask.left) + max_hidden_right + 1)
            allowed_shifts = - (aug_mask.left - allowed_shifts)

            if self.position_shuffle:
                np.random.shuffle(allowed_shifts)

            mask_is_found = False
            for horizontal_shift in allowed_shifts:
                aug_mask_left = deepcopy(aug_mask)
                aug_mask_left.shift(horizontal=horizontal_shift, inplace=True)
                aug_mask_left.crop_to_canvas(inplace=True)

                prev_masks = [mask] + chosen_masks
                is_mask_suitable = self._check_masks_intersection(aug_mask_left, total_aug_area, prev_masks) & \
                                   self._check_foreground_intersection(aug_mask_left, foreground)
                if is_mask_suitable:
                    aug_draw = aug_mask_left.restore_full_mask()
                    chosen_masks.append(aug_draw)
                    augmentation_params["horizontal_shift"] = horizontal_shift / aug_mask_left.width
                    chosen_parameters.append(augmentation_params)
                    mask_is_found = True
                    break

            if not mask_is_found:
                break

        return chosen_parameters

    def _prepare_mask(self, mask):
        height, width = mask.shape
        target_width = width if self._is_power_of_two(width) else (1 << width.bit_length())
        target_height = height if self._is_power_of_two(height) else (1 << height.bit_length())

        return resize(mask.astype('float32'), (target_height, target_width), order=0, mode='edge').round().astype('int32')

    def get_masks(self, im, return_panoptic=False):
        panoptic_seg, segments_info = self.get_segmentation(im)
        potential_mask_ids = self.identify_candidates(panoptic_seg, segments_info)

        panoptic_seg_scaled = self._prepare_mask(panoptic_seg.detach().cpu().numpy())
        downsampled = self.downsample_mask(panoptic_seg_scaled)
        scene_objects = []
        for segment in segments_info:
            if not segment["isthing"]:
                continue
            mask = downsampled == segment["id"]
            if not np.any(mask):
                continue
            scene_objects.append(mask)

        mask_set = []
        for mask_id in potential_mask_ids:
            mask = downsampled == mask_id
            if not np.any(mask):
                continue

            if self.rigidness_mode is RigidnessMode.soft:
                foreground = [mask]
            elif self.rigidness_mode is RigidnessMode.rigid:
                foreground = scene_objects
            else:
                raise ValueError(f'Unexpected rigidness_mode: {rigidness_mode}')

            masks_params = self._move_mask(mask, foreground)

            full_mask = ObjectMask((panoptic_seg == mask_id).detach().cpu().numpy())

            for params in masks_params:
                aug_mask = deepcopy(full_mask)
                aug_mask.rescale(params["scaling_factor"], inplace=True)
                if params["horizontal_flip"]:
                    aug_mask.horizontal_flip(inplace=True)

                vertical_shift = int(round(aug_mask.height * params["vertical_shift"]))
                horizontal_shift = int(round(aug_mask.width * params["horizontal_shift"]))
                aug_mask.shift(vertical=vertical_shift, horizontal=horizontal_shift, inplace=True)
                aug_mask = aug_mask.restore_full_mask().astype('uint8')
                if aug_mask.mean() <= self.min_mask_area:
                    continue
                mask_set.append(aug_mask)

        if return_panoptic:
            return mask_set, panoptic_seg.detach().cpu().numpy()
        else:
            return mask_set


class LinearRamp:
    def __init__(self, start_value=0, end_value=1, start_iter=-1, end_iter=0):
        self.start_value = start_value
        self.end_value = end_value
        self.start_iter = start_iter
        self.end_iter = end_iter

    def __call__(self, i):
        if i < self.start_iter:
            return self.start_value
        if i >= self.end_iter:
            return self.end_value
        part = (i - self.start_iter) / (self.end_iter - self.start_iter)
        return self.start_value * (1 - part) + self.end_value * part


# LOGGER = logging.getLogger(__name__)


class DrawMethod(Enum):
    LINE = 'line'
    CIRCLE = 'circle'
    SQUARE = 'square'


def make_random_irregular_mask(shape, max_angle=4, max_len=60, max_width=20, min_times=0, max_times=10,
                               draw_method=DrawMethod.LINE):
    draw_method = DrawMethod(draw_method)

    height, width = shape
    mask = np.zeros((height, width), np.float32)
    times = np.random.randint(min_times, max_times + 1)
    for i in range(times):
        start_x = np.random.randint(width)
        start_y = np.random.randint(height)
        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(max_angle)
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = 10 + np.random.randint(max_len)
            brush_w = 5 + np.random.randint(max_width)
            end_x = np.clip((start_x + length * np.sin(angle)).astype(np.int32), 0, width)
            end_y = np.clip((start_y + length * np.cos(angle)).astype(np.int32), 0, height)
            if draw_method == DrawMethod.LINE:
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, brush_w)
            elif draw_method == DrawMethod.CIRCLE:
                cv2.circle(mask, (start_x, start_y), radius=brush_w, color=1., thickness=-1)
            elif draw_method == DrawMethod.SQUARE:
                radius = brush_w // 2
                mask[start_y - radius:start_y + radius, start_x - radius:start_x + radius] = 1
            start_x, start_y = end_x, end_y
    return mask[None, ...]


class RandomIrregularMaskGenerator:
    def __init__(self, max_angle=4, max_len=60, max_width=20, min_times=0, max_times=10, ramp_kwargs=None,
                 draw_method=DrawMethod.LINE):
        self.max_angle = max_angle
        self.max_len = max_len
        self.max_width = max_width
        self.min_times = min_times
        self.max_times = max_times
        self.draw_method = draw_method
        self.ramp = LinearRamp(**ramp_kwargs) if ramp_kwargs is not None else None

    def __call__(self, shape, iter_i=None, raw_image=None):
        coef = self.ramp(iter_i) if (self.ramp is not None) and (iter_i is not None) else 1
        cur_max_len = int(max(1, self.max_len * coef))
        cur_max_width = int(max(1, self.max_width * coef))
        cur_max_times = int(self.min_times + 1 + (self.max_times - self.min_times) * coef)
        return make_random_irregular_mask(shape, max_angle=self.max_angle, max_len=cur_max_len,
                                          max_width=cur_max_width, min_times=self.min_times, max_times=cur_max_times,
                                          draw_method=self.draw_method)


def make_random_rectangle_mask(shape, margin=10, bbox_min_size=30, bbox_max_size=100, min_times=0, max_times=3):
    height, width = shape
    mask = np.zeros((height, width), np.float32)
    bbox_max_size = min(bbox_max_size, height - margin * 2, width - margin * 2)
    times = np.random.randint(min_times, max_times + 1)
    for i in range(times):
        box_width = np.random.randint(bbox_min_size, bbox_max_size)
        box_height = np.random.randint(bbox_min_size, bbox_max_size)
        start_x = np.random.randint(margin, width - margin - box_width + 1)
        start_y = np.random.randint(margin, height - margin - box_height + 1)
        mask[start_y:start_y + box_height, start_x:start_x + box_width] = 1
    return mask[None, ...]


class RandomRectangleMaskGenerator:
    def __init__(self, margin=10, bbox_min_size=30, bbox_max_size=100, min_times=0, max_times=3, ramp_kwargs=None):
        self.margin = margin
        self.bbox_min_size = bbox_min_size
        self.bbox_max_size = bbox_max_size
        self.min_times = min_times
        self.max_times = max_times
        self.ramp = LinearRamp(**ramp_kwargs) if ramp_kwargs is not None else None

    def __call__(self, shape, iter_i=None, raw_image=None):
        coef = self.ramp(iter_i) if (self.ramp is not None) and (iter_i is not None) else 1
        cur_bbox_max_size = int(self.bbox_min_size + 1 + (self.bbox_max_size - self.bbox_min_size) * coef)
        cur_max_times = int(self.min_times + (self.max_times - self.min_times) * coef)
        return make_random_rectangle_mask(shape, margin=self.margin, bbox_min_size=self.bbox_min_size,
                                          bbox_max_size=cur_bbox_max_size, min_times=self.min_times,
                                          max_times=cur_max_times)


def make_random_superres_mask(shape, min_step=2, max_step=4, min_width=1, max_width=3):
    height, width = shape
    mask = np.zeros((height, width), np.float32)
    step_x = np.random.randint(min_step, max_step + 1)
    width_x = np.random.randint(min_width, min(step_x, max_width + 1))
    offset_x = np.random.randint(0, step_x)

    step_y = np.random.randint(min_step, max_step + 1)
    width_y = np.random.randint(min_width, min(step_y, max_width + 1))
    offset_y = np.random.randint(0, step_y)

    for dy in range(width_y):
        mask[offset_y + dy::step_y] = 1
    for dx in range(width_x):
        mask[:, offset_x + dx::step_x] = 1
    return mask[None, ...]


class RandomSuperresMaskGenerator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, shape, iter_i=None):
        return make_random_superres_mask(shape, **self.kwargs)


class MixedMaskGenerator:
    def __init__(self, irregular_proba=1 / 3, hole_range=[0.0, 0.7], irregular_kwargs=None,
                 box_proba=1 / 3, box_kwargs=None,
                 segm_proba=1 / 3, segm_kwargs=None,
                 squares_proba=0, squares_kwargs=None,
                 superres_proba=0, superres_kwargs=None,
                 outpainting_proba=0, outpainting_kwargs=None,
                 invert_proba=0):
        self.probas = []
        self.gens = []
        self.hole_range = hole_range

        if irregular_proba > 0:
            self.probas.append(irregular_proba)
            if irregular_kwargs is None:
                irregular_kwargs = {}
            else:
                irregular_kwargs = dict(irregular_kwargs)
            irregular_kwargs['draw_method'] = DrawMethod.LINE
            self.gens.append(RandomIrregularMaskGenerator(**irregular_kwargs))

        if box_proba > 0:
            self.probas.append(box_proba)
            if box_kwargs is None:
                box_kwargs = {}
            self.gens.append(RandomRectangleMaskGenerator(**box_kwargs))

        if squares_proba > 0:
            self.probas.append(squares_proba)
            if squares_kwargs is None:
                squares_kwargs = {}
            else:
                squares_kwargs = dict(squares_kwargs)
            squares_kwargs['draw_method'] = DrawMethod.SQUARE
            self.gens.append(RandomIrregularMaskGenerator(**squares_kwargs))

        if superres_proba > 0:
            self.probas.append(superres_proba)
            if superres_kwargs is None:
                superres_kwargs = {}
            self.gens.append(RandomSuperresMaskGenerator(**superres_kwargs))

        self.probas = np.array(self.probas, dtype='float32')
        self.probas /= self.probas.sum()
        self.invert_proba = invert_proba

    #shape is mask size (h x w)
    def __call__(self, shape, iter_i=None, raw_image=None):
        kind = np.random.choice(len(self.probas), p=self.probas)
        gen = self.gens[kind]
        result = gen(shape, iter_i=iter_i, raw_image=raw_image)
        if self.invert_proba > 0 and random.random() < self.invert_proba:
            result = 1 - result
        if np.mean(result) <= self.hole_range[0] or np.mean(result) >= self.hole_range[1]:
            return self.__call__(shape, iter_i=iter_i, raw_image=raw_image)
        else:
            return result[0]


class RandomSegmentationMaskGenerator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.impl = SegmentationMask(**self.kwargs)

    def __call__(self, img, iter_i=None, raw_image=None, hole_range=[0.0, 0.3]):

        masks = self.impl.get_masks(img)
        fil_masks = []
        for cur_mask in masks:
            if len(np.unique(cur_mask)) == 0 or cur_mask.mean() > hole_range[1]:
                continue
            fil_masks.append(cur_mask)

        mask_index = np.random.choice(len(fil_masks),
                                      size=1,
                                      replace=False)[0]
        mask = fil_masks[mask_index]

        return mask


class SegMaskGenerator:
    def __init__(self, hole_range=[0.1, 0.2], segm_kwargs=None):
        if segm_kwargs is None:
            segm_kwargs = {}
        self.gen = RandomSegmentationMaskGenerator(**segm_kwargs)
        self.hole_range = hole_range

    def __call__(self, img, iter_i=None, raw_image=None):
        result = self.gen(img=img, iter_i=iter_i, raw_image=raw_image, hole_range=self.hole_range)
        return result


class FGSegmentationMaskGenerator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.impl = SegmentationMask(**self.kwargs)

    def __call__(self, img, iter_i=None, raw_image=None, hole_range=[0.0, 0.3]):

        masks = self.impl.get_masks(img)
        mask = masks[0]
        for cur_mask in masks:
            if len(np.unique(cur_mask)) == 0 or cur_mask.mean() > hole_range[1]:
                continue
            mask += cur_mask

        mask = mask > 0
        return mask


class SegBGMaskGenerator:
    def __init__(self, hole_range=[1, 2], segm_kwargs=None):
        if segm_kwargs is None:
            segm_kwargs = {}
        self.gen = FGSegmentationMaskGenerator(**segm_kwargs)
        self.hole_range = hole_range
        self.cfg = {
            'irregular_proba': 1,
            'hole_range': [0.0, 1.0],
            'irregular_kwargs': {
                'max_angle': 4,
                'max_len': 250,
                'max_width': 150,
                'max_times': 3,
                'min_times': 1,
            },
            'box_proba': 0,
            'box_kwargs': {
                'margin': 10,
                'bbox_min_size': 30,
                'bbox_max_size': 150,
                'max_times': 4,
                'min_times': 1,
            }
        }
        self.bg_mask_gen = MixedMaskGenerator(**self.cfg)

    def __call__(self, img, iter_i=None, raw_image=None):
        shape = img.shape[:2]
        mask_fg = self.gen(img=img, iter_i=iter_i, raw_image=raw_image, hole_range=self.hole_range)
        bg_ratio = 1 - np.mean(mask_fg)
        result = self.bg_mask_gen(shape, iter_i=iter_i, raw_image=raw_image)
        result = result - mask_fg
        # if np.mean(result) <= self.hole_range[0] * bg_ratio or np.mean(result) >= self.hole_range[1] * bg_ratio:
        #     return self.__call__(img, iter_i=iter_i, raw_image=raw_image)
        return result

class MySegMaskGen():
    def __init__(self,hole_range=[0.0,0.7],max_obj_area = 0.4,max_obj_num = 3,confidence_threshold=0.5):
        assert DETECTRON_INSTALLED, 'Cannot use SegmentationMask without detectron2'
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        self.cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
        self.predictor = DefaultPredictor(self.cfg)
        self.hole_range = hole_range
        self.max_obj_area = max_obj_area
        self.max_obj_num = max_obj_num
        bg_cfg = {
            'irregular_proba': 1,
            'hole_range': [0.0, 0.5],
            'irregular_kwargs': {
                'max_angle': 3,
                'max_len': 200,
                'max_width': 100,
                'max_times': 3,
                'min_times': 1,
            },
            'box_proba': 0.3,
            'box_kwargs': {
                'margin': 10,
                'bbox_min_size': 30,
                'bbox_max_size': 150,
                'max_times': 3,
                'min_times': 1,
            }
        }
        self.bg_mask_gen = MixedMaskGenerator(**bg_cfg)

    def get_segmentation(self, img):
        im = img_as_ubyte(img)
        panoptic_seg, segment_info = self.predictor(im)["panoptic_seg"]
        return panoptic_seg, segment_info

    def identify_candidates(self, panoptic_seg, segments_info):
        potential_mask_ids = []
        for segment in segments_info:
            if not segment["isthing"]:
                continue
            mask = (panoptic_seg == segment["id"]).int().detach().cpu().numpy()
            area = mask.sum().item() / np.prod(panoptic_seg.shape)
            if area >= self.max_obj_area:
                continue
            potential_mask_ids.append(segment["id"])
        return potential_mask_ids

    def get_mask(self,im,single_obj=False,objRemoval=True):
        mask_set = []
        try:
            panoptic_seg, segments_info = self.get_segmentation(im)
        except Exception as e:
            pass
        else:
            potential_mask_ids = self.identify_candidates(panoptic_seg, segments_info)
            for mask_id in potential_mask_ids:
                mask = (panoptic_seg == mask_id).int().detach().cpu().numpy()
                if not np.any(mask):
                    continue

                mask = mask.astype(np.uint8)
                if objRemoval:
                    mask = self.dilated_mask(mask)  #make sure the mask cover the whole object
                mask_set.append(mask)

        if len(mask_set) != 0:
            all_fg_mask = np.zeros(shape=im.shape[:2])
            fg_mask = np.zeros(shape=im.shape[:2])
            if single_obj:
                np.random.shuffle(mask_set)
                for idx in range(len(mask_set)):
                    fg_mask = mask_set[idx]
                    if np.mean(fg_mask) >= self.max_obj_area:
                        continue
                    else:
                        break
            else:
                np.random.shuffle(mask_set)
                mask = np.zeros(shape=im.shape[:2])
                for m in mask_set:
                    all_fg_mask += m
                for i,m in enumerate(mask_set):
                    area = np.mean(mask + m)
                    if area <= self.max_obj_area or i <=self.max_obj_num:
                        mask += m
                    else:
                        break

                fg_mask = mask

            fg_mask = fg_mask > 0
            if objRemoval:
                return fg_mask.astype(np.uint8)
            else:
                mask = self.get_bg_mask(all_fg_mask)
                while np.mean(mask) < self.hole_range[0] or np.mean(mask) > self.hole_range[1]:
                    mask = self.get_bg_mask(all_fg_mask)

                return mask
        else:
            mask = self.bg_mask_gen(shape=im.shape[:2])
            mask = (mask > 0).astype(np.uint8)
            while np.mean(mask) < self.hole_range[0] or np.mean(mask) > self.hole_range[1]:
                mask = self.bg_mask_gen(shape=im.shape[:2])
                mask = (mask > 0).astype(np.uint8)
            return mask

    def get_bg_mask(self,fg_mask):
        bg_mask = self.bg_mask_gen(shape=fg_mask.shape)
        bg_mask = (bg_mask > 0).astype(np.uint8)
        fg_mask = (fg_mask >0).astype(np.uint8)
        mask = bg_mask + fg_mask
        mask = (mask > 0).astype(np.uint8)
        mask = mask - fg_mask
        mask = (mask > 0).astype(np.uint8)
        return mask

    def dilated_mask(self,mask):
        kernel = np.ones((3, 3), np.uint8)
        iter = np.random.randint(5, 10)
        mask = cv2.dilate(mask, kernel, iterations=iter)  # generate more aggressive mask
        return mask

    def __call__(self, im):
        if np.random.binomial(1, 0.5) > 0:
            objRemoval = True
            if np.random.binomial(1, 0.5) > 0:
                single_obj = True
            else:
                single_obj = False
        else:
            objRemoval = False
            single_obj = False

        mask = self.get_mask(im,single_obj=single_obj,objRemoval=objRemoval)

        return mask

from data.lama_mask_cfg import *
def get_mask_generator(kind,**kwargs):
    if kind == 'medium_256':
        cfg = medium_256
    elif kind == 'medium_512':
        cfg = medium_512
    elif kind == 'thick_256':
        cfg = thick_256
    elif kind == 'thick_512':
        cfg = thick_512
    elif kind == 'thin_256':
        cfg = thin_256
    elif kind == 'thin_512':
        cfg = thin_512
    elif kind == 'segm_256':
        cfg = segm_256
    elif kind == 'segm_512':
        cfg = segm_512
    else:
        cfg = {
                'irregular_proba': 1,
                'hole_range': [0.0, 0.7],
                'irregular_kwargs': {
                    'max_angle': 4,
                    'max_len': 200,
                    'max_width': 100,
                    'max_times': 5,
                    'min_times': 1,
                },
                'box_proba': 1,
                'box_kwargs': {
                    'margin': 10,
                    'bbox_min_size': 30,
                    'bbox_max_size': 150,
                    'max_times': 4,
                    'min_times': 1,
                },
                'segm_proba': 0, }

    if not 'segm' in kind:
        return MixedMaskGenerator(**cfg)
    else:
        return MySegMaskGen(**kwargs)


if __name__ == '__main__':
    from PIL import Image
    img = Image.open('3.jpg')
    # img.show()
    seg_mask_gen = get_mask_generator(kind='default')
    mask = seg_mask_gen(shape=(256,256))
    mask = np.expand_dims(mask,axis=-1)
    # masked_img = img * (1 - mask)
    mask = np.concatenate([mask,mask,mask],axis=-1)
    mask = (mask * 255).astype(np.uint8)
    Image.fromarray(mask).show()
    # masked_img = masked_img.astype(np.uint8)
    # Image.fromarray(masked_img).show()
    # test_mask = Image.open('test_mask.png')
    # mask = np.array(test_mask)
    # mean = np.mean(mask)
    # sum = np.sum(mask)
    # print(mean)
    # img = np.array(Image.open('./3.jpg'))
    # mask_gen = get_mask_generator(kind='medium_256')
    # mask = mask_gen(shape=(256,256))
    # mask = np.expand_dims(mask, axis=2)
    # mask = (mask > 0).astype(np.uint8)
    # masked_img = img * (1 - mask) + mask * 255
    # comp_img = img * mask
    # mask = np.concatenate([mask, mask, mask], axis=2)
    # mask = mask * 255
    # Image.fromarray(mask).show()
    # Image.fromarray(comp_img).show()
    # Image.fromarray(masked_img).show()


