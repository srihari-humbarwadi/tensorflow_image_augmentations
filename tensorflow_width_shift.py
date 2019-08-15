import tensorflow as tf
import tensorflow_probability as tfp

print('TensorFlow', tf.__version__)


@tf.function
def random_width_shit(image, max_delta=0.2, pad_value=0):
    image_width = tf.shape(image)[1]
    random_number = tf.random.uniform((), minval=0, maxval=max_delta)
    delta = tf.cast(tf.cast(image_width, dtype=tf.float32)
                    * random_number, dtype=tf.int32)
    if tf.random.uniform(()) > 0.5:
        padding = [[0, 0], [delta, 0], [0, 0]]
        shifted_image = tf.pad(image, padding)
        shifted_image = shifted_image[:, :image_width, :]
    else:
        padding = [[0, 0], [0, delta], [0, 0]]
        shifted_image = tf.pad(image, padding, constant_values=pad_value)
        shifted_image = shifted_image[:, delta:, :]
    return shifted_image


@tf.function
def random_height_shit(image, max_delta=0.2, pad_value=0):
    image_height = tf.shape(image)[0]
    random_number = tf.random.uniform((), minval=0, maxval=max_delta)
    delta = tf.cast(tf.cast(image_height, dtype=tf.float32)
                    * random_number, dtype=tf.int32)
    if tf.random.uniform(()) > 0.5:
        padding = [[delta, 0], [0, 0], [0, 0]]
        shifted_image = tf.pad(image, padding, constant_values=pad_value)
        shifted_image = shifted_image[:image_height, :, :]
    else:
        padding = [[0, delta], [0, 0], [0, 0]]
        shifted_image = tf.pad(image, padding)
        shifted_image = shifted_image[delta:, :, :]
    return shifted_image


@tf.function
def random_scale(image, min_scale=0.5, max_scale=2, method='bilinear'):
    scale = tf.random.uniform((), minval=min_scale, maxval=max_scale)
    dims = tf.cast(tf.shape(image), dtype=tf.float32)
    scaled_height = tf.cast(dims[0] * scale, dtype=tf.int32)
    scaled_width = tf.cast(dims[1] * scale, dtype=tf.int32)
    scaled_image = tf.image.resize(
        image, size=[scaled_height, scaled_width], method=method)
    scaled_image = tf.cast(scaled_image, dtype=image.dtype)
    scaled_image.set_shape([None, None, 3])
    return scaled_image


@tf.function
def random_crop(image, crop_height=256, crop_width=256):
    dims = tf.shape(image)
    tf.assert_less(crop_height, dims[0])
    tf.assert_less(crop_width, dims[1])
    offset_x = tf.random.uniform(
        (), maxval=dims[1] - crop_width, dtype=tf.int32)
    offset_y = tf.random.uniform(
        (), maxval=dims[0] - crop_height, dtype=tf.int32)
    cropped_image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, crop_height, crop_width)
    return cropped_image


@tf.function
def cut_out(image, n_holes=2, length=40):
    dims = tf.shape(image)
    x_min = tf.random.uniform(
        (n_holes,), maxval=dims[1] - length, dtype=tf.int32)
    y_min = tf.random.uniform(
        (n_holes,), maxval=dims[0] - length, dtype=tf.int32)
    x_max = tf.clip_by_value(x_min + length, 0, dims[1])
    y_max = tf.clip_by_value(y_min + length, 0, dims[0])
    blank_square = tf.zeros(shape=[length, length, 1], dtype=image.dtype)

    pad_right = dims[1] - x_max
    pad_left = x_min
    pad_bottom = dims[0] - y_max
    pad_top = y_min
    paddings = tf.stack([pad_right, pad_left, pad_top, pad_bottom], axis=-1)

    def pad_tensor(padding):
        right = padding[0]
        left = padding[1]
        top = padding[2]
        bottom = padding[3]
        mask = tf.pad(blank_square, [[top, bottom], [
                      left, right], [0, 0]], constant_values=1)
        return mask

    masks = tf.map_fn(pad_tensor, paddings,
                      parallel_iterations=16, dtype=image.dtype)
    cut_out_mask = tf.cast(tf.reduce_prod(masks, axis=0), dtype=image.dtype)
    cut_out_image = image * cut_out_mask
    return cut_out_image


@tf.function
def random_center_crop(image, target_size=256):
    dims = tf.shape(image)[:2]
    min_dim = tf.reduce_min(dims)
    c_x = tf.cast(dims[1] // 2, dtype=tf.int32)
    c_y = tf.cast(dims[0] // 2, dtype=tf.int32)
    min_val = tf.cast(target_size - (target_size % 2), dtype=tf.int32)
    max_val = tf.cast(min_dim - (target_size % 2), dtype=tf.int32)
    random_size = tf.random.uniform(
        (), minval=min_val, maxval=max_val, dtype=tf.int32)
    offset_x = c_x - tf.cast(random_size / 2, dtype=tf.int32)
    offset_y = c_y - tf.cast(random_size / 2, dtype=tf.int32)
    cropped_image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, random_size, random_size)
    return cropped_image
    resized_image = tf.cast(tf.image.resize(cropped_image, size=[
                            target_size, target_size], method='bilinear'),
                            dtype=image.dtype)
    return resized_image


@tf.function
def random_channel_shuffle(image):
    tf.assert_equal(tf.rank(image), 3)
    tf.assert_equal(tf.shape(image)[-1], 3)
    indices = tf.random.shuffle(tf.range(3, dtype=tf.int32))
    shuffled_image = tf.stack([image[:, :, indices[0]],
                               image[:, :, indices[1]],
                               image[:, :, indices[2]]], axis=-1)
    return shuffled_image


@tf.function
def mix_up(images, labels, alpha=0.4, stack_labels=True):
    batch_size = tf.shape(images)[0]
    t = tfp.distributions.Beta(alpha, alpha).sample(batch_size)
    t = tf.reduce_max(t, 1 - t, axis=-1)
    indices = tf.random.shuffle(tf.range(batch_size))
    mixed_images = t * images + (1 - t) * tf.gather(images, indices)
    mixed_labels = t * labels + (1 - t) * tf.gather(labels, indices)
    if stack_labels:
        labels = tf.stack([labels, mixed_labels], axis=-1)
    return mixed_images, labels


@tf.function
def mix_up_loss(loss_fn, y_pred, labels, stacked_labels=True, t=None):
    if stacked_labels:
        labels, mixed_labels = tf.split(labels, num_or_size_splits=2, axis=-1)
        return t * loss_fn(labels, y_pred)
        + (1 - t) * loss_fn(mixed_labels, y_pred)
    return loss_fn(labels, y_pred)
