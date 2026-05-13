from core.preprocessing import decode_image, preprocess_image


def test_preprocess_image_returns_chw_tensor(sample_image_bytes):
    tensor = preprocess_image(sample_image_bytes, image_size=256, require_face=False)
    assert tuple(tensor.shape) == (3, 256, 256)
    assert tensor.dtype.is_floating_point


def test_decode_image_rgb(sample_image_bytes):
    image = decode_image(sample_image_bytes)
    assert image.mode == "RGB"
