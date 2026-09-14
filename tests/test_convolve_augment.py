from astro_datatools.augment.convolve import ConvolveAugment
import numpy as np

def test_convolve_augment_2d():
    image = np.random.rand(10, 10)
    psf_kernel = np.random.rand(3, 3)
    augment = ConvolveAugment(psf_kernel)
    convolved = augment.augment(image)
    assert convolved.shape == image.shape

def test_convolve_augment_3d(kernel_2d=True):
    image = np.random.rand(3, 10, 10)
    if kernel_2d:
        psf_kernel = np.random.rand(3, 3)
    else:
        psf_kernel = np.random.rand(3, 3, 3)
    augment = ConvolveAugment(psf_kernel)
    convolved = augment.augment(image)
    assert convolved.shape == image.shape

def test_convolve_augment_batched(kernel_2d=True):
    image = np.random.rand(5, 3, 10, 10)
    if kernel_2d:
        psf_kernel = np.random.rand(3, 3)
    else:
        psf_kernel = np.random.rand(3, 3, 3)
    augment = ConvolveAugment(psf_kernel)
    convolved = augment.augment(image)
    assert convolved.shape == image.shape

def test_batched_faster_than_single(kernel_2d=True):
    batch_size = 500
    batched_image = np.random.rand(batch_size, 3, 10, 10)
    if kernel_2d:
        psf_kernel = np.random.rand(3, 3)
    else:
        psf_kernel = np.random.rand(3, 3, 3)

    batched_augment = ConvolveAugment(psf_kernel)
    single_augment = ConvolveAugment(psf_kernel)

    import time
    start = time.perf_counter()
    batched_convolved = batched_augment.augment(batched_image)
    batched_time = time.perf_counter() - start

    start = time.perf_counter()
    single_convolved = np.array([
        single_augment.augment(image)
        for image in batched_image
    ])
    simple_for_loop_time = time.perf_counter() - start

    if kernel_2d:
        # Collapse channel and batch
        start = time.perf_counter()
        batched_image_collapsed = batched_image.reshape(-1, 10, 10)
        warning_premise = batched_augment.augment(batched_image_collapsed)
        warning_premise_time = time.perf_counter() - start

    if kernel_2d:
        print("Using 2D kernel")
    else:
        print("Using 3D kernel")
    print(f"Batched time: {batched_time}")
    if kernel_2d:
        print(f"Collapsed channel and batch dim time: {warning_premise_time}")
    print(f"Simple for loop time: {simple_for_loop_time}")

    assert np.allclose(batched_convolved, single_convolved)
    assert batched_convolved.shape == (batch_size, 3, 10, 10)
    assert single_convolved.shape == (batch_size, 3, 10, 10)

def test_batched_matches_single(kernel_2d=True):
    batch_size = 500
    batched_image = np.random.rand(batch_size, 3, 10, 10)

    psf_kernel = (
        np.random.rand(3, 3)
        if kernel_2d
        else np.random.rand(3, 3, 3)
    )

    batched_augment = ConvolveAugment(psf_kernel)
    single_augment = ConvolveAugment(psf_kernel)

    batched_convolved = batched_augment.augment(batched_image)

    single_convolved = np.array([
        single_augment.augment(image)
        for image in batched_image
    ])

    assert np.allclose(batched_convolved, single_convolved)
    assert batched_convolved.shape == (batch_size, 3, 10, 10)

def run_all_test_with_3d_kernels():
    test_convolve_augment_3d(kernel_2d=False)
    test_convolve_augment_batched(kernel_2d=False)
    test_batched_faster_than_single(kernel_2d=False)

if __name__ == "__main__":
    test_convolve_augment_2d()
    test_convolve_augment_3d()
    test_convolve_augment_batched()
    test_batched_faster_than_single()
    test_batched_matches_single()
    run_all_test_with_3d_kernels()